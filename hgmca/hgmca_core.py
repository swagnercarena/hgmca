from hgmca import helpers, gmca_core, wavelets_hgmca
import numpy as np
import numba
from tqdm import tqdm
import time, os


@numba.njit
def allocate_A_hier(m_level,A_shape,A_init=None):
	"""Allocates the hierarchy of mixing matrices for HGMCA analysis

	Parameters:
		m_level (int): The deepest level of analysis to consider
		A_shape (tuple): The shape of the mixing matrix. Should be
			(n_freqs,n_sources).
		A_init (np.array): A value at which to initialize all of the
			matrices in the hierarchy. If None then matrices will be
			initialized to 1/np.sqrt(n_freqs)

	Returns
		[np.array,...]: A numba.typed.List of numpy arrays containing the
		mixing matrix hierarchy for each level of analysis.
	"""
	# Get the initialization value
	if A_init is None:
		A = np.ones(A_shape)/np.sqrt(A_shape[0])
	else:
		A = A_init

	# Initialize all of the matrices
	A_hier_list = numba.typed.List()
	for level in range(m_level+1):
		npatches = wavelets_hgmca.level_to_npatches(level)
		# Initialize the array and then set it to the desired value.
		A_hier = np.zeros((npatches,)+A_shape)
		for patch in range(npatches):
			A_hier[patch] += A
		A_hier_list.append(A_hier)

	return A_hier_list


@numba.njit
def allocate_S_level(m_level,X_level,n_sources):
	"""Allocates the S matrices for all the levels of analysis.

	Parameters:
		m_level (int): The deepest level of analysis to consider
		X_level ([np.array,...]): A numba.typed.List of numpy arrays
			corresponding to the data for each level of analysis.
		n_sources (int): The number of sources

	Returns
		[np.array,...]: A numba.typed.List of numpy arrays containing the
		source matrices for each level of analysis.
	"""
	# Initialize our list
	S_level = numba.typed.List()
	for level in range(m_level+1):
		# If there is no data at that level, initialize a 0x0 array.
		if X_level[level].size == 0:
			S_level.append(np.empty((0,0,0)))
			continue
		# Otherwise initialize an array of zeros.
		npatches = wavelets_hgmca.level_to_npatches(level)
		n_wavs = X_level[level].shape[2]
		S_level.append(np.zeros((npatches,n_sources,n_wavs)))

	return S_level


def convert_wav_to_X_level(wav_analysis_maps):
	"""Converts wav_analysis_maps output from the WaveletsHGMCA class to
	the X_level numba.typed.List required by HGMCA.

	Parameters:
		wav_analysis_maps (dict): A dictionary containing the information
				about the wavelet functions used for analysis.

	Returns
		[np.array,...]: A numba.typed.List of numpy arrays corresponding to
		the data for each level of analysis.
	"""
	# Iterate through each level and feed it into our numba.typed.List
	X_level = numba.typed.List()
	for level in range(wav_analysis_maps['m_level']+1):
		if wav_analysis_maps[str(level)] is None:
			X_level.append(np.empty((0,0,0)))
			continue
		X_level.append(wav_analysis_maps[str(level)])
	return X_level


@numba.njit
def get_A_prior(A_hier_list,level,patch,lam_hier):
	"""Returns the prior on the mixing matrix at the specified level
	and patch.

	Parameters:
		A_hier_list ([np.array]): A numba.typed.List of numpy arrays containing
			the mixing matrix hierarchy for each level of analysis.
		level (int): The level of analysis
		patch (int): The patch at the given level
		lam_hier (np.array): A n_sources long array of the prior for each of
			the columns of the mixing matrices in A_hier.

	Returns:
		(np.array): A two dimensional numpy array containing the mixing
		matrix prior multiplied by the correct lamda (taking into account
		how many connections exist in the hierarchy).
	"""
	# Initialize our mixing matrix prior
	A_prior = np.zeros(A_hier_list[0][0].shape)
	# Add all the mixing matrices below. There are three cases, level 0,
	# level > 0 but less than m_level, and m_level that has no matrices
	# below
	if level == 0:
		for prior_patch in range(12):
			A_prior += lam_hier * A_hier_list[1][prior_patch]
	elif level < len(A_hier_list)-1:
		for prior_patch in range(4*patch,4*(patch+1)):
			A_prior += lam_hier * A_hier_list[level+1][prior_patch]

	# Now add all the mixing matrices above. There are three cases, level
	# 0 that has none, level 1 that has only 1, and level > 1.
	if level == 1:
		A_prior += lam_hier * A_hier_list[level-1][0]
	elif level>0:
		A_prior += lam_hier * A_hier_list[level-1][patch//4]

	return A_prior


@numba.njit
def hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,lam_global,S_level,
	n_epochs,m_level,n_iterations,lam_s,seed,enforce_nn_A,min_rmse_rate,
	epoch_start=0):
	""" Runs the epoch loop for a given level of hgmca using numba optimization.

	For an in depth description of the algorithm see arxiv 1910.08077

	Parameters:
		X_level (np.array): A numba.typed.List of numpy arrays corresponding to
		the data for each level of analysis.
		A_hier_list ([np.array]): A numba.typed.List of numpy arrays containing
			the mixing matrix hierarchy for each level of analysis.
		lam_hier (np.array): A n_sources long array of the prior for each of
			the columns of the mixing matrices in A_hier. This allows for a
			source dependent prior.
		A_global (np.array): A global mixing matrix prior that will be
			applied to all matrices in the hierarchy
		lam_global (np.array): A n_sources long array of the prior for each
			of the columns of the global prior.
		S_level ([np.array,...]):A numba.typed.List of numpy arrays containing
			the source matrices for each level of analysis.
		n_epochs (int): The number of times the algorithm should pass
			through all the levels.
		m_level (int): The maximum level of analysis.
		n_iterations (int): The number of iterations of coordinate descent
			to conduct per epoch.
		lam_s (float): The lambda parameter for the sparsity l1 norm.
		seed (int): An integer to seed the random number generator.
		enforce_nn_A (bool): A boolean that determines if the mixing matrix
			will be forced to only have non-negative values.
		min_rmse_rate (int): How often the source matrix will be set to the
			minimum rmse solution. 0 will never return min_rmse within the
			gmca optimization.
		epoch_start (int): What epoch the code is starting at. Important
			for min_rmse_rate.

	Notes:
		A_hier and S_level will be updated in place.
	"""
	# Set the random seed.
	np.random.seed(seed)
	# Now we iterate through our graphical model using our approximate closed
	# form solution for the desired number of epochs.
	for epoch in range(epoch_start,epoch_start+n_epochs):
		# We want to iterate through the levels in random order. This should
		# theoretically speed up convergence.
		level_perm = np.random.permutation(m_level+1)
		for level in level_perm:
			npatches = wavelets_hgmca.level_to_npatches(level)
			for patch in range(npatches):
				# Calculate the mixing matrix prior and the number of matrices
				# used to construct that prior.
				A_prior = get_A_prior(A_hier_list,level,patch,lam_hier)
				A_prior += lam_global * A_global
				# First we deal with the relatively simple case where there are
				# no sources at this level
				if X_level[level].size == 0:
					A_hier_list[level][patch] = A_prior
					helpers.A_norm(A_hier_list[level][patch])
				# If there are sources at this level we must run GMCA
				else:
					# Extract the data for the patch.
					X_p = X_level[level][patch]
					S_p = S_level[level][patch]
					# For HGMCA we want to store the source signal that
					# includes the lasso shooting subtraction of lam_s for
					# stability of the loss function. Only in the last step do
					# we want gmca to return the min_rmse solution.
					if min_rmse_rate == 0:
						ret_min_rmse = False
					else:
						ret_min_rmse = (((epoch+1)%min_rmse_rate)==0)

					# Call gmca for the patch. Note the lam_p has already been
					# accounted for.
					n_sources = len(S_p)
					gmca_core.gmca_numba(X_p,n_sources,n_iterations,
						A_hier_list[level][patch],S_p,A_p=A_prior,
						lam_p=np.ones(n_sources),enforce_nn_A=enforce_nn_A,
						lam_s=lam_s,ret_min_rmse=ret_min_rmse)


def save_numba_hier_lists(A_hier_list,S_level,save_path):
	"""Saves the numpy arrays used for the hierarchical analysis to the
	provided path.

	Parameters:
		A_hier_list ([np.array]): A numba.typed.List of numpy arrays containing
			the mixing matrix hierarchy for each level of analysis.
		S_level ([np.array,...]):A numba.typed.List of numpy arrays containing
			the source matrices for each level of analysis.
		save_path (str): The path to save the files to. A folder will be
			created at this path.
	"""
	# If the directory isn't already there, make it
	folder_path = os.path.join(save_path,'hgmca_save')
	if not os.path.isdir(folder_path):
		os.mkdir(folder_path)
	for level in range(len(A_hier_list)):
		np.save(os.path.join(folder_path,'A_%d.npy'%(level)),A_hier_list[level])
		np.save(os.path.join(folder_path,'S_%d.npy'%(level)),S_level[level])


def load_numba_hier_list(save_path,m_level):
	"""Laods the numpy arrays used for the hierarchical analysis from the
	provided path.

	Parameters:
		save_path (str): The path to save the files to. A folder will be
			created at this path.
		m_level (int): The maximum level of analysis.
	"""
	# Make sure the folder exists
	folder_path = os.path.join(save_path,'hgmca_save')
	if not os.path.isdir(folder_path):
		raise ValueError('%s is not a valid path to load files from'%(
			folder_path))
	# Go through and append all the elements to numba.typed.List objects
	S_level = numba.typed.List()
	A_hier_list = numba.typed.List()
	for level in range(m_level+1):
		A_hier_list.append(np.load(os.path.join(folder_path,'A_%d.npy'%(
			level))))
		S_level.append(np.load(os.path.join(folder_path,'S_%d.npy'%(level))))

	return A_hier_list, S_level


@numba.njit()
def init_min_rmse(X_level,A_hier_list,S_level):
	"""Initializes the source hierarhcy to the minimum RMSE solution
	given the mixing matrix hierarchy

	Parameters:
		X_level (np.array): A numba.typed.List of numpy arrays corresponding to
			the data for each level of analysis.
		A_hier_list ([np.array]): A numba.typed.List of numpy arrays containing
			the mixing matrix hierarchy for each level of analysis.
		S_level ([np.array,...]):A numba.typed.List of numpy arrays containing
			the source matrices for each level of analysis.
	"""
	for level in range(len(X_level)):
		# Skip levels with no data
		if X_level[level].size==0:
			continue
		# Go through each patch and calculate the min rmse. This requires
		# removing nans, so we will use a temp array to store a version
		# of X with nans converted to 0s.
		X_temp = np.zeros(X_level[level][0].shape)
		for patch in range(wavelets_hgmca.level_to_npatches(level)):
			X_temp *= 0
			X_temp += X_level[level][patch]
			helpers.nan_to_num(X_temp)
			np.dot(np.linalg.pinv(A_hier_list[level][patch]),X_temp,
				out=S_level[level][patch])


def hgmca_opt(wav_analysis_maps,n_sources,n_epochs,lam_hier,lam_s,
	n_iterations,A_init=None,A_global=None,lam_global=None,seed=0,
	enforce_nn_A=True,min_rmse_rate=0,save_dict=None,verbose=False):
	"""Runs the Hierachical GMCA algorithm on a dictionary of input maps.

	Paramters:
		wav_analysis_maps (dict): A dictionary containing the wavelet maps that
			we will run HGMCA on.
		n_sources (int): The number of sources.
		n_epochs (int): The number of times the algorithm should pass
			through all the levels.
		lam_hier (np.array): A n_sources long array of the prior for each of
			the columns of the mixing matrices in A_hier. This allows for a
			source dependent prior.
		lam_s (float): The lambda parameter for the sparsity l1 norm.
		n_iterations (int): The number of iterations of coordinate descent
			to conduct per epoch.
		A_init (np.array): A value at which to initialize all of the
			matrices in the hierarchy. If None then matrices will be
			initialized to 1/np.sqrt(n_freqs)
		A_global (np.array): A global mixing matrix prior that will be
			applied to all matrices in the hierarchy. If None no global
			prior will be enforced.
		lam_global (np.array): A n_sources long array of the prior for each
			of the columns of the global prior. Must be set if A_global is
			set.
		seed (int): An integer to seed the random number generator.
		enforce_nn_A (bool): A boolean that determines if the mixing matrix
			will be forced to only have non-negative values.
		min_rmse_rate (int): How often the source matrix will be set to the
			minimum rmse solution. 0 will never return min_rmse within the
			gmca optimization.
		save_dict (dict): A dictionary containing two entries, save_rate
			which is how often (per epoch) the results will be saved,
			and save_path, a folder the results will be saved to. If
			save_dict is provided the algorithm will try to initialize from
			the last save state.
		verbose (bool): If set to true, some timing statistics will be
			outputted for the epochs.

	Returns:
		(dict): Returns a dict with the mixing matrix and the source matrix at
		each level of analysis.
	"""
	if wav_analysis_maps['analysis_type'] != 'hgmca':
		raise ValueError('These wavelet functions were not generated using '+
			'the hgmca analysis type')

	if (A_global is None) != (lam_global is None):
		raise ValueError('Either both A_global and lam_global should be ' +
			'passed in or neither should be passed in.')

	# Copy over the information we need from the wav_analysis_maps dict.
	hgmca_analysis_maps = {
		'input_maps_dict':wav_analysis_maps['input_maps_dict'],
		'analysis_type':'hgmca','scale_int':wav_analysis_maps['scale_int'],
		'j_min':wav_analysis_maps['j_min'],'j_max':wav_analysis_maps['j_max'],
		'band_lim':wav_analysis_maps['band_lim'],
		'target_fwhm':wav_analysis_maps['target_fwhm'],
		'output_nside':wav_analysis_maps['output_nside'],
		'm_level':wav_analysis_maps['m_level']}

	# Get all the pieces we need to pass into the numba code
	m_level = wav_analysis_maps['m_level']
	X_level = convert_wav_to_X_level(wav_analysis_maps)
	n_freqs = wav_analysis_maps['n_freqs']
	A_shape = (n_freqs,n_sources)

	# If a save dict is provided that already has values in it, load them.
	# Otherwise initialize our A_hier_list and S_level
	if save_dict is None or not os.path.isdir(os.path.join(
		save_dict['save_path'],'hgmca_save')):
		A_hier_list = allocate_A_hier(m_level,A_shape,A_init=A_init)
		S_level = allocate_S_level(m_level,X_level,n_sources)
		# Initialize S to the minimum rmse solution
		init_min_rmse(X_level,A_hier_list,S_level)
	else:
		if verbose:
			print('Loading previous values from %s'%(save_dict['save_path']))
		A_hier_list, S_level = load_numba_hier_list(save_dict['save_path'],
			m_level)

	# The numba code is not built to accept None arguments, so modify things
	# accordingly.
	if A_global is None:
		# Setting lam_global to 0 ensures no effect from A_global
		lam_global = np.zeros(n_sources)
		A_global = np.ones((n_freqs,n_sources))
		helpers.A_norm(A_global)

	if verbose:
		print('Running HGMCA with the following parameters:')
		print('	maximum level: %d'%(m_level))
		print('	number of epochs: %d'%(n_epochs))
		print('	iterations per epoch: %d'%(n_iterations))
		print('	enforce non-negativity of mixing matrix: %s'%(
			enforce_nn_A))
		print('	minimum rmse rate: %d'%(min_rmse_rate))
		print('Running main HGMCA loop')

	if save_dict:
		save_rate = save_dict['save_rate']
		if verbose:
			print('Saving results to %s every %d epochs'%(
				save_dict['save_path'],save_rate))
		for si in tqdm(range(n_epochs//save_rate),desc='hgmca epochs',
			unit_scale=save_rate):
			hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,lam_global,
				S_level,n_epochs,m_level,n_iterations,lam_s,seed,enforce_nn_A,
				min_rmse_rate,epoch_start=si*save_rate)
			save_numba_hier_lists(A_hier_list,S_level,save_dict['save_path'])
			# We want reproducible behavior, but we don't want the same seed
			# for each set of epochs. This is the best quick fix.
			if seed > 0:
				seed += 1
	else:
		if verbose:
			start = time.time()
		hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,lam_global,
			S_level,n_epochs,m_level,n_iterations,lam_s,seed,enforce_nn_A,
			min_rmse_rate)
		if verbose:
			print('HGMCA loop took %f seconds'%(time.time()-start))

	# Package the output into the hgmca_analysis_maps
	for level in range(m_level+1):
		hgmca_analysis_maps[str(level)] = {'A':A_hier_list[level],
			'S':S_level[level]}

	return hgmca_analysis_maps


def extract_source(hgmca_analysis_maps,A_target,freq_ind=0):
	"""Modifies wav_analysis_maps to only include the source closest
	in its frequency dependence to A_target

	Parameters:
		hgmca_analysis_maps (dict): A dictionary containing the information
			about the wavelet functions used for analysis, the original nside
			of the input map, and the wavelet maps that need to be transformed
			back to the original healpix space.
		A_target (np.array): An n_sources long np.array that contains the
			desired frequency scaling of the target source.
		freq_ind (int): The frequency to return the source at.
	Returns:
		(dict): A dictionary containing the information about the wavelet
		functions used for analysis, the original nside of the input map, and
		the wavelet maps that needto be transformed back to the original
		healpix space.
	"""
	wav_analysis_maps = {
		'input_maps_dict':hgmca_analysis_maps['input_maps_dict'],
		'analysis_type':'hgmca','scale_int':hgmca_analysis_maps['scale_int'],
		'j_min':hgmca_analysis_maps['j_min'],
		'j_max':hgmca_analysis_maps['j_max'],
		'band_lim':hgmca_analysis_maps['band_lim'],
		'target_fwhm':hgmca_analysis_maps['target_fwhm'],
		'output_nside':hgmca_analysis_maps['output_nside'],
		'm_level':hgmca_analysis_maps['m_level']}
	# Iterate through each of the levels and pick out the mixing matrix column
	# that is most similar to the target column.
	if len(A_target.shape) == 1:
		A_target = np.expand_dims(A_target,axis=-1)
	m_level = hgmca_analysis_maps['m_level']
	for level in range(m_level+1):
		S = hgmca_analysis_maps[str(level)]['S']
		if S.size == 0:
			continue
		# Allocate the array
		wav_analysis_maps[str(level)] = np.zeros((S.shape[0],S.shape[2]))
		# Find the best match for each patch
		A_hier = hgmca_analysis_maps[str(level)]['A']
		target_match = np.argmin(np.sum(np.abs(A_hier-A_target),axis=-2),
			axis=1)
		# Calculate the source time the mixing matrix normalization for
		# each patch.
		for patch in range(wavelets_hgmca.level_to_npatches(level)):
			wav_analysis_maps[str(level)][patch] += (
				A_hier[patch,freq_ind,target_match[patch]] *
				S[patch,target_match[patch]])

	# Return the new wav_analysis_maps
	return wav_analysis_maps
