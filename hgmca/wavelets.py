import numpy as np
import healpy as hp
import subprocess, os, sys, math, numba
file_path = os.path.dirname(os.path.abspath(__file__))
healpy_weights_folder = os.path.join(file_path,'../healpy-data')


@numba.njit
def k_sdw(k,scale_int):
	""" The modified Schwartz function used to generate the wavelet
		kernerls.

		Parameters:
			k (int): The input at which the function will be evaluated
			scale_int (int): The scaling function to use to reparameterize
				the inputs for evaluation.
	"""
	reparam_k = (k-(1.0/scale_int))*(2.0*scale_int/(scale_int-1)) - 1
	if reparam_k<=-1 or reparam_k>=1:
		return 0
	else:
		return np.exp(-2.0/(1.0-reparam_k**2.0))/k


@numba.njit
def kappa_integral(lower,upper,n_quads,scale_int):
	""" Carry out the ratio of Schwartz Function integrals required
		for the kappa.

		Parameters:
			lower (float): The lower bound of integration
			upper (float): The upper bound of integration
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
	"""
	# Unit of integration
	dx = (upper-lower)/n_quads
	if upper==lower:
		return 0
	int_sum = 0
	for i in range(n_quads):
		f1 = k_sdw(lower+i*dx,scale_int)
		f2 = k_sdw(lower+(i+1)*dx,scale_int)
		int_sum += (f1+f2)*dx/2
	return int_sum


@numba.njit
def calc_j_max(band_lim,scale_int):
	"""	Return the maximum level of analysis appropriate for a
		specific wavelet scaling integer and band limit.

		Parameters:
			band_lim (int): The band limit for the function / map being
				analyzed by the wavelets
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
	"""
	return int(np.ceil(np.log(band_lim)/np.log(scale_int)))


@numba.njit
def phi2_s2dw(phi2,band_lim,scale_int,n_quads):
	"""	Calculate the smoothly decreasing function that will form the
		basis of the wavelet kernel function (and the scaling function).

		Parameters:
			phi2 (np.array): A pre-initialized array that will be modified
				in place to reflect the values of the smoothly decreasing
				function for the wavelet kernel.
			band_lim (int): The band limit for the function / map being
				analyzed by the wavelets
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration
	"""
	# Calculate the normalizing coefficient for our wavelets
	norm = kappa_integral(1.0/scale_int,1.0,n_quads,scale_int)
	# Calculate the maximum value of J given the scaling integer and the
	# band limit of the data.
	j_max = calc_j_max(band_lim,scale_int)
	for j in range(j_max+2):
		for ell in range(band_lim):
			if ell < scale_int**(j-1):
				phi2[ell+j*band_lim] = 1
			elif ell > scale_int**(j):
				phi2[ell+j*band_lim] = 0
			else:
				phi2[ell+j*band_lim] = kappa_integral(ell/scale_int**j,1.0,
					n_quads,scale_int)/norm


@numba.njit
def s2dw_harmonic(wav_har,scale_har,band_lim,scale_int,n_quads,j_min):
	"""	Calculate the harmonic space representation of the wavelet and
		scaling function kernels.

		Parameters:
			wav_har (np.array): The harmonic space representation of the
				wavelet kernels.
			scale_har (np.array): The harmonic space representation of the
				scaling function kernel.
			band_lim (int): The band limit for the function / map being
				analyzed by the wavelets
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration
			j_min (int): The minimum wavelet scale to use in the decomposition
	"""
	# Calculate the maximum value of J given the scaling integer and the
	# band limit of the data.
	j_max = calc_j_max(band_lim,scale_int)
	phi2 = np.zeros((j_max+2)*band_lim)
	# Calculate the function we will use as the basis for our kernel.
	phi2_s2dw(phi2,band_lim,scale_int,n_quads)
	# Use to check for cases where numerical error gives a slightly
	# negative number.
	dif = 0

	# Start by filling in the harmonic space represenation of the wavelet
	# kernel
	for j in range(j_min,j_max+1):
		for ell in range(band_lim):
			# Add normalizing factor
			wav_har[ell+j*band_lim] = np.sqrt((2*ell+1)/(4*np.pi))
			# We want sqrt(phi2(ell/lambda^(j+1))-phi2(ell/lambda^j)).
			# Adding one level of j adds another power of lambda.
			dif = phi2[ell+(j+1)*band_lim]-phi2[ell+(j)*band_lim]
			# Eliminate negative numerical error.
			if dif<0 and dif>-1e-10:
				dif = 0
			wav_har[ell+j*band_lim] *= np.sqrt(dif)

	# Now fill in the harmonic space representation of the scaling functions
	for ell in range(band_lim):
		scale_har[ell] = np.sqrt((2*ell+1)/(4*np.pi))*np.sqrt(
			phi2[ell+j_min*band_lim])


@numba.njit
def get_max_nside(scale_int,j,nside):
	"""	Get the largest useful nside given the scale and the scaling
		integer. Will always be capped at input map nside.

		Parameters:
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			j (int): The current wavelet scale
			nside (int): The nside of the input map.

		Returns:
			int: The maximum useful nside. Minimum is 32.
	"""
	# The nside will be set to half the max ell (rounded up to the nearest
	# power of 2)
	new_nside = min((scale_int**j)/2,nside)
	new_nside = 2**(np.ceil(np.log(new_nside)/np.log(2)))
	new_nside = max(new_nside,32)
	return int(new_nside)


def get_alm_indices(old_lmax,new_lmax):
	"""	Gives the indices for the alms of a smaller lmax in a larger
		lmax's array. This funciton is required due to the way healpy
		and healpix deal with their alm arrays.

		Parameters:
			old_lmax (int): The lmax or band limit of the alm array
				you are mapping from
			new_lmax (int): The lmax of the array you are mapping to.

		Returns:
			(np.array) The indices of the alms in the old array in the
			new array.
	"""
	new_indices = np.zeros(hp.Alm.getsize(old_lmax),dtype=np.int)
	# Run through each index in the old array, find the l and m value
	# and check their index in the new array.
	for i in range(len(new_indices)):
		new_indices[i] = hp.Alm.getidx(*((new_lmax,)+
			hp.Alm.getlm(old_lmax,i)))
	return new_indices


def s2dw_wavelet_tranform(input_map,output_maps_prefix,band_lim,scale_int,
	j_min,input_fwhm,target_fwhm=None,n_quads=1000):
	"""	Transform input healpix map into scale discretized wavelet
		representation.

		Parameters:
			input_map (np.array): The 1D array describing the healpix
				representation of the input map.
			band_lim (int): The band limit for the function / map being
				analyzed by the wavelets
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			j_min (int): The minimum wavelet scale to use in the decomposition
			input_fwhm (float): In radians, the fwhm of the input map.
			target_fwhm ([float,...]): A list of the fwhm to use for each
				scale in radians. In general this should be set to the lowest
				resolution map that will be analyzed at that scale. If None,
				an arbitrarily small target beam will be used.
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration

		Returns:
			dict: A dictionary with one entry per wavelet scale that includes
				the healpix map.

		Notes:
			Multiple arrays will be generated, one for the scaling function,
			and one for each of the wavelet scales.
	"""
	# Initialize the dictionary object.
	wavelet_dict = dict()
	# Calculate the maximum value of J given the scaling integer and the
	# band limit of the data.
	j_max = calc_j_max(band_lim,scale_int)
	wavelet_dict.update({'band_lim':band_lim})
	wavelet_dict.update({'scale_int':scale_int})
	wavelet_dict.update({'j_min':j_min})
	wavelet_dict.update({'j_max':j_max})
	wavelet_dict.update({'input_fwhm':input_fwhm})

	# Grab the nside from the input map
	nside = np.sqrt(len(input_map)//12).astype(np.int)
	wavelet_dict.update({'original_nside':nside})

	# Generate the array that will contain our wavelet maps in healpix space
	n_scales = j_max-j_min+2
	wavelet_dict.update({'n_scales':n_scales})

	if target_fwhm is None:
		target_fwhm = np.ones(n_scales+1)*1e-10

	wavelet_dict.update({'target_fwhm':target_fwhm})

	# Convert the input map to alms
	flm = hp.map2alm(input_map,lmax=band_lim,datapath=healpy_weights_folder,
		use_pixel_weights=True)

	# Generate the kernel function in harmonic space for our wavelets and
	# scaling functions
	wav_har = np.zeros((j_max+2)*band_lim)
	scale_har = np.zeros(band_lim)
	s2dw_harmonic(wav_har,scale_har,band_lim,scale_int,n_quads,j_min)

	# We have another normalizing factor in the tranform (that in this case
	# would have been equivalent to not including a normalizing factor in our
	# original calculation)
	ell = np.arange(band_lim)
	norm_factor = np.sqrt((4*np.pi)/(2*ell+1))

	# Grab the ratio of the input to target beam for the
	scale_bl = hp.gauss_beam(target_fwhm[0],lmax=band_lim-1)/hp.gauss_beam(
		input_fwhm,lmax=band_lim-1)

	# Write out the scaling map.
	scale_alm = hp.almxfl(flm,scale_har*norm_factor*scale_bl)
	# The nside will be set to half the max ell (rounded up to the nearest
	# power of 2)
	scale_nside = get_max_nside(scale_int,j_min,nside)
	scale_path = output_maps_prefix+'_scaling.fits'
	hp.write_map(scale_path,hp.alm2map(scale_alm,nside=scale_nside),
		dtype=np.float64,overwrite=True)
	wavelet_dict.update({'scale_map':{'path':scale_path,'nside':scale_nside}})

	# Write out the wavelet maps.
	for j in range(j_min,j_max+1):
		# Get the ratio of the bl for this wavelet band
		wav_bl = (hp.gauss_beam(target_fwhm[j-j_min+1],lmax=band_lim-1)/
			hp.gauss_beam(input_fwhm,lmax=band_lim-1))
		wav_alm = hp.almxfl(flm,wav_har[j*band_lim:(j+1)*band_lim]*
			norm_factor*wav_bl)
		# The nside will be set to half the max ell (rounded up to the nearest
		# power of 2)
		wav_nside = get_max_nside(scale_int,j+1,nside)
		wav_path = output_maps_prefix+'_wav_%d.fits'%(j)
		hp.write_map(wav_path,hp.alm2map(wav_alm,nside=wav_nside),
			dtype=np.float64,overwrite=True)
		wavelet_dict.update({'wav_%d_map'%(j):{'path':wav_path,
			'nside':wav_nside}})

	return wavelet_dict


def s2dw_wavelet_inverse_transform(wavelet_dict,output_fwhm,n_quads=1000):
	""" Reverse the wavelet transform to recreate the map.

		Parameters:
			wavelet_dict (dict): A dictionairy with the wavelet parameters
				used in the transform and the path to the wavelet map
				fits files.
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration
			output_fwhm (float): In radians, the desired fwhm of the output
				map. Note that if this is a higher fwhm than is supported by
				the data this will create numerical artifacts.

		Returns:
			np.array: The healpix map resulting from the inverse wavelet
			transform.
	"""
	# First pull out the transform parameters from the wavelet_dict object.
	scale_int = wavelet_dict['scale_int']
	band_lim = wavelet_dict['band_lim']
	j_max = wavelet_dict['j_max']
	j_min = wavelet_dict['j_min']

	# Initialize the alm for the scaling function and the wavelet functions
	final_alm_size = hp.Alm.getsize(band_lim)
	scale_alm = np.zeros(final_alm_size,dtype=np.complex64)
	wav_alm = np.zeros(final_alm_size*(j_max+2),dtype=np.complex64)

	# Generate the kernel functions in harmonic space for our wavelets and
	# scaling function.
	wav_har = np.zeros((j_max+2)*band_lim)
	scale_har = np.zeros(band_lim)
	s2dw_harmonic(wav_har,scale_har,band_lim,scale_int,n_quads,j_min)

	# Get the spherical harmonics for each of the maps we'll be using.
	scale_map = hp.read_map(wavelet_dict['scale_map']['path'],
		dtype=np.float64)
	scale_lim = min(int(scale_int**j_min),band_lim)
	# We need to find what indices to use
	if scale_lim < band_lim:
		new_indices = get_alm_indices(scale_lim,band_lim)
	else:
		new_indices = np.arange(final_alm_size)
	scale_alm[new_indices] = hp.map2alm(scale_map,
		lmax=scale_lim,datapath=healpy_weights_folder,use_pixel_weights=True)

	# Repeat the same for the wav_alm values
	for j in range(j_min,j_max+1):
		wav_map = hp.read_map(wavelet_dict['wav_%d_map'%(j)]['path'],
			dtype=np.float64)
		wav_lim = min(int(scale_int**(j+1)),band_lim)
		# We need to find what indices to use
		if wav_lim < band_lim:
			new_indices = get_alm_indices(wav_lim,band_lim)
		else:
			new_indices = np.arange(final_alm_size)
		wav_alm[final_alm_size*j+new_indices] = hp.map2alm(wav_map,
			lmax=wav_lim,datapath=healpy_weights_folder,
			use_pixel_weights=True)

	# Initialize our spherical harmonic alm and add them up. Also include the
	# norm
	ell = np.arange(band_lim)
	norm_factor = np.sqrt((4*np.pi)/(2*ell+1))

	# Correct for the beam
	target_fwhm = wavelet_dict['target_fwhm']
	scale_bl = (hp.gauss_beam(output_fwhm,lmax=band_lim-1)/
		hp.gauss_beam(target_fwhm[0],lmax=band_lim-1))

	# First add the scaling function
	flm = hp.almxfl(scale_alm,scale_har*norm_factor*scale_bl)

	# Then add the wavelet maps
	for j in range(j_min,j_max+1):
		wav_bl = (hp.gauss_beam(output_fwhm,lmax=band_lim-1)/
			hp.gauss_beam(target_fwhm[j-j_min+1],lmax=band_lim-1))
		flm += hp.almxfl(
			wav_alm[final_alm_size*j:final_alm_size*(j+1)],
			wav_har[j*band_lim:(j+1)*band_lim]*norm_factor*wav_bl)

	f_map = hp.alm2map(flm,wavelet_dict['original_nside'])

	return f_map



class AxisymWaveletTransformation(object):
	""" Wavelet transformation class that interfaces with s2let's axisym wavelet
		transformation.

		Paramters:
			wav_b (int): A wavelet parameter for axisym wavelets
			min_scale (int): The minimum wavelet scale to be used
			band_lim (int): The bandlimit for decomposition
			s2let_path (str): The path to the compiled s2let directory bin.
			samp (int): The sampling scheme to be used for the wavelet maps. 0
			is minimal sampling, 1 is full resolution sampling, and 2 is
			oversampling (wavelet maps will have larger nside than the
			original map). 0 should almost always be used.
	"""
	def __init__(self, wav_b, min_scale, band_lim, s2let_path = None,
		samp=0):
		# Save each of the parameters to the class.
		self.wav_b = wav_b
		self.min_scale = min_scale
		self.band_lim = band_lim
		if s2let_path is not None:
			self.s2let_bin = s2let_path
		else:
			self.s2let_bin = os.path.dirname(os.path.abspath(__file__))+'/../s2letbin/'
		self.samp = samp
		# Initialize the nside list which will be filled as we write a map.
		self._nside_list = []

	def set_nside_list(self,nside_list):
		""" Set the value of nside list.

			Parameters:
				nside_list ([int,...]): The values to set self._nside_list to
		"""
		self._nside_list = nside_list

	def get_nside_list(self):
		""" Return the value of nside list.

			Return:
				[int,...]: The nside_list values
		"""
		return self._nside_list

	def _s2let_generate_wavelet_maps(self, orig_map_file, wav_map_prefix,
		stdout=None):
		""" Given the path to a Healpix map, generate the wavelet coefficient
			maps.

			Parameters:
				orig_map_file (str): The path to the healpix map that will be
					transformed.
				wav_map_prefix (str): The prefix (including directory) to save
					the output wavelet fits files to.
				stdout (File): Where to write the standard output of the c
					calls. Can be used to suppress print statements.
		"""
		if len(wav_map_prefix) > 82:
			raise ValueError('Length of prefix is greater than 82 characters.' +
				'This will cause the s2let fortran code to crash. Change' +
				'naming conventions.')
		subprocess.call([os.path.join(self.s2let_bin,
			's2let_transform_analysis_hpx_multi'),
			orig_map_file, str(self.wav_b), str(self.min_scale),
			str(self.band_lim), wav_map_prefix, str(self.samp)],
			stdout=stdout)

	def _s2let_generate_recon_map(self, wav_map_prefix, recon_map_file, nside,
		stdout=None):
		""" Given the path to the wavelet coefficient maps, reconstruct the
			original healpix map.

			Parameters:
				wav_map_prefix (str): The prefix (including directory) to load
					the input wavelet fits files from.
				recon_map_file (str): The path to save the reconstructed
					healpix map to.
				nside (int): The nside for the healpix map to be written.
				stdout (File): Where to write the standard output of the c
					calls. Can be used to suppress print statements.

			Notes:
				The nside must be specified since the variable sampling
				options for wavelet maps allow for multiple nsides.
		"""
		print(recon_map_file)
		if len(recon_map_file) > 1000:
			raise ValueError('Length of recon_map_file is greater than 1000' +
				' characters. This will cause the s2let fortran code to crash.'+
				' Change naming conventions.')
		subprocess.call([os.path.join(self.s2let_bin,
			's2let_transform_synthesis_hpx_multi'),
			wav_map_prefix, str(self.wav_b), str(self.min_scale),
			str(self.band_lim), str(nside), recon_map_file, str(self.samp)],
			stdout=stdout)

	def _clean_prefix(self, wav_map_prefix):
		""" Given the wavelet map prefix, delete all the wavelet coefficient
			maps.

			Parameters:
				wav_map_prefix (str): The prefix (including directory) where
					the wavelet fits will be / are saved.
		"""
		# Go through the coefficients by name, and remove them.
		subprocess.call(['rm',wav_map_prefix+'_scal_%d_%d_%d'%(self.band_lim,
			self.wav_b,self.min_scale)+'.fits'])
		min_wav_index = self.min_scale
		max_wav_index = self.min_scale+len(self._nside_list)-1
		for curr_j in range(min_wav_index,max_wav_index):
			subprocess.call(['rm',wav_map_prefix+'_wav_%d_%d_%d_%d'%(
				self.band_lim,self.wav_b,self.min_scale,curr_j)+'.fits'])

	def _np_wavelet_coefficients_from_prefix(self,wav_map_prefix):
		""" Given the wav_map_prefix, load the wavelet coefficient maps and
			store them as a numpy file.

			Parameters:
				wav_map_prefix (str): The prefix (including directory) where
					the wavelet fits will are saved.
			Return:
				np.array: A 2D array with dimensions n_freqs x n_wavs.
		"""
		# Reset nside_list
		self._nside_list = []
		# Initialize the wavelet coefficients array.
		wav_coeff = np.array([])
		# Read the scale file
		wav_temp = hp.read_map(wav_map_prefix+"_scal_%d_%d_%d"%(self.band_lim,
			self.wav_b,self.min_scale)+".fits",verbose=False,dtype=np.float64)
		# We will re-order the wavelet coefficients so that they are in nested
		# ordering. Note that we need to reverse this change before writing the
		# coefficients.
		wav_temp = hp.reorder(wav_temp, r2n=True)
		wav_coeff = np.append(wav_coeff,wav_temp)
		# Append the nside of the map to the list.
		self._nside_list.append(int(np.sqrt(len(wav_temp)/12)))
		# Repeat the same process for the rest of the wavelet maps.
		curr_j = self.min_scale
		while (os.path.isfile(wav_map_prefix+"_wav_%d_%d_%d_%d"%(self.band_lim,
			self.wav_b, self.min_scale,curr_j)+".fits")):
			wav_temp = hp.read_map(wav_map_prefix+"_wav_%d_%d_%d_%d"%(
				self.band_lim,self.wav_b, self.min_scale,curr_j)+".fits",
				verbose=False,dtype=np.float64)
			wav_temp = hp.reorder(wav_temp, r2n=True)
			wav_coeff = np.append(wav_coeff,wav_temp)
			self._nside_list.append(int(np.sqrt(len(wav_temp)/12)))
			curr_j = curr_j + 1
		# Transform our nside_list into a numpy array for convenience.
		self._nside_list = np.array(self._nside_list)
		return wav_coeff

	def _np_wavelet_coefficients_to_fits(self,wav_map_prefix,wav_coeff):
		""" Given the wavelet coefficients, write out the wavelet maps.

			Parameters:
				wav_map_prefix (str): The prefix (including directory) where
					the wavelet fits will are saved.
				wav_coeff (np.array): A 1D array with the wavelet coefficients
					to be written into fits files.
		"""
		# Isolate the coefficients in each map, and reorder them back to
		# ring ordering.
		wav_temp = hp.reorder(wav_coeff[:self._nside_list[0]**2*12],n2r=True)
		hp.write_map(wav_map_prefix+"_scal_%d_%d_%d"%(self.band_lim,self.wav_b,
				self.min_scale)+".fits",wav_temp,overwrite=True,dtype=np.float64)
		start = self._nside_list[0]**2*12
		for curr_j in range(1,len(self._nside_list)):
			wav_temp = hp.reorder(wav_coeff[
				start:start+self._nside_list[curr_j]**2*12], n2r=True)
			hp.write_map(wav_map_prefix+"_wav_%d_%d_%d_%d"%(self.band_lim,
				self.wav_b, self.min_scale,self.min_scale+curr_j-1)+".fits",
				wav_temp,overwrite=True,dtype=np.float64)
			start = start + self._nside_list[curr_j]**2*12

	def get_wavelet_coeff(self, hpx_map_file, wav_map_prefix):
		""" Given the path to a healpix map, return the numpy array
			with the wavelet coefficients.

			Parameters:
				hpx_map_file (str): The path to the healpix map.
				wav_map_prefix (str): The prefix (including directory) to save
					the output wavelet fits files to.

			Return:
				np.array: A numpy array with the wavelet coefficients with
				dimensions n_wavs
		"""
		# Supress the output of our commamndline calls for readability.
		FNULL = open(os.devnull,'w')
		# TODO - check if the files are already there, and skip this step
		# if they are.
		self._s2let_generate_wavelet_maps(hpx_map_file,wav_map_prefix,
			stdout=FNULL)
		wav_coeff = self._np_wavelet_coefficients_from_prefix(wav_map_prefix)
		FNULL.close()
		return wav_coeff

	def get_map_from_wavelet_coeff(self, hpx_map_file, nside, wav_map_prefix,
			wav_coeff):
		"""	Given the wavelet coefficients, reconstruct and write out the
			healpix map file.

			Parameters:
				hpx_map_file (str): The path to save the reconstructed healpix
					map to.	If a file already exists at this location it will
					be overwritten.
				nside (int): The nside for the healpix map to be written.
				wav_map_prefix (str): The prefix (including directory) to save
					the output wavelet fits files to.
				wav_coeff (np.array): A numpy array with the wavelet
					coefficients with dimensions n_freq x n_wavs.

			Notes:
				The nside must be specified since the variable sampling
				options for wavelet maps allow for multiple nsides.
		"""
		# Supress the output of our commamndline calls for readability.
		FNULL = open(os.devnull,'w')
		self._np_wavelet_coefficients_to_fits(wav_map_prefix,wav_coeff)
		self._s2let_generate_recon_map(wav_map_prefix,hpx_map_file,nside,
			stdout=FNULL)
		FNULL.close()


	def hgmca_is_initialized(self):
		""" Checks whether the wavelet class is initialized to the data.

			Return:
				boolean: True if the class is initialized, false otherwise.
		"""
		if self.nside_list is not None and self.wav_lim_list is not None:
			return True
		else:
			return False

	def hgmca_divide_by_scale(self, X, ret_empty=False):
		""" Given the wavelet coefficients X, returns X divided into the seperate
			scales.

			Parameters:
				X (np.array): The wavelet coefficients in the form of a numpy matrix with
					dimensions n_freq x n_wavelets
				ret_empty (boolean): If set to true, the function will return a second list
					with identical dimensions to the first output and all array
					entries initialized to 0.

			Return:
				list: A list of the numpy arrays containing X at the different
					scales. If ret_empty is true it will return (X_scale,zero_scale)
					where the second list is identical in its dimensions to X_scale but
					all values are initialized to 0.

		"""
		X_scale = list()
		X_scale.append(X[:,:self.nside_list[0]**2*12])
		start = self.nside_list[0]**2*12
		for curr_j in range(1,len(self.nside_list)):
			X_scale.append(X[:,start: start+self.nside_list[curr_j]**2*12])
			start = start + self.nside_list[curr_j]**2*12

		if ret_empty:
			zero_scale = list()
			for curr_j in range(len(X_scale)):
				zero_scale.append(np.zeros(X_scale[curr_j].shape))
			return X_scale, zero_scale
		else:
			return X_scale

	def hgmca_reconstruct_by_scale(self, X_scale):
		""" Given the list of wavelet coefficients divided by scale, returns the
			full matrix of wavelet coefficients, reversing hgmca_divide_by_scale.

			Parameters:
				X_scale (list): the coefficients divided by scale

			Return:
				np.array: The full coefficients matrix with dimensions
					n_freqs x n_wavelets
		"""
		X = np.array(X_scale[0])
		for curr_j in range(1,len(X_scale)):
			X = np.append(X,X_scale[curr_j],axis=1)
		return X

	def hgmca_calc_wav_lim_list(self, nside, wav_map_prefix, cutoff = 0.05):
		""" Sets the level limits for each scale corresponding to the
			support of the wavelet being used. The limit is chosen such that
			the smallest patch size incorporates the entirety of a wavelet
			positioned in the middle of the patch

			Parameters:
				nside (int): the nside of the map the limits are generated for
				wav_map_prefix (str): the prefix to use for the wavelet maps
					cutoff: the percentage of the peak signal from which the signal
					must decay to no longer be considered within the domain.
		"""
		filler_hpx_map = np.ones(12*(nside**2))
		hpx_map_file = wav_map_prefix+'filler.fits'
		hp.write_map(hpx_map_file, filler_hpx_map,dtype=np.float64)
		wav_coeff = self.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		os.remove(hpx_map_file)
		start = 0
		wav_lim_list = list()
		for curr_j in range(len(self.nside_list)):
			# set non zero wavelet coefficient to be roughly at the center
			# of the map.
			start = start + 6*(self.nside_list[curr_j]**2)
			# set all but one wavelet coefficient to 0
			wav_coeff[:] = 0
			wav_coeff[start] = 1e3
			hpx_map_file = wav_map_prefix+'_wav_sup_scale_%d.fits'%(curr_j)
			self.get_map_from_wavelet_coeff(hpx_map_file, nside, wav_map_prefix,
				wav_coeff)
			# calculate the domain from the nonzero values on each map
			map_vals = hp.read_map(hpx_map_file,verbose=False,dtype=np.float64)
			# Find what portion of the map has signal > 5% the maximum (we will
			# roughly consider this to be the domain of our wavelet).
			ratio = len(map_vals)/len(np.where(np.abs(map_vals)>cutoff*np.max(
				map_vals))[0])//12
			ratio = max(ratio,1)
			wav_lim_list.append(int(math.log(ratio,4)))
			# deal with inconsistency that arise from reconstruction error for
			# large bandlimits
			if curr_j>0 and wav_lim_list[curr_j] < wav_lim_list[curr_j-1]:
				wav_lim_list[curr_j] = wav_lim_list[curr_j-1]+1

			# Clean up before the next scale
			os.remove(hpx_map_file)
			self.clean_prefix(wav_map_prefix)
			start = start + 6*(self.nside_list[curr_j]**2)

		self.wav_lim_list = wav_lim_list

	def hgmca_get_max_level(self, X, scale):
		"""Returns the maximum level of division permitted by the data. There are
			two limiting factors here - the support of the wavelets in question
			and the number of pixels in our map. The first can be intuited from
			the shape of the data, and the second requires knowledge of the scale.

			Parameters:
				X (np.array): The wavelet coefficients in the form of a numpy matrix with
					dimensions n_freqs x n_wavelets.
				scale (int): the wavelet scale of the data X.

			Return:
				int: The maximum level of dyadic division permitted by the constraints
					on the data.
		"""
		wav_lim = self.wav_lim_list[scale]
		# The number of divisions that are allowed such that we have 256 pixels
		# per patch (this may need to be played with). It is of course possible
		# to have a map small enough that no level of division can achieve this,
		# in which case we want to return 0.
		nside_lim = max(int(np.log2(np.sqrt(len(X[0])//12))) - 3,0)
		return min(nside_lim,wav_lim)

	def hgmca_get_max_level_list(self, X_scale):
		"""Returns a list of the maximum levels of division permitted by the data.

			Parameters:
				X_scale (list): The data divided up into wavelet scales.

			Return:
				list: A list of the maximum levels permitted by the data.
		"""
		max_level_list = np.zeros(len(X_scale))
		for scale_i in range(len(X_scale)):
			max_level_list[scale_i] = self.hgmca_get_max_level(X_scale[scale_i],
				scale_i)
		return max_level_list

	def hgmca_get_min_level(self):
		""" Returns the minimum level at which hgmca can begin its dyadic
			division. In the case of healpix this is limited to 1 since the first
			level of division is the 12 healpix faces.

			Return:
				int: Change manually depending on map representation.
		"""
		return 1

	def hgmca_get_n_patches(self, X,lev):
		""" Returns the number of patches that X will be divided into at level lev.

			Parameters:
				X (np.array): The data that will be divided into patches. Pass in None if
					there is no such data (in the case of their not being sources
					at this particular scale).
				lev (int): the level of division

			Return:
				int: The number of patches
		"""
		if lev == 0:
			return 1
		else:
			if X is None or 12*(4**(lev-1))<=len(X[0]):
				return 12*(4**(lev-1))
			else:
				sys.exit("Level exceed maximum allowed by dimensions of data")

	def hgmca_get_lev_from_n_patches(self,n_patches):
		""" Computes the level from the number of patches, assuming a healpix like
			representation which divides the map into 12 patches in the first step,
			and 4 patches in each subsequent step.

			Return:
				int: The current level of hgmca.
		"""
		if n_patches==1:
			return 0
		elif n_patches == 12:
			return 1
		else:
			return math.log(n_patches//12,4)+1

	def hgmca_data_for_patch(self, X, lev, patch, scale=False):
		""" Returns the subset of the data in X corresponding to the patch and
			level provided.

			Parameters:
				X (np.array): The data from which the patch will be extracted
				lev (int): the level of subdivision
				patch (int): the patch number
				scale (boolean): a boolean to indicate if X is multiple scales

			Return:
				np.array: The subset of the data corresponding to the patch of X
		"""
		if not scale:
			n_patches = self.hgmca_get_n_patches(X,lev)
			n_pix = len(X[0])
			return X[:,int(n_pix*patch/n_patches):int(n_pix*(patch+1)/n_patches)]
		else:
			n_patches = self.hgmca_get_n_patches(X[0],lev)
			X_p = np.array([])
			n_pix = len(X[0][0])
			X_p = X[0][:,int(n_pix*patch/n_patches):int(n_pix*(patch+1)/n_patches)]
			for scale_i in range(1,len(X)):
				n_pix = len(X[scale_i][0])
				X_p = np.append(X_p,X[scale_i][
					:,int(n_pix*patch/n_patches):int(n_pix*(patch+1)/n_patches)],
					axis=1)
			return X_p

	def hgmca_divide_by_level(self, X, mu_dict, X_scale=None):
		"""	Divide the data by level and arrange the patches such that they
			neighbor each other in memory. Note also that rather than
			n_sig X n_pixels the matrices in X_level will be n_pixels X n_sig
			to further benefit from adjacent memory optimization.

			Parameters:
				X (np.array): The original data
				mu_dict (dict): A dictionary that maps from the level to scales that
					will be analyzed at that level of subdivision
				X_scale (list): If X_scale has already been calculated it can be
					passed in to speed up computation.

			Return:
				list: A list of matrices by level where each matrix includes
					the data from all of the scales that will be analyzed at that
					level. Note that the dimensions are (n_patches,n_freq,n_wav)
				np.array: A boolean numpy array defining whether or not there is data
					at each level
		"""
		# We initialize the list that will contain our matrices by level
		X_level = list()
		# Keeps track of whether or not there is data for that level
		lev_data = []
		# We need the data divided by scale if that has not been done ahead of
		# time.
		if X_scale is None:
			X_scale = self.hgmca_divide_by_scale(X)

		# Now we will go through level by level and piece together all of the
		# scales that corresond to a level. Note that instead of just stitching
		# together the data scale by scale, we stitch it together patch by patch.
		# This along with making the columns the frequencies / sources and the
		# rows the pixels should make memory access much faster.
		X_c = 0
		for lev in range(len(mu_dict)):
			if mu_dict[lev] is None:
				lev_data.append(False)
				continue
			lev_data.append(True)
			n_pix = 0
			for scale in mu_dict[lev]:
				n_pix += X_scale[scale].shape[1]
			n_patches = self.hgmca_get_n_patches(None,lev)
			X_p = self.hgmca_data_for_patch(
						[X_scale[scale] for scale in mu_dict[lev]],lev,0,
						scale=True)
			X_level.append(np.zeros((n_patches,)+X_p.shape))
			for patch in range(n_patches):
				X_p = self.hgmca_data_for_patch(
						[X_scale[scale] for scale in mu_dict[lev]],lev,patch,
						scale=True)
				X_level[X_c][patch] += X_p
			X_c += 1

		return X_level, np.array(lev_data,dtype=np.bool_)

	def hgmca_set_patch_coeff(self, X, X_p, lev, patch, scale=False):
		""" Set the data corresponding to the level and patch specified in S to
			S_p

			Parameters:
				X (np.array): The matrix to which the data will be written
				X_p (np.array): The patch data to be written
				lev (int): The corresponding level
				patch (int): The corresponding patch
				scale (boolean): indicates whether X is multiple scales
		"""
		if not scale:
			n_patches = self.hgmca_get_n_patches(X,lev)
			n_pix = len(X[0])
			X[:,int(n_pix*patch/n_patches):int(n_pix*(patch+1)/n_patches)]=X_p
		else:
			n_patches = self.hgmca_get_n_patches(X[0],lev)
			n_pix_tot = 0
			for scale_i in range(len(X)):
				n_pix = len(X[scale_i][0])
				X[scale_i][:,int(n_pix*patch/n_patches):int(n_pix*(
					patch+1)/n_patches)]=X_p[:,int(n_pix_tot):int(
					n_pix_tot+n_pix/n_patches)]
				n_pix_tot += n_pix/n_patches

	def hgmca_reconstruct_by_level(self, X_level, mu_dict, X_scale):
		"""	Reconstruct the data that has been organized by hgmca_divide_by_level.

			Parameters:
				X_level (list): The data organized by level
				mu_dict (dict): A dictionary that maps from the level to scales that
					will be analyzed at that level of subdivision
				X_scale (list): A list of the data divided by scale instead of level.
					This does not need to be populated with the correct values,
					and will be used as an intermediary step in calculations. It
					is passed in only to avoid reallocating memory that has
					already been allocated.

			Return:
				np.array: The reconstructed data.
		"""
		# Now we will go through level by level and piece together all of the
		# scales that corresond to a level. Note that instead of just stitching
		# together the data scale by scale, we stitch it together patch by patch.
		# This along with making the columns the frequencies / sources and the
		# rows the pixels should make memory access much faster.
		X_c = 0
		for lev in range(len(mu_dict)):
			if mu_dict[lev] is None:
				continue
			n_patches = self.hgmca_get_n_patches(None,lev)
			for patch in range(n_patches):
				X_p = X_level[X_c][patch]
				X_temp = [X_scale[scale] for scale in mu_dict[lev]]
				self.hgmca_set_patch_coeff(X_temp,X_p,lev,patch,scale=True)
				x_temp_i = 0
				for scale in mu_dict[lev]:
					X_scale[scale] = X_temp[x_temp_i]
					x_temp_i += 1
			X_c += 1

		return self.hgmca_reconstruct_by_scale(X_scale)

	def hgmca_generate_A_hier(self, m_level, A_init):
		""" Given a maximum level of subdivision will return a list of numpy
			matrices the represent the hierarchy of mixing matrices.

			Parameters:
				m_level (int): the maximum level of subdivision to be considered in the
					hierarchy
			Return:
				list: A list with m_level+1 entries with each entry having
					dimensions [n_1,n_2,n_3,...n_(m_level),A.shape] with n_l being
					the number of subdivisions at level l.
		"""
		A_hier = [np.zeros((1,) + A_init.shape)]
		A_hier[0] += A_init
		for lev in range(1,m_level+1):
			dims = 12*(4**(lev-1))
			A_temp = np.tile(A_init,(dims,1))
			A_temp = A_temp.reshape([dims]+list(A_init.shape))
			A_hier.append(A_temp)
		return A_hier

	def hgmca_get_mu_dict(self,X_scale,m_level):
		""" Generates a dictionary that given a maximum level and a realization
			of the data by scale will return a dictionary that, given a level,
			returns the sources that should be considered at that scale.

			Parameters:
				X_scale (list): the coefficients divided by scale
				m_level (int): The maximum level of subdivision that will be considered

			Return:
				dict: a dictionary of the scales to consider at each level.
		"""
		max_level_list = self.hgmca_get_max_level_list(X_scale)
		max_level_list[max_level_list>m_level] = m_level
		mu_dict = dict()
		for lev in range(int(np.max(max_level_list))+1):
			indices = np.where(max_level_list == lev)[0]
			if indices.size == 0:
				mu_dict[lev] = None
			else:
				mu_dict[lev] = indices
		return mu_dict


spec = [('lev_data',numba.boolean[:])]
@numba.jitclass(spec)
class JitAxisymWaveletTransformation(object):
	"""Wavelet transformation class that interfaces with s2let's axisym wavelet
		transformation.

		Paramters:
			lev_data (np.array): Whether or not there is data corresponding to a specific
				level (stored as a boolean array).
	"""
	def __init__(self,lev_data):
		self.lev_data = lev_data

	def hgmca_get_n_patches(self, lev):
		""" Returns the number of patches that X will be divided into at level
			lev.

			Parameters:
				lev (int): the level of division

			Return:
				int: The number of patches
		"""
		if lev == 0:
			return 1
		return 12*(4**(lev-1))

	def hgmca_data_for_patch(self, X_level, lev, patch):
		""" Returns the subset of the data in X corresponding to the patch and
			level provided.

			Parameters:
				X_level (list): The data from which the patch will be extracted
				lev (int): the level of subdivision
				patch (int): the patch number

			Return:
				np.array: The subset of the data corresponding to the patch of
				X.
		"""
		# Deal with the fact that X_level has no entries for levels with no data
		X_c = -1
		for l in range(lev+1):
			if self.lev_data[l] == True:
				X_c += 1
		return X_level[X_c][patch]

	def hgmca_get_A_prior(self,lev,patch,A_hier):
		""" Given the matrix hiearchy and the level and patch of a specific
			mixing matrix will return the sum of mixing matrices connected in
			the graphical model to that matrix.

			Parameters:
				lev (int): the level of the matrix for which the prior should
					be calculated
				patch (int): the patch of the aformentioned matrix
				A_hier ([np.array,...]): A list of numpy matrices (one per
					level). The arrays should have dimensions number of patches x
					number of maps x number of sources.

			Return:
				np.array: The sum of the connected matrices
		"""
		# If A_hier is only one level deep, return None to indicate no prior
		if len(A_hier) == 1:
			return None
		A_p = np.zeros(A_hier[0][0].shape)
		# Add the prior related to the previous level.
		if lev > 0:
			if lev == 1:
				A_p += A_hier[lev-1][0]
			else:
				A_p += A_hier[lev-1][patch//4]
		# Add the prior related to the next level.
		if lev == 0:
			# No need to calculate scaling here. It's 1 by definition.
			for child in range(12):
				A_p += A_hier[lev+1][patch*4+child]
		elif lev < len(A_hier)-1:
			for child in range(4):
				A_p += A_hier[lev+1][patch*4+child]
		return A_p