import numpy as np
import healpy as hp
import subprocess
import os
import sys
import math
import numba
file_path = os.path.dirname(os.path.abspath(__file__))
healpy_weights_folder = os.path.join(file_path,'../healpy-data')


@numba.njit
def k_sdw(k,scale_int):
	"""The modified Schwartz function used to generate the wavelet
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
	"""Carry out the ratio of Schwartz Function integrals required
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
	"""Return the maximum level of analysis appropriate for a
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
	"""Calculate the smoothly decreasing function that will form the
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
	"""Calculate the harmonic space representation of the wavelet and
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
	"""Get the largest useful nside given the scale and the scaling
	integer. Will always be capped at input map nside.

	Parameters:
		scale_int (int): The integer used as the basis for scaling
			the wavelet functions
		j (int): The current wavelet scale
		nside (int): The nside of the input map.

	Returns:
		(int): The maximum useful nside. Minimum is 32.
	"""
	# The nside will be set to half the max ell (rounded up to the nearest
	# power of 2)
	new_nside = min((scale_int**j)/2,nside)
	new_nside = 2**(np.ceil(np.log(new_nside)/np.log(2)))
	new_nside = max(new_nside,32)
	return int(new_nside)


def get_alm_indices(old_lmax,new_lmax):
	"""Gives the indices for the alms of a smaller lmax in a larger
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
	j_min,input_fwhm,target_fwhm=None,n_quads=1000,precomputed=False,
	nest=False):
	"""Transform input healpix map into scale discretized wavelet
	representation.

	Parameters:
		input_map (np.array): The 1D array describing the healpix
			representation of the input map.
		output_maps_prefix (str): The prefix that the output wavelet maps
			will be written to.
		band_lim (int): The band limit for the function / map being
			analyzed by the wavelets
		scale_int (int): The integer used as the basis for scaling
			the wavelet functions
		j_min (int): The minimum wavelet scale to use in the decomposition
		input_fwhm (float): In radians, the fwhm of the input map. Units of
			radians.
		target_fwhm ([float,...]): A list of the fwhm to use for each
			scale in radians. In general this should be set to the lowest
			resolution map that will be analyzed at that scale. If None,
			an arbitrarily small target beam will be used. Units of radians.
		n_quads (int): Using the trapezoid rule, the number of
			bins to consider for integration
		precomputed (float): If true, will grab paths to precomputed maps
			based on the output_maps_prefix provided.
		nest (bool): If true the input map will be treated as though it's in
			the nested configuration.

	Returns:
		dict: A dictionary with one entry per wavelet scale that includes
			the healpix map.

	Notes:
		Multiple arrays will be generated, one for the scaling function,
		and one for each of the wavelet scales. All the wavelet maps
		are written in nested ordering for us with HGMCA.
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

	if precomputed:
		# In the precomputed case, we just need to check that the files
		# exist and add them to the dictionary
		scale_path = output_maps_prefix+'_scaling.fits'
		if not os.path.isfile(scale_path):
			raise FileNotFoundError
		scale_nside = get_max_nside(scale_int,j_min,nside)
		wavelet_dict.update({'scale_map':{'path':scale_path,
			'nside':scale_nside}})

		for j in range(j_min,j_max+1):
			# The nside will be set to half the max ell (rounded up to the
			# nearest power of 2)
			wav_nside = get_max_nside(scale_int,j+1,nside)
			wav_path = output_maps_prefix+'_wav_%d.fits'%(j)
			if not os.path.isfile(wav_path):
				raise FileNotFoundError
			wavelet_dict.update({'wav_%d_map'%(j):{'path':wav_path,
				'nside':wav_nside}})
	else:
		# If the input map is in nested ordering, convert it to ring
		# for these calculations
		if nest:
			input_map = np.reorder(input_map,n2r=True)
		# Convert the input map to alms
		flm = hp.map2alm(input_map,lmax=band_lim,
			datapath=healpy_weights_folder,use_pixel_weights=True)

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
		# Always write the wavelet maps in nested ordering.
		hp.write_map(scale_path,hp.reorder(hp.alm2map(scale_alm,
			nside=scale_nside),r2n=True),dtype=np.float64,overwrite=True,
			nest=True)
		wavelet_dict.update({'scale_map':{'path':scale_path,
			'nside':scale_nside}})

		# Write out the wavelet maps.
		for j in range(j_min,j_max+1):
			# Get the ratio of the bl for this wavelet band
			wav_bl = (hp.gauss_beam(target_fwhm[j-j_min+1],lmax=band_lim-1)/
				hp.gauss_beam(input_fwhm,lmax=band_lim-1))
			wav_alm = hp.almxfl(flm,wav_har[j*band_lim:(j+1)*band_lim]*
				norm_factor*wav_bl)
			# The nside will be set to half the max ell (rounded up to the
			# nearest power of 2)
			wav_nside = get_max_nside(scale_int,j+1,nside)
			wav_path = output_maps_prefix+'_wav_%d.fits'%(j)
			# Always write the wave maps in nested ordering.
			hp.write_map(wav_path,hp.reorder(hp.alm2map(wav_alm,
				nside=wav_nside),r2n=True),dtype=np.float64,overwrite=True,
				nest=True)
			wavelet_dict.update({'wav_%d_map'%(j):{'path':wav_path,
				'nside':wav_nside}})

	return wavelet_dict


def s2dw_wavelet_inverse_transform(wavelet_dict,output_fwhm,n_quads=1000):
	"""Reverse the wavelet transform to recreate the map.

	Parameters:
		wavelet_dict (dict): A dictionairy with the wavelet parameters
			used in the transform and the path to the wavelet map
			fits files.
		n_quads (int): Using the trapezoid rule, the number of
			bins to consider for integration
		output_fwhm (float): In radians, the desired fwhm of the output
			map. Note that if this is a higher fwhm than is supported by
			the data this will create numerical artifacts. Units of radians.

	Returns:
		np.array: The healpix map resulting from the inverse wavelet
		transform (ring ordering).

	Notes:
		The input wavelet maps must be in nested ordering (this is assumed
		for use with HGMCA).
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

	# Get the spherical harmonics for each of the maps we'll be using. We
	# need to put it back into ring ordering for map2alm
	scale_map = hp.read_map(wavelet_dict['scale_map']['path'],
		verbose=False,dtype=np.float64,nest=True)
	scale_map = hp.reorder(scale_map,n2r=True)
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
			verbose=False,dtype=np.float64,nest=True)
		wav_map = hp.reorder(wav_map,n2r=True)
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


def multifrequency_wavelet_maps(input_maps_dict,output_maps_prefix,
	analysis_type,scale_int,j_min,precomputed=False,nest=False,n_quads=1000):
	"""Creates and groups the wavelet coefficients of several maps by
	analysis level.

	This function allows for wavelet coefficients from several frequency maps
	to be grouped for the purposes of (h)gmca analysis.

	Parameters:
		input_maps_dict (dict): A dictionary that maps frequencies to
			band limits, fwhm, nside, and input map path. Units of radians.
		output_maps_prefix (str): The prefix that the output wavelet maps
			will be written to.
		analysis_type (string): A string specifying what type of analysis
			to divide the wavelet scales for. Current options are 'gmca'
			and 'hgmca'.
		scale_int (int): The integer used as the basis for scaling
			the wavelet functions
		j_min (int): The minimum wavelet scale to use in the decomposition
		precomputed (float): If true, will grab paths to precomputed maps
			based on the output_maps_prefix provided.

	Returns:
		(dict): A dictionary with one entry per level of analysis. Each
		entry contains a dict with the frequencies that are included and
		the np.array containing the wavelet coefficients

	Notes:
		The frequencies will be ordered from smallest to largest bandlimit.
		This choice is important to maintain contiguous arrays in a
		hierarchical analysis.
	"""
	# First we want to order the frequencies by fwhm. Keys will be strings.
	freq_list = np.array(list(input_maps_dict.keys()))
	fwhm_list = np.array(list(map(lambda x: input_maps_dict[x]['fwhm'],
		input_maps_dict)))
	nside_list = np.array(list(map(lambda x: input_maps_dict[x]['nside'],
		input_maps_dict)))
	nside_list = nside_list[np.argsort(fwhm_list)[::-1]]
	freq_list = freq_list[np.argsort(fwhm_list)[::-1]]

	# Get the maximum wavelet scale for each map
	j_max_list = np.array(list(map(lambda x: calc_j_max(
		input_maps_dict[x]['band_lim'],scale_int),input_maps_dict)))
	j_max_list = j_max_list[np.argsort(fwhm_list)[::-1]]
	fwhm_list = fwhm_list[np.argsort(fwhm_list)[::-1]]

	# We will always target the smallest fwhm.
	target_fwhm = np.ones(2+np.max(j_max_list)-j_min)*np.min(fwhm_list)

	# The wavelet analysis maps we will populated. Save the information
	# in the input_maps_dict for later reconstruction.
	wav_analysis_maps = {'input_maps_dict':input_maps_dict,'analysis_type':
		analysis_type}

	# In the case of gmca, we want to group the wavelet scales such that
	# the number of frequencies is constant. Therefore, the minimum and
	# maximum scale of each group will be set by maximum scale of each
	# frequency being analyzed.
	if analysis_type == 'gmca':
		# Pre-allocate the numpy arrays we're going to fill with each
		# set of wavelet scales
		n_pix = hp.nside2npix(get_max_nside(scale_int,j_min,
			np.max(nside_list)))
		scale_group = 0
		unique_j_max = np.unique(j_max_list)

		# Go through the scales and create arrays.
		for j in range(j_min,np.max(j_max_list)+1):
			# If this j starts a new group reset the number of pixels and
			# allocate an array.
			if j > unique_j_max[scale_group]:
				n_freqs = np.sum(j_max_list>=unique_j_max[scale_group])
				wav_analysis_maps[str(scale_group)] = (np.zeros((n_freqs,
					n_pix),dtype=np.float64))
				n_pix = 0
				scale_group += 1
			# Add the number of pixels in this scale.
			n_pix += hp.nside2npix(get_max_nside(scale_int,j+1,
				np.max(nside_list)))
		# Write out the final group
		n_freqs = np.sum(j_max_list>=unique_j_max[scale_group])
		wav_analysis_maps[str(scale_group)] = (np.zeros((n_freqs,n_pix),
			dtype=np.float64))

		# Now we have to iterate through the frequency maps and populate
		# the wavelet maps.
		for i, freq in enumerate(freq_list):
			n_scales = 2+j_max_list[i]-j_min
			input_map = hp.read_map(input_maps_dict[str(freq)]['path'],
				verbose=False,dtype=np.float64)
			freq_wav_dict = s2dw_wavelet_tranform(input_map,
				output_maps_prefix+str(freq),
				input_maps_dict[str(freq)]['band_lim'],scale_int,j_min,
				fwhm_list[i],target_fwhm=target_fwhm[:n_scales],
				precomputed=precomputed,nest=nest,n_quads=n_quads)

			# Now we populate our arrays with the wavelets
			nside = get_max_nside(scale_int,j_min,np.max(nside_list))
			n_pix = 0
			scale_group = 0
			dn = hp.nside2npix(nside)
			# The number of frequencies currently being excluded.
			n_exc_freq = 0
			wav_analysis_maps[str(scale_group)][i,n_pix:n_pix+dn] = (
				hp.ud_grade(hp.read_map(freq_wav_dict['scale_map']['path'],
				nest=True,verbose=False,dtype=np.float64),nside,
				order_in='NESTED',order_out='NESTED'))

			# Update our position in the array
			n_pix += dn

			for j in range(j_min,j_max_list[i]+1):
				if j > unique_j_max[scale_group]:
					scale_group += 1
					n_exc_freq = np.sum(j_max_list<unique_j_max[scale_group])
					n_pix = 0
				nside = get_max_nside(scale_int,j+1,np.max(nside_list))
				dn = hp.nside2npix(nside)
				# The frequency index needs to ignore excluded frequencies
				freq_i = i-n_exc_freq
				wav_analysis_maps[str(scale_group)][freq_i,n_pix:n_pix+dn] = (
					hp.ud_grade(hp.read_map(
						freq_wav_dict['wav_%d_map'%(j)]['path'],nest=True,
						verbose=False,dtype=np.float64),nside,
						order_in='NESTED',order_out='NESTED'))
				n_pix += dn

	elif analysis_type == 'hgmca':
		return
	else:
		raise ValueError('Analysis type %s not supported'%(analysis_type))

	return wav_analysis_maps


def wavelet_maps_to_real(wav_analysis_maps,nside):
	"""Take a wav_analysis_map dictionary correponding to a single
	frequency and the corresponding healpix map.

	Parameters:
		wav_analysis_maps (dict): A dictionary containing the information
			about the wavelet functions used for analysis, the original
			nside of the input map, and the wavelet maps that need
			to be transformed back to the original healpix space.

	Returns:
		(np.array): The reconstructed healpix map (in ring ordering).
	"""
	return
