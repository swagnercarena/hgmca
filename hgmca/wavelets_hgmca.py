from hgmca import wavelets_base
import numpy as np
import healpy as hp
import numba


@numba.njit()
def nside_to_level(nside,m_level):
	"""Maps from the nside to the maximum level of subdivision possible
	on this map

	Parameters:
		nside (int): The nside of the healpix map
		m_level (int): The maximum level that should be considered.

	Returns
		(int): The maximum level of subdivision.

	Notes:
		This can be modified to change the minimum number of pixels at
		each level. This choice makes it so that the minimum number of
		pixels is 256.
	"""
	# Return the largest level that gives at minimum 256 pixels
	return int(min(max(np.log2(nside)-3,0),m_level))


@numba.njit()
def level_to_npatches(level):
	"""Maps from the level of analysis to the number of patches.

	Parameters:
		level (int): The level of analysis

	Returns
		(int): The number of patches at that level of analysis
	"""
	if level == 0:
		return 1
	else:
		return int(12*4**(level-1))


class WaveletsHGMCA(wavelets_base.WaveletsBase):
	""" Class for conducting the wavlet transforms for the HGMCA algorithm.

		Parameters:
			m_level (int): The maximum level of analysis for HGMCA to use.
	"""

	def __init__(self,m_level):

		# Initialize the super class
		super().__init__()

		self.m_level = m_level

	@staticmethod
	def get_analysis_level(scale_int,j_min,j_max,m_level,max_nside):
		""" Returns the analysis level appropriate for each wavelet scale

		Parameters:
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			j_min (int): The minimum wavelet scale to use in the decomposition
			j_max (int): The maximum wavelet scale to use in the decomposition
			m_level (int): The maximum level of analysis to consider
			max_nside (int): The maximum nside that each map will be at.

		Returns:
			(np.array): A numpy array with the analysis level to be used
				by each wavelet scale.

		"""
		# Check the level analysis that will be used by each scale, starting
		# with the scale coefficients.
		wav_level = np.zeros(2+j_max-j_min)
		scale_nside = wavelets_base.get_max_nside(scale_int,j_min,max_nside)
		wav_level[0] = nside_to_level(scale_nside,m_level)
		# Now all the remaining wavelet coefficients
		for ji, j in enumerate(range(j_min,j_max+1)):
			wav_nside = wavelets_base.get_max_nside(scale_int,j+1,max_nside)
			wav_level[ji+1] = nside_to_level(wav_nside,m_level)

		return wav_level

	def allocate_analysis_arrays(self,wav_analysis_maps,scale_int,j_min,j_max,
		m_level,max_nside,n_freqs):
		"""Allocate the analysis arrays in which we'll distribute the
		wavelet coefficients.

		Parameters:
			wav_analysis_maps (dict): A dictionary object where the allocated
				arrays will be stored.
			scale_int (int): The integer used as the basis for scaling
				the wavelet functions
			j_min (int): The minimum wavelet scale to use in the decomposition
			j_max (int): The maximum wavelet scale to use in the decomposition
			m_level (int): The maximum level of analysis to consider
			max_nside (int): The maximum nside that each map will be at.
			n_freqs (int): The number of frequencies to allocate for
		Notes:
			wav_analysis_maps dict will be modified in place. Each level of
			analysis will be allocated and stored with the level number as the
			key.
		"""
		# Get the analysis level for each coefficient
		wav_level = self.get_analysis_level(scale_int,j_min,j_max,m_level,
			max_nside)

		# Create a matching array with the j index for each scale (including
		# 0 for the scaling coefficients).
		wav_j_ind = np.zeros(2+j_max-j_min)
		wav_j_ind[1:] = np.arange(j_min,j_max+1)

		# Iterate through the layers to allocate the arrays
		for level in range(m_level+1):
			# If no wavelet scales should be analyzed at this level then
			# set it to None
			if np.sum(wav_level==level) == 0:
				wav_analysis_maps[str(level)] = None
				continue

			# Which scales belong at this level
			level_j_ind = wav_j_ind[wav_level==level]

			# Allocate the arrays
			n_pix = 0
			for j in level_j_ind:
				if j == 0:
					n_pix += hp.nside2npix(wavelets_base.get_max_nside(
						scale_int,j_min,max_nside))
				else:
					n_pix += hp.nside2npix(wavelets_base.get_max_nside(
						scale_int,j+1,max_nside))

			# Get the number of patches for a given level.
			n_patches = level_to_npatches(level)
			# Initialize to nans so that scales were data at some frequencies
			# is missing will be easy to detect.
			wav_analysis_maps[str(level)] = np.zeros((n_patches,n_freqs,
				n_pix//n_patches)) + np.nan

	def multifrequency_wavelet_maps(self,input_maps_dict,output_maps_prefix,
			scale_int,j_min,precomputed=False,nest=False,n_quads=1000):
		"""Creates and groups the wavelet coefficients of several maps by
		analysis level.

		This function allows for wavelet coefficients from several frequency
		maps to be grouped for the purposes of (h)gmca analysis.

		Parameters:
			input_maps_dict (dict): A dictionary that maps frequencies to
				band limits, fwhm, nside, and input map path. Units of arcmin.
			output_maps_prefix (str): The prefix that the output wavelet maps
				will be written to.
			analysis_type (string): A string specifying what type of analysis
				to divide the wavelet scales for. Current options are 'mgmca'
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
			hierarchical analysis. For hgmca the analysis groups are governed
			by the level of subdivision we'll conduct on the data. This is just
			governed by the nside of the analysis. Unlike mgmca, we won't reshape
			the arrays by the number of frequencies available. Instead, when
			certain frequencies contain no data at those scales we will
			populate the data with nans. This will then be used in the hgmca
			code to indicate that no constraining power should be derived from
			those frequencies at those scales.
		"""
		# First we want to order the frequencies by fwhm. Keys will be strings.
		freq_list = np.array(list(input_maps_dict.keys()))
		fwhm_list = np.array(list(map(lambda x: input_maps_dict[x]['fwhm'],
			input_maps_dict)))
		band_lim_list = np.array(list(map(
			lambda x:input_maps_dict[x]['band_lim'],input_maps_dict)))
		nside_list = np.array(list(map(lambda x: input_maps_dict[x]['nside'],
			input_maps_dict)))
		nside_list = nside_list[np.argsort(fwhm_list)[::-1]]
		freq_list = freq_list[np.argsort(fwhm_list)[::-1]]
		n_freqs = len(freq_list)
		band_lim_list = band_lim_list[np.argsort(fwhm_list)[::-1]]

		# Get the maximum wavelet scale for each map
		j_max_list = np.array(list(map(lambda x: wavelets_base.calc_j_max(
			input_maps_dict[x]['band_lim'],scale_int),input_maps_dict)))
		j_max_list = j_max_list[np.argsort(fwhm_list)[::-1]]
		fwhm_list = fwhm_list[np.argsort(fwhm_list)[::-1]]

		# We will always target the smallest fwhm.
		target_fwhm = np.ones(2+np.max(j_max_list)-j_min)*np.min(fwhm_list)

		# The wavelet analysis maps we will populated. Save the information
		# in the input_maps_dict for later reconstruction.
		wav_analysis_maps = {'input_maps_dict':input_maps_dict,
			'analysis_type':'hgmca','scale_int':scale_int,'j_min':j_min,
			'j_max':np.max(j_max_list),'band_lim':np.max(band_lim_list),
			'target_fwhm':target_fwhm,'output_nside':np.max(nside_list),
			'n_freqs':len(freq_list)}

		# Get the largest n_side that will be considered and therefore the
		# actual largest level that will be used (this will be less than
		# or equal to the maximum level specified).
		max_nside = wavelets_base.get_max_nside(scale_int,np.max(j_max_list)+1,
			np.max(nside_list))
		m_level = nside_to_level(max_nside,self.m_level)
		wav_analysis_maps['m_level'] = m_level

		self.allocate_analysis_arrays(wav_analysis_maps,scale_int,j_min,
			np.max(j_max_list),m_level,max_nside,n_freqs)

		# Get the analysis level for each coefficient
		wav_level = self.get_analysis_level(scale_int,j_min,
			np.max(j_max_list),m_level,max_nside)

		# Create a matching array with the j index for each scale (including
		# 0 for the scaling coefficients).
		wav_j_ind = np.zeros(2+np.max(j_max_list)-j_min)
		wav_j_ind[1:] = np.arange(j_min,np.max(j_max_list)+1)

		# Now we go through each input frequency map and populate the
		# wavelet map arrays.
		for freq_i, freq in enumerate(freq_list):
			n_scales = 2+j_max_list[freq_i]-j_min
			input_map = hp.read_map(input_maps_dict[str(freq)]['path'],
				verbose=False,dtype=np.float64)
			freq_wav_dict = self.s2dw_wavelet_tranform(input_map,
				output_maps_prefix+str(freq),
				input_maps_dict[str(freq)]['band_lim'],scale_int,j_min,
				fwhm_list[freq_i],target_fwhm=target_fwhm[:n_scales],
				precomputed=precomputed,nest=nest,n_quads=n_quads)

			# Iterate through the levels
			for level in range(m_level+1):
				# If no wavelet scales should be analyzed at this level
				# continue
				if np.sum(wav_level==level) == 0:
					continue
				# Which scales belong at this level
				level_j_ind = wav_j_ind[wav_level==level]
				# Get the number of patches for a given level.
				n_patches = level_to_npatches(level)

				# Keep track of how many pixels into the level we've
				# gone so far.
				offset = 0
				for j in level_j_ind:
					# Check that this scale exists for this frequency
					if j > j_max_list[freq_i]:
						continue
					# Now deal with scaling or wavelet coefficient
					if j == 0:
						nside = wavelets_base.get_max_nside(scale_int,j_min,
							max_nside)
						wav_map_freq = hp.ud_grade(hp.read_map(
							freq_wav_dict['scale_map']['path'],nest=True,
							verbose=False,dtype=np.float64),nside,
							order_in='NESTED',order_out='NESTED')
					else:
						nside = wavelets_base.get_max_nside(scale_int,j+1,
							max_nside)
						# Read in the map for this frequency and scale
						wav_map_freq = hp.ud_grade(hp.read_map(
							freq_wav_dict['wav_%d_map'%(j)]['path'],nest=True,
							verbose=False,dtype=np.float64),nside,
							order_in='NESTED',order_out='NESTED')
					n_pix = hp.nside2npix(nside)
					n_pix_patch = n_pix//n_patches

					# Now populate each patch
					for patch in range(n_patches):
						wav_analysis_maps[str(level)][patch,freq_i,
							offset:offset+n_pix_patch] = wav_map_freq[
							patch*n_pix_patch:(patch+1)*n_pix_patch]

					# Update the number of pixels that have already been
					# filled.
					offset += n_pix_patch

		return wav_analysis_maps

	def wavelet_maps_to_real(self,wav_analysis_maps,output_maps_prefix,
		n_quads=1000):
		"""Take a wav_analysis_map dictionary correponding to a single
		frequency and the corresponding healpix map.

		Parameters:
			wav_analysis_maps (dict): A dictionary containing the information
				about the wavelet functions used for analysis, the original
				nside of the input map, and the wavelet maps that need
				to be transformed back to the original healpix space.
			output_maps_prefix (str): The prefix that the intermediary wavelet
				maps will be written to.

		Returns:
			(np.array): The reconstructed healpix map (in ring ordering).
		"""
		# Make the wavelet dict we'll feed into the reconstruction script.
		target_fwhm = wav_analysis_maps['target_fwhm']
		scale_int = wav_analysis_maps['scale_int']
		j_min = wav_analysis_maps['j_min']
		j_max = wav_analysis_maps['j_max']
		output_nside = wav_analysis_maps['output_nside']
		wavelet_dict = {'scale_int':scale_int,
			'band_lim':wav_analysis_maps['band_lim'],'j_max':j_max,
			'j_min':j_min,'original_nside':output_nside,
			'target_fwhm':target_fwhm}
		analysis_type = wav_analysis_maps['analysis_type']
		m_level = wav_analysis_maps['m_level']

		# Check that the right type of map dict was passed in.
		if analysis_type != 'hgmca':
			raise ValueError('A non-hgmca wav_analysis_maps was passed in.')

		# Get the analysis level for each coefficient
		wav_level = self.get_analysis_level(scale_int,j_min,j_max,m_level,
			output_nside)
		wav_j_ind = np.zeros(2+j_max-j_min)
		wav_j_ind[1:] = np.arange(j_min,j_max+1)

		# Iterate through the levels
		for level in range(m_level+1):
			# If no wavelet scales should be analyzed at this level
			# continue
			if np.sum(wav_level==level) == 0:
				continue
			# Which scales belong at this level
			level_j_ind = wav_j_ind[wav_level==level]
			# Get the number of patches for a given level.
			n_patches = level_to_npatches(level)

			# Keep track of how many pixels into the level we've
			# gone so far.
			offset = 0
			for j in level_j_ind:
				# Now deal with scaling or wavelet coefficient
				if j == 0:
					nside = wavelets_base.get_max_nside(scale_int,j_min,
						output_nside)
					path = output_maps_prefix+'_scaling.fits'
					wavelet_dict.update({'scale_map':{'path':path,
						'nside':nside}})
				else:
					nside = wavelets_base.get_max_nside(scale_int,j+1,
						output_nside)
					path = output_maps_prefix+'_wav_%d.fits'%(j)
					wavelet_dict.update({'wav_%d_map'%(j):{'path':path,
						'nside':nside}})
				n_pix = hp.nside2npix(nside)
				n_pix_patch = n_pix//n_patches

				# Allocate the array we'll use to write the wavelets
				wav_coeff = np.zeros(n_pix)

				# Now grab the data from each patch
				for patch in range(n_patches):
					wav_coeff[patch*n_pix_patch:(patch+1)*n_pix_patch] = (
						wav_analysis_maps[str(level)][patch,
						offset:offset+n_pix_patch])
				offset += n_pix_patch

				# Write the map and point the dictionary to the path
				hp.write_map(path,wav_coeff,dtype=np.float64,
					overwrite=True,nest=True)

		return self.s2dw_wavelet_inverse_transform(wavelet_dict,np.min(
			target_fwhm),n_quads=n_quads)
