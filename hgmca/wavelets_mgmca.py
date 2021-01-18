from hgmca import wavelets_base
import numpy as np
import healpy as hp


class WaveletsMGMCA(wavelets_base.WaveletsBase):
	""" Class for conducting the wavlet transforms for the MGMCA algorithm.
	"""
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
			nest (bool): If true the input maps are in the nested
				configuration.
			n_quads (int): Using the trapezoid rule, the number of
				bins to consider for integration


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
		band_lim_list = np.array(list(map(
			lambda x: input_maps_dict[x]['band_lim'],input_maps_dict)))
		nside_list = np.array(list(map(lambda x: input_maps_dict[x]['nside'],
			input_maps_dict)))
		nside_list = nside_list[np.argsort(fwhm_list)[::-1]]
		freq_list = freq_list[np.argsort(fwhm_list)[::-1]]
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
			'analysis_type':'mgmca','scale_int':scale_int,'j_min':j_min,
			'j_max':np.max(j_max_list),'band_lim':np.max(band_lim_list),
			'target_fwhm':target_fwhm,'output_nside':np.max(nside_list),
			'n_freqs':len(freq_list)}

		# In the case of mgmca, we want to group the wavelet scales such that
		# the number of frequencies is constant. Therefore, the minimum and
		# maximum scale of each group will be set by maximum scale of each
		# frequency being analyzed.
		# Pre-allocate the numpy arrays we're going to fill with each
		# set of wavelet scales
		n_pix = hp.nside2npix(wavelets_base.get_max_nside(scale_int,j_min,
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
			n_pix += hp.nside2npix(wavelets_base.get_max_nside(scale_int,j+1,
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
			freq_wav_dict = self.s2dw_wavelet_tranform(input_map,
				output_maps_prefix+str(freq),
				input_maps_dict[str(freq)]['band_lim'],scale_int,j_min,
				fwhm_list[i],target_fwhm=target_fwhm[:n_scales],
				precomputed=precomputed,nest=nest,n_quads=n_quads)

			# Now we populate our arrays with the wavelets
			nside = wavelets_base.get_max_nside(scale_int,j_min,
				np.max(nside_list))
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
				nside = wavelets_base.get_max_nside(scale_int,j+1,
					np.max(nside_list))
				dn = hp.nside2npix(nside)
				# The frequency index needs to ignore excluded frequencies
				freq_i = i-n_exc_freq
				wav_analysis_maps[str(scale_group)][freq_i,n_pix:n_pix+dn] = (
					hp.ud_grade(hp.read_map(
						freq_wav_dict['wav_%d_map'%(j)]['path'],nest=True,
						verbose=False,dtype=np.float64),nside,
						order_in='NESTED',order_out='NESTED'))
				# Update the position in the array
				n_pix += dn

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

		# Check that the right type of map dict was passed in.
		if analysis_type != 'mgmca':
			raise ValueError('A non-mgmca wav_analysis_maps was passed in.')

		# Start by grabbing the scale coefficients
		scale_nside = wavelets_base.get_max_nside(scale_int,j_min,output_nside)
		n_pix = 0
		scale_group = 0
		dn = hp.nside2npix(scale_nside)
		scale_path = output_maps_prefix+'_scaling.fits'
		hp.write_map(scale_path,
			wav_analysis_maps[str(scale_group)][n_pix:n_pix+dn],
			dtype=np.float64,overwrite=True,nest=True)
		wavelet_dict.update({'scale_map':{'path':scale_path,
			'nside':scale_nside}})

		# Update our position in the array
		n_pix += dn

		# Now iterate through the remaining scales
		for j in range(j_min,j_max+1):
			wav_nside = wavelets_base.get_max_nside(scale_int,j+1,output_nside)
			dn = hp.nside2npix(wav_nside)
			if n_pix+dn > len(wav_analysis_maps[str(scale_group)]):
				scale_group += 1
				n_pix = 0
			wav_path = output_maps_prefix+'_wav_%d.fits'%(j)
			hp.write_map(wav_path,
				wav_analysis_maps[str(scale_group)][n_pix:n_pix+dn],
				dtype=np.float64,overwrite=True,nest=True)
			wavelet_dict.update({'wav_%d_map'%(j):{'path':wav_path,
				'nside':wav_nside}})
			# Update the position in the array
			n_pix += dn

		return self.s2dw_wavelet_inverse_transform(wavelet_dict,np.min(
			target_fwhm),n_quads=n_quads)
