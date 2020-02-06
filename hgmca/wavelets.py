import numpy as np
import healpy as hp
import subprocess, os, sys, warnings

class AxisymWaveletTransformation(object):
	""" Wavelet transformation class that interfaces with s2let's axisym wavelet
		transformation.

		Paramters:
			wav_b (int): A wavelet parameter for axisym wavelets
			min_scale (int): The minimum wavelet scale to be used
			band_lim (int): The bandlimit for decomposition
			s2let_path (str): The path to the compiled s2let directory.
			samp (int): The sampling scheme to be used for the wavelet maps. 0
				is minimal sampling, 1 is full resolution sampling, and 2 is 
				oversampling (wavelet maps will have larger nside than the 
				original map). 0 should almost always be used.
	"""
	def __init__(self, wav_b, min_scale, band_lim, s2let_path = '../s2let', 
		samp=0):
		# Save each of the parameters to the class.
		self.wav_b = wav_b
		self.min_scale = min_scale
		self.band_lim = band_lim
		self.s2let_bin = s2let_path + '/bin'
		self.samp = samp
		# Initialize the nside list which will be filled as we write a map.
		self._nside_list = []

	def set_nside_list(self,nside_list):
		""" Set the value of nside list.

			Parameters:
				nside_list [int,...]: The values to set self._nside_list to
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
		subprocess.call([self.s2let_bin + 's2let_transform_analysis_hpx_multi', 
			orig_map_file, str(self.wav_b), str(self.min_scale), 
			str(self.band_lim), wav_map_prefix, str(self.samp)],stdout=stdout)

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
		subprocess.call([self.s2let_bin + 's2let_transform_synthesis_hpx_multi', 
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
				(np.array): A 2D array with dimensions (n_freqs,n_wavs).
		"""
		# If nside_list is already initialized, it's possible this class
		# was already used to generate wavelet coefficients. Warn that the
		# nside list will be reset.
		if not self._nside_list:
			warnings.warn('nside_list is already set. Overwritting it.')
			self._nside_list = []
		# Initialize the wavelet coefficients array.
		wav_coeff = np.array([])
		# Read the scale file
		wav_temp = hp.read_map(wav_map_prefix+"_scal_%d_%d_%d"%(self.band_lim,
			self.wav_b,self.min_scale)+".fits",verbose=False)
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
				verbose=False)
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
				wav_coeff (np.array): A 2D array with the wavelet coefficients
					to be written into fits files.
		"""
		# Isolate the coefficients in each map, and reorder them back to 
		# ring ordering.
		wav_temp = hp.reorder(wav_coeff[:self._nside_list[0]**2*12],n2r=True)
		hp.write_map(wav_map_prefix+"_scal_%d_%d_%d"%(self.band_lim,self.wav_b, 
				self.min_scale)+".fits",wav_temp,overwrite=True)
		start = self._nside_list[0]**2*12
		for curr_j in range(1,len(self._nside_list)):
			wav_temp = hp.reorder(wav_coeff[
				start:start+self._nside_list[curr_j]**2*12], n2r=True)
			hp.write_map(wav_map_prefix+"_wav_%d_%d_%d_%d"%(self.band_lim,
				self.wav_b, self.min_scale,self.min_scale+curr_j-1)+".fits", 
				wav_temp,overwrite=True)
			start = start + self._nside_list[curr_j]**2*12

	def get_wavelet_coeff(self, hpx_map_file, wav_map_prefix):
		""" Given the path to a healpix map, return the numpy array
			with the wavelet coefficients.

			Parameters:
				hpx_map_file (str): The path to the healpix map.
				wav_map_prefix (str): The prefix (including directory) to save
					the output wavelet fits files to.
			Returns:
				A numpy array with the wavelet coefficients with dimensions
					(n_freq,n_wavs)
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
					coefficients with dimensions (n_freq,n_wavs).

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