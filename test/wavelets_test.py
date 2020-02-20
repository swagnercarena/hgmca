import numpy as np
import healpy as hp
from hgmca import wavelets
import unittest

class TestAxisymWaveletTransformation(unittest.TestCase):

	def test_reconstruction(self):
		# The reconstruction still seems to have considerable error, so this simply
		# checks that the error is small.
		cmb_maps_path = 'test_data/'
		hpx_map_file = cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = cmb_maps_path + 'gmca_test_full_sim_90'
		recon_map_file = cmb_maps_path + 'gmca_test_full_sim_90_GHZ_recon.fits'	
		wav_b = 3
		min_scale = 1
		band_lim = 512
		nside = 128
		# No subsampling here
		samp = 1
		wav_t = wavelets.AxisymWaveletTransformation(wav_b,min_scale,band_lim,
			samp=samp)	
		wav_coeff = wav_t.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t.get_map_from_wavelet_coeff(recon_map_file,nside,wav_map_prefix,
			wav_coeff)
		#wav_t.clean_prefix(wav_map_prefix)
		orig_map = hp.read_map(hpx_map_file,verbose=False)
		recon_map = hp.read_map(recon_map_file,verbose=False)

		self.assertLess(np.mean(np.abs(orig_map-recon_map))/
			np.mean(np.abs(orig_map)),0.3)
		self.assertLess(np.max(np.abs(orig_map-recon_map))/
			np.max(np.abs(orig_map)),0.3)

	def test_recon_bandlim(self):
		# Tests how the reconstruction improves with increasing bandlimit
		cmb_maps_path = 'test_data/'
		hpx_map_file = cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = cmb_maps_path + 'gmca_test_full_sim_90'
		small_b_lim_map = cmb_maps_path + 'gmca_test_full_sim_90_GHZ_128_recon.fits'
		large_b_lim_map	= cmb_maps_path + 'gmca_test_full_sim_90_GHZ_420_recon.fits'
		wav_b = 3
		min_scale = 1
		nside = 128
		samp = 1

		wav_t_128  = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			128,samp=samp)	
		wav_coeff = wav_t_128.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_128.get_map_from_wavelet_coeff(small_b_lim_map,nside,wav_map_prefix,
			wav_coeff)
		#wav_t_128.clean_prefix(wav_map_prefix)

		wav_t_420 = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			420,samp=samp)
		wav_coeff = wav_t_420.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_420.get_map_from_wavelet_coeff(large_b_lim_map,nside,wav_map_prefix,
			wav_coeff)
		#wav_t_420.clean_prefix(wav_map_prefix)

		s_b_lim_hpx_map = hp.read_map(small_b_lim_map,verbose=False)
		l_b_lim_hpx_map = hp.read_map(large_b_lim_map,verbose=False)
		orig_map = hp.read_map(hpx_map_file,verbose=False)

		self.assertLess(np.mean(np.abs(orig_map-l_b_lim_hpx_map)),
			np.mean(np.abs(orig_map-s_b_lim_hpx_map)))
		self.assertLess(np.max(np.abs(orig_map-l_b_lim_hpx_map)),
			np.max(np.abs(orig_map-s_b_lim_hpx_map)))

	def test_subsampling(self):
		# Test that subsampling does not reduce the quality of the reconstruction
		cmb_maps_path = 'test_data/'
		hpx_map_file = cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = cmb_maps_path + 'gmca_test_full_sim_90'
		sub_samp_map = cmb_maps_path + 'gmca_test_full_sim_90_GHZ_sub_recon.fits'
		full_samp_map = cmb_maps_path + 'gmca_test_full_sim_90_GHZ_full_recon.fits'
		over_samp_map = cmb_maps_path + 'gmca_test_full_sim_90_GHZ_over_recon.fits'
		wav_b = 3
		min_scale = 1
		nside = 128

		wav_t_s0 = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			128,samp=0)	
		wav_t_s1 = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			128,samp=1)	
		wav_t_s2 = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			128,samp=2)	

		wav_coeff = wav_t_s0.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_s0.get_map_from_wavelet_coeff(sub_samp_map,nside,wav_map_prefix,
			wav_coeff)
		#wav_t_s0.clean_prefix(wav_map_prefix)

		wav_coeff = wav_t_s1.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_s1.get_map_from_wavelet_coeff(full_samp_map,nside,wav_map_prefix,
			wav_coeff)
		#wav_t_s1.clean_prefix(wav_map_prefix)

		wav_coeff = wav_t_s2.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_s2.get_map_from_wavelet_coeff(over_samp_map,nside,wav_map_prefix,
			wav_coeff)
		#wav_t_s2.clean_prefix(wav_map_prefix)

		s0_hpx_map = hp.read_map(sub_samp_map,verbose=False)
		s1_hpx_map = hp.read_map(full_samp_map,verbose=False)
		s2_hpx_map = hp.read_map(over_samp_map,verbose=False)

		# Allow for a small amount of floating point error
		self.assertLess(np.sum(np.abs(s0_hpx_map-s1_hpx_map)),0.1)
		self.assertLess(np.sum(np.abs(s1_hpx_map-s2_hpx_map)),0.1)
