import numpy as np
import healpy as hp
from hgmca import wavelets
import unittest, os
import scipy.integrate as integrate


class TestBaseFunctions(unittest.TestCase):

	def test_k_sdw(self):
		# Test that the modified Schwartz function is returning the
		# expected values
		scale_int = 2
		self.assertAlmostEqual(wavelets.k_sdw(0.3,scale_int),0)
		self.assertAlmostEqual(wavelets.k_sdw(0.6,scale_int),
			0.073228222705679)
		self.assertAlmostEqual(wavelets.k_sdw(0.9,scale_int),
			0.04881881513711933)
		self.assertAlmostEqual(wavelets.k_sdw(0.95,scale_int),
			0.00406938962049771)

	def test_kappa_integral(self):
		# Test that the integration is equivalent to scipy integration.
		scale_int = 2
		n_quads = 1000
		lower = 0.7
		upper = 1.0

		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		lower = 0.6
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		lower = 0.5
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

		scale_int = 3
		self.assertAlmostEqual(wavelets.kappa_integral(lower,upper,
			n_quads,scale_int),integrate.quad(wavelets.k_sdw,lower,upper,
			args=(scale_int))[0])

	def test_j_max(self):
		# Test that j_max returns the expected results
		self.assertEqual(wavelets.calc_j_max(128,2),7)
		self.assertEqual(wavelets.calc_j_max(256,2),8)
		self.assertEqual(wavelets.calc_j_max(255,2),8)
		self.assertEqual(wavelets.calc_j_max(2048,2),11)

	def test_phi2_s2dw(self):
		# Test that the phi2_s2dw integral behaves as it should.
		scale_int = 2
		band_lim = 1024
		j_max = wavelets.calc_j_max(band_lim,scale_int)
		phi2 = np.zeros((j_max+2)*band_lim)
		n_quads = 1000
		norm = wavelets.kappa_integral(1.0/scale_int,1.0,n_quads,scale_int)

		wavelets.phi2_s2dw(phi2,band_lim,scale_int,n_quads)
		self.assertAlmostEqual(np.max(phi2),1)
		self.assertEqual(np.min(phi2),0)

		# We'll go an manually test the values for a few scales
		correct_values = np.zeros(band_lim)
		correct_values[0] = 1
		np.testing.assert_almost_equal(phi2[0:band_lim],correct_values)

		correct_values = np.zeros(band_lim)
		correct_values[0:2] = 1
		np.testing.assert_almost_equal(phi2[band_lim:2*band_lim],
			correct_values)

		correct_values = np.zeros(band_lim)
		correct_values[0:2] = 1
		correct_values[2] = wavelets.kappa_integral(2/4,1.0,n_quads,
			scale_int)/norm
		correct_values[3] = wavelets.kappa_integral(3/4,1.0,n_quads,
			scale_int)/norm
		np.testing.assert_almost_equal(phi2[2*band_lim:3*band_lim],
			correct_values)

		correct_values = np.ones(band_lim)
		np.testing.assert_almost_equal(phi2[(j_max+1)*band_lim:],
			correct_values)


class TestAxisymWaveletTransformation(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TestAxisymWaveletTransformation, self).__init__(*args, **kwargs)
		# Open up the config file.
		self.cmb_maps_path = os.path.join(
			os.path.dirname(os.path.abspath(__file__)),'test_data/')

	def test_reconstruction(self):
		# The reconstruction still seems to have considerable error, so this simply
		# checks that the error is small.
		hpx_map_file = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = self.cmb_maps_path + 'gmca_90'
		recon_map_file = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_recon.fits'
		wav_b = 3
		min_scale = 1
		band_lim = 128*3
		nside = 128
		# No subsampling here
		samp = 1
		wav_t = wavelets.AxisymWaveletTransformation(wav_b,min_scale,band_lim,
			samp=samp)
		wav_coeff = wav_t.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t.get_map_from_wavelet_coeff(recon_map_file,nside,wav_map_prefix,
			wav_coeff)
		wav_t._clean_prefix(wav_map_prefix)
		orig_map = hp.read_map(hpx_map_file,verbose=False)
		recon_map = hp.read_map(recon_map_file,verbose=False)

		self.assertLess(np.mean(np.abs(orig_map-recon_map))/
			np.mean(np.abs(orig_map)),0.1)
		self.assertLess(np.max(np.abs(orig_map-recon_map))/
			np.max(np.abs(orig_map)),0.1)

	def test_recon_bandlim(self):
		# Tests how the reconstruction improves with increasing bandlimit
		hpx_map_file = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = self.cmb_maps_path + 'gmca_90'
		small_b_lim_map = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_128_recon.fits'
		large_b_lim_map	= self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_420_recon.fits'
		wav_b = 3
		min_scale = 1
		nside = 128
		samp = 1

		wav_t_128  = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			128,samp=samp)
		wav_coeff = wav_t_128.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_128.get_map_from_wavelet_coeff(small_b_lim_map,nside,wav_map_prefix,
			wav_coeff)
		wav_t_128._clean_prefix(wav_map_prefix)

		wav_t_420 = wavelets.AxisymWaveletTransformation(wav_b,min_scale,
			420,samp=samp)
		wav_coeff = wav_t_420.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_420.get_map_from_wavelet_coeff(large_b_lim_map,nside,wav_map_prefix,
			wav_coeff)
		wav_t_420._clean_prefix(wav_map_prefix)

		s_b_lim_hpx_map = hp.read_map(small_b_lim_map,verbose=False)
		l_b_lim_hpx_map = hp.read_map(large_b_lim_map,verbose=False)
		orig_map = hp.read_map(hpx_map_file,verbose=False)

		self.assertLess(np.mean(np.abs(orig_map-l_b_lim_hpx_map)),
			np.mean(np.abs(orig_map-s_b_lim_hpx_map)))
		self.assertLess(np.max(np.abs(orig_map-l_b_lim_hpx_map)),
			np.max(np.abs(orig_map-s_b_lim_hpx_map)))

	def test_subsampling(self):
		# Test that subsampling does not reduce the quality of the reconstruction
		hpx_map_file = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ.fits'
		wav_map_prefix = self.cmb_maps_path + 'gmca_90'
		sub_samp_map = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_sub_recon.fits'
		full_samp_map = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_full_recon.fits'
		over_samp_map = self.cmb_maps_path + 'gmca_test_full_sim_90_GHZ_over_recon.fits'
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
		wav_t_s0._clean_prefix(wav_map_prefix)

		wav_coeff = wav_t_s1.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_s1.get_map_from_wavelet_coeff(full_samp_map,nside,wav_map_prefix,
			wav_coeff)
		wav_t_s1._clean_prefix(wav_map_prefix)

		wav_coeff = wav_t_s2.get_wavelet_coeff(hpx_map_file,wav_map_prefix)
		wav_t_s2.get_map_from_wavelet_coeff(over_samp_map,nside,wav_map_prefix,
			wav_coeff)
		wav_t_s2._clean_prefix(wav_map_prefix)

		s0_hpx_map = hp.read_map(sub_samp_map,verbose=False)
		s1_hpx_map = hp.read_map(full_samp_map,verbose=False)
		s2_hpx_map = hp.read_map(over_samp_map,verbose=False)

		# Allow for a small amount of floating point error
		self.assertLess(np.sum(np.abs(s0_hpx_map-s1_hpx_map)),0.1)
		self.assertLess(np.sum(np.abs(s1_hpx_map-s2_hpx_map)),0.1)
