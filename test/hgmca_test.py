from hgmca import hgmca_core, wavelets_hgmca, helpers
import numpy as np
import unittest
import os
import numba


class HGMCATests(unittest.TestCase):

	def setUp(self, *args, **kwargs):
		self.root_path = (os.path.dirname(os.path.abspath(__file__))+
			'/test_data/')
		# Remember, healpy expects radians but we use arcmins.
		self.a2r = np.pi/180/60
		self.m_level = 3
		self.wav_class = wavelets_hgmca.WaveletsHGMCA(self.m_level)
		np.random.seed(5)

	def test_allocate_A_hier(self):
		# Test that the right arrays are initialized.
		m_level = 4
		A_shape = (5,5)
		A_init = None
		A_hier_list = hgmca_core.allocate_A_hier(m_level,A_shape)

		for level, A_hier in enumerate(A_hier_list):
			self.assertEqual(len(A_hier),wavelets_hgmca.level_to_npatches(
				level))
			self.assertTupleEqual(A_hier[0].shape,A_shape)
			for A_patch in A_hier:
				np.testing.assert_equal(A_patch,
					np.ones(A_shape)/np.sqrt(5))

		# Repeat the same but with an A_init now
		A_init = np.random.rand(A_shape[0]*A_shape[1]).reshape(A_shape)
		A_hier_list = hgmca_core.allocate_A_hier(m_level,A_shape,
			A_init=A_init)
		for level, A_hier in enumerate(A_hier_list):
			self.assertEqual(len(A_hier),wavelets_hgmca.level_to_npatches(
				level))
			self.assertTupleEqual(A_hier[0].shape,A_shape)
			for A_patch in A_hier:
				np.testing.assert_equal(A_patch,A_init)

	def test_allocate_S_level(self):
		m_level = 4
		n_sources = 5
		X_level = numba.typed.List()
		X_level.append(np.empty((0,0,0)))
		for level in range(m_level):
			npatches = wavelets_hgmca.level_to_npatches(level+1)
			X_level.append(np.ones((npatches,10,np.random.randint(5,10))))
		S_level = hgmca_core.allocate_S_level(m_level,X_level,n_sources)

		self.assertEqual(S_level[0].size,0)
		for level in range(1,m_level+1):
			npatches = wavelets_hgmca.level_to_npatches(level)
			self.assertTupleEqual(S_level[level].shape,(npatches,n_sources,
				X_level[level].shape[2]))

	def test_convert_wav_to_X_level(self):
		# Test that the wav_analysis maps are correctly converted
		input_map_path = self.root_path + 'gmca_test_full_sim_90_GHZ.fits'
		input_map_path_256 = (self.root_path +
			'gmca_test_full_sim_90_GHZ_256.fits')
		input_maps_dict = {
			'30':{'band_lim':256,'fwhm':33,'path':input_map_path,
				'nside':128},
			'44':{'band_lim':512,'fwhm':5,'path':input_map_path_256,
				'nside':256}}
		output_maps_prefix = self.root_path + 's2dw_test'
		scale_int = 2
		j_min = 1

		# Generate the wavelet maps using the python code
		wav_analysis_maps = self.wav_class.multifrequency_wavelet_maps(
			input_maps_dict,output_maps_prefix,scale_int,j_min)
		X_level = hgmca_core.convert_wav_to_X_level(wav_analysis_maps)

		self.assertEqual(X_level[0].size,0)
		self.assertEqual(X_level[1].size,0)
		np.testing.assert_equal(X_level[2],wav_analysis_maps['2'])
		np.testing.assert_equal(X_level[3],wav_analysis_maps['3'])

		os.remove(output_maps_prefix+'30_scaling.fits')
		for j in range(j_min,9):
			os.remove(output_maps_prefix+'30_wav_%d.fits'%(j))

		os.remove(output_maps_prefix+'44_scaling.fits')
		for j in range(j_min,10):
			os.remove(output_maps_prefix+'44_wav_%d.fits'%(j))

	def test_get_A_prior(self):
		# Test that the correct prior is extracted by comparing to manual
		# values.
		n_freqs = 8
		n_sources = 5
		A_shape = ((n_freqs,n_sources))
		A_size = A_shape[0]*A_shape[1]
		A_hier_list = numba.typed.List()
		# Make our A_hier_list
		A_hier_list.append(np.random.rand(A_size).reshape((1,)+A_shape))
		A_hier_list.append(np.random.rand(12*A_size).reshape((12,)+A_shape))
		A_hier_list.append(np.random.rand(48*A_size).reshape((48,)+A_shape))
		A_hier_list.append(np.random.rand(192*A_size).reshape((192,)+
			A_shape))

		# Now test our edge cases
		lam_hier = np.ones(n_sources)
		level = 0
		patch = 0
		A_prior = hgmca_core.get_A_prior(A_hier_list,level,patch,lam_hier)
		np.testing.assert_almost_equal(A_prior,np.sum(A_hier_list[1],axis=0))

		level = 3
		patch = 191
		A_prior = hgmca_core.get_A_prior(A_hier_list,level,patch,lam_hier)
		np.testing.assert_almost_equal(A_prior,A_hier_list[2][-1])

		# And now an inbetween case for each level
		level = 1
		patch = 1
		A_prior = hgmca_core.get_A_prior(A_hier_list,level,patch,lam_hier)
		A_manual = A_hier_list[0][0]
		A_manual += np.sum(A_hier_list[2][4:8],axis=0)
		np.testing.assert_almost_equal(A_prior,A_manual)

		level = 2
		patch = 12
		A_prior = hgmca_core.get_A_prior(A_hier_list,level,patch,lam_hier)
		A_manual = A_hier_list[1][3]
		A_manual += np.sum(A_hier_list[3][48:52],axis=0)
		np.testing.assert_almost_equal(A_prior,A_manual)

	def test_hgmca_epoch_numba(self):
		# Test that the hgmca optimization returns roughly what we would
		# expect for a small lam_s
		n_freqs = 8
		n_sources = 5
		n_wavs = 256
		m_level = 2
		lam_s = 1e-6
		lam_hier = np.zeros(n_sources)
		lam_global = np.zeros(n_sources)

		# The true values
		s_mag = 100
		A_org = np.random.rand(n_freqs*n_sources).reshape((n_freqs,n_sources))
		helpers.A_norm(A_org)
		S_org = np.random.rand(n_sources*n_wavs).reshape((n_sources,n_wavs))
		S_org *= s_mag

		# Allocate what we need
		X_level = numba.typed.List()
		X_level.append(np.empty((0,0,0)))

		# Level 1
		npatches = wavelets_hgmca.level_to_npatches(1)
		X_level.append(np.zeros((npatches,n_freqs,n_wavs)))
		X_level[1][:] += np.dot(A_org,S_org)

		# Level 2
		npatches = wavelets_hgmca.level_to_npatches(2)
		X_level.append(np.zeros((npatches,n_freqs,n_wavs)))
		X_level[2][:] += np.dot(A_org,S_org)

		# The rest
		A_hier_list = hgmca_core.allocate_A_hier(m_level,(n_freqs,n_sources))
		S_level = hgmca_core.allocate_S_level(m_level,X_level,n_sources)
		A_global = np.random.rand(n_freqs*n_sources).reshape(
			n_freqs,n_sources)
		helpers.A_norm(A_global)

		# Run hgmca
		min_rmse_rate = 5
		enforce_nn_A = True
		seed = 5
		n_epochs = 5
		n_iterations = 30
		hgmca_core.hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,
			lam_global,S_level,n_epochs,m_level,n_iterations,lam_s,seed,
			enforce_nn_A,min_rmse_rate)

		for level in range(1,3):
			for patch in range(wavelets_hgmca.level_to_npatches(level)):
				np.testing.assert_almost_equal(X_level[level][patch],
					np.dot(A_hier_list[level][patch],S_level[level][patch]),
					decimal=3)
