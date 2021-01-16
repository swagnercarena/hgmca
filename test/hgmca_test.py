from hgmca import hgmca_core, wavelets_hgmca, helpers, wavelets_base
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
					decimal=1)

		# Repeat the same but with strong priors. Start with the global prior
		lam_global = np.ones(n_sources)*1e12
		n_epochs = 2
		n_iterations = 5
		A_hier_list = hgmca_core.allocate_A_hier(m_level,(n_freqs,n_sources))
		S_level = hgmca_core.allocate_S_level(m_level,X_level,n_sources)
		hgmca_core.hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,
			lam_global,S_level,n_epochs,m_level,n_iterations,lam_s,seed,
			enforce_nn_A,min_rmse_rate)
		for level in range(3):
			for patch in range(wavelets_hgmca.level_to_npatches(level)):
				np.testing.assert_almost_equal(A_hier_list[level][patch],
					A_global,decimal=4)

		# The same test, but now for the hierarchy
		lam_hier = np.ones(n_sources)*1e12
		lam_global = np.zeros(n_sources)
		n_epochs = 2
		n_iterations = 5
		A_init = np.random.rand(n_freqs*n_sources).reshape((n_freqs,n_sources))
		A_hier_list = hgmca_core.allocate_A_hier(m_level,(n_freqs,n_sources),
			A_init=A_init)
		S_level = hgmca_core.allocate_S_level(m_level,X_level,n_sources)
		for level in range(3):
			for patch in range(wavelets_hgmca.level_to_npatches(level)):
				np.testing.assert_almost_equal(A_hier_list[level][patch],
					A_init,decimal=4)

	def test_save_load_numba_hier_list(self):
		# Create a A_hier_list and S_level and make sure that
		# the save load functions behave as expected.
		save_path = self.root_path
		m_level = 4
		# First check we get a value error if the files aren't
		# there.
		with self.assertRaises(ValueError):
			hgmca_core.load_numba_hier_list(save_path,m_level)
		# Now just make sure saving and loading preserves the identity
		# transform.
		A_shape = (5,5)
		n_sources = 5
		X_level = numba.typed.List()
		X_level.append(np.empty((0,0,0)))
		for level in range(m_level):
			npatches = wavelets_hgmca.level_to_npatches(level+1)
			X_level.append(np.ones((npatches,10,np.random.randint(5,10))))
		A_init = np.random.rand(A_shape[0]*A_shape[1]).reshape(A_shape)
		A_hier_list = hgmca_core.allocate_A_hier(m_level,A_shape,
			A_init=A_init)
		S_level = hgmca_core.allocate_S_level(m_level,X_level,n_sources)

		hgmca_core.save_numba_hier_lists(A_hier_list,S_level,save_path)
		A_test, S_test = hgmca_core.load_numba_hier_list(save_path,m_level)

		for level in range(m_level+1):
			np.testing.assert_almost_equal(A_test[level],A_hier_list[level])
			np.testing.assert_almost_equal(S_test[level],S_level[level])

		folder_path = os.path.join(save_path,'hgmca_save')
		for level in range(m_level+1):
			os.remove(os.path.join(folder_path,'A_%d.npy'%(level)))
			os.remove(os.path.join(folder_path,'S_%d.npy'%(level)))
		os.rmdir(folder_path)

	def test_hgmca_opt(self):
		# Generate a quick approximation using the hgmca_opt code and
		# make sure it gives the same results as the core hgmca code.
		input_map_path = self.root_path + 'gmca_test_full_sim_90_GHZ.fits'
		input_maps_dict = {
			'30':{'band_lim':64,'fwhm':33,'path':input_map_path,
				'nside':128},
			'44':{'band_lim':64,'fwhm':24,'path':input_map_path,
				'nside':128},
			'70':{'band_lim':64,'fwhm':14,'path':input_map_path,
				'nside':128},
			'100':{'band_lim':256,'fwhm':10,'path':input_map_path,
				'nside':128},
			'143':{'band_lim':256,'fwhm':7.1,'path':input_map_path,
				'nside':128},
			'217':{'band_lim':256,'fwhm':5.5,'path':input_map_path,
				'nside':128}}
		output_maps_prefix = self.root_path + 'hgmca_test'
		scale_int = 2
		j_min = 1
		n_freqs = len(input_maps_dict)
		wav_analysis_maps = self.wav_class.multifrequency_wavelet_maps(
			input_maps_dict,output_maps_prefix,scale_int,j_min)

		# Run hgmca on the wavelet analysis maps
		n_sources = 5
		n_epochs = 5
		n_iterations = 2
		lam_s = 1e-3
		lam_hier = np.random.rand(n_sources)
		lam_global = np.random.rand(n_sources)
		A_global = np.random.rand(n_freqs*n_sources).reshape(
			n_freqs,n_sources)
		seed = 5
		min_rmse_rate = 2
		hgmca_dict = hgmca_core.hgmca_opt(wav_analysis_maps,n_sources,n_epochs,
			lam_hier,lam_s,n_iterations,A_init=None,A_global=A_global,
			lam_global=lam_global,seed=seed,enforce_nn_A=True,
			min_rmse_rate=min_rmse_rate,verbose=True)

		# Allocate what we need for the core code.
		X_level = hgmca_core.convert_wav_to_X_level(wav_analysis_maps)
		n_freqs = wav_analysis_maps['n_freqs']
		A_shape = (n_freqs,n_sources)
		A_hier_list = hgmca_core.allocate_A_hier(self.m_level,A_shape,
			A_init=None)
		S_level = hgmca_core.allocate_S_level(self.m_level,X_level,n_sources)
		hgmca_core.hgmca_epoch_numba(X_level,A_hier_list,lam_hier,A_global,
			lam_global,S_level,n_epochs,self.m_level,n_iterations,lam_s,seed,
			True,min_rmse_rate)

		for level in range(self.m_level+1):
			for patch in range(wavelets_hgmca.level_to_npatches(level)):
				np.testing.assert_almost_equal(A_hier_list[level][patch],
					hgmca_dict[str(level)]['A'][patch])
				if S_level[level].size == 0:
					self.assertEqual(hgmca_dict[str(level)]['S'].size,0)
				else:
					np.testing.assert_almost_equal(S_level[level][patch],
						hgmca_dict[str(level)]['S'][patch])

		# Check that saving doesn't cause issues
		n_epochs = 6
		save_dict = {'save_path':self.root_path,'save_rate':2}
		hgmca_dict = hgmca_core.hgmca_opt(wav_analysis_maps,n_sources,n_epochs,
			lam_hier,lam_s,n_iterations,A_init=None,A_global=A_global,
			lam_global=lam_global,seed=seed,enforce_nn_A=True,
			min_rmse_rate=min_rmse_rate,save_dict=save_dict,verbose=True)
		# Make sure loading works as well.
		hgmca_dict = hgmca_core.hgmca_opt(wav_analysis_maps,n_sources,n_epochs,
			lam_hier,lam_s,n_iterations,A_init=None,A_global=A_global,
			lam_global=lam_global,seed=seed,enforce_nn_A=True,
			min_rmse_rate=min_rmse_rate,save_dict=save_dict,verbose=True)

		# Delete all the files we made
		folder_path = os.path.join(save_dict['save_path'],'hgmca_save')
		for level in range(self.m_level+1):
			os.remove(os.path.join(folder_path,'A_%d.npy'%(level)))
			os.remove(os.path.join(folder_path,'S_%d.npy'%(level)))
		os.rmdir(folder_path)

		for freq in input_maps_dict.keys():
			os.remove(output_maps_prefix+freq+'_scaling.fits')
			j_max = wavelets_base.calc_j_max(input_maps_dict[freq]['band_lim'],
				scale_int)
			for j in range(j_min,j_max+1):
				os.remove(output_maps_prefix+freq+'_wav_%d.fits'%(j))

	def test_extract_source(self):
		# Test that extract source returns the correct source
		hgmca_analysis_maps = {'input_maps_dict':{},'analysis_type':'hgmca',
			'scale_int':2,'j_min':1,'j_max':9,'band_lim':128,
			'target_fwhm':1.0*self.a2r,'output_nside':128,'m_level':2}
		n_sources = 5
		n_freqs = 6
		A_truth = np.random.rand(n_sources*n_freqs).reshape(n_freqs,n_sources)
		S_truth = np.random.rand(n_sources*10).reshape(n_sources,10)
		hgmca_analysis_maps['0'] = {'A':np.copy(A_truth),
			'S':np.empty((0,0,0))}
		for level in range(1,hgmca_analysis_maps['m_level']+1):
			permute = np.random.permutation(n_sources)
			A_rand = A_truth.T[permute].T
			S_rand = S_truth[permute]
			hgmca_analysis_maps[str(level)] = {
				'A':np.repeat(A_rand[np.newaxis,:,:],
					wavelets_hgmca.level_to_npatches(level),axis=0),
				'S':np.repeat(S_rand[np.newaxis,:,:],
					wavelets_hgmca.level_to_npatches(level),axis=0)}

		A_target = A_truth[:,0]
		wav_analysis_maps = hgmca_core.extract_source(hgmca_analysis_maps,
			A_target)

		for level in range(1,hgmca_analysis_maps['m_level']+1):
			for patch in range(wavelets_hgmca.level_to_npatches(level)):
				np.testing.assert_almost_equal(
					wav_analysis_maps[str(level)][patch],
					A_truth[0,0]*S_truth[0,:])
