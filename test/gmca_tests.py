from hgmca import gmca_core, helpers, wavelets
import numpy as np
import unittest
import warnings
import os


class GmcaTests(unittest.TestCase):
	# A set of tests to verify that the basic functionality of gmca is working
	# as expected.

	def setUp(self, *args, **kwargs):
		self.root_path = os.path.dirname(os.path.abspath(__file__))+'/test_data/'
		# Remember, healpy expects radians but we use arcmins.
		self.a2r = np.pi/180/60

	def test_update_S(self):
		warnings.filterwarnings("ignore")

		# Check that the S update step works as intended.
		n_wavs = 1000
		n_freqs = 8
		n_sources = 5

		# Start by generating a our A and S matrix
		S = np.ones((n_sources,n_wavs))
		A = np.ones((n_freqs,n_sources))/np.sqrt(n_freqs)

		# Create the temporary matrices that will be used for the calculations
		A_i = np.zeros((n_freqs,1))
		A_R = np.zeros((1,n_wavs))

		# Create the remainder matrix
		R_i = np.ones((n_freqs,n_wavs))

		# Check that the calculation is correct for multiple values of lam_s
		lam_s_tests = [0.1,0.2,0.5,10,20]
		for lam_s in lam_s_tests:
			# Carry out the update step
			for i in range(n_sources):
				gmca_core.update_S(S,A,A_R,R_i,A_i,lam_s,i)

			# Make sure that all the sources are identical and that they have
			# the correct value.
			self.assertAlmostEqual(np.max(np.std(S,axis=0)),0)
			if lam_s < np.sqrt(n_freqs):
				self.assertAlmostEqual(np.max(np.abs(S-np.sqrt(n_freqs)+lam_s)),
					0)
			else:
				self.assertAlmostEqual(np.max(np.abs(S)),0)

		# Check that the calculation still holds for random values
		A = np.random.randn(n_freqs*n_sources).reshape((n_freqs,n_sources))
		R_i = np.random.randn(n_freqs*n_wavs).reshape((n_freqs,n_wavs))
		for lam_s in lam_s_tests:
			# Carry out the update step
			for i in range(n_sources):
				gmca_core.update_S(S,A,A_R,R_i,A_i,lam_s,i)
				S_check = np.dot(A[:,i],R_i)
				S_check[np.abs(S_check)<lam_s] = 0
				S_check -= lam_s*np.sign(S_check)
				self.assertAlmostEqual(np.max(np.abs(S[i]-S_check)),0)

	def test_update_A(self):
		warnings.filterwarnings("ignore")

		# Check that the A update step works as intended
		n_wavs = 1000
		n_freqs = 8
		n_sources = 5

		# Start by generating a our A and S matrix
		S = np.ones((n_sources,n_wavs))
		A = np.ones((n_freqs,n_sources))/np.sqrt(n_freqs)

		# Create the remainder matrix
		R_i = np.ones((n_freqs,n_wavs))

		# Create an A_p to use for testing
		A_p = np.random.randn(n_freqs*n_sources).reshape((n_freqs,n_sources))
		helpers.A_norm(A_p)
		enforce_nn_A = False

		# Test results for various values of lam_p
		lam_p_tests = [0.0,0.1,1,2,100,200,1000,1e8]
		for lam_p_val in lam_p_tests:
			lam_p = [lam_p_val] * n_sources
			for i in range(n_sources):
				gmca_core.update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i)

				# Make sure that all the columns have the correct value.
				check_A = np.ones(n_freqs) * n_wavs
				check_A += lam_p_val*A_p[:,i]
				check_A /= np.linalg.norm(check_A)
				self.assertAlmostEqual(np.max(np.abs(A[:,i] - check_A)),0)

		# Test that everything still holds when nonegativity is enforced
		enforce_nn_A = True
		lam_p_tests = [0.0,0.1,1,2,100,200,1000,1e8]
		for lam_p_val in lam_p_tests:
			lam_p = [lam_p_val] * n_sources
			for i in range(n_sources):
				gmca_core.update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i)

				# Make sure that all the columns have the correct value.
				check_A = np.ones(n_freqs) * n_wavs
				check_A += lam_p_val*A_p[:,i]
				check_A[check_A<0] = 0
				if np.sum(check_A) > 0:
					check_A /= np.linalg.norm(check_A)
				self.assertAlmostEqual(np.max(np.abs(A[:,i] - check_A)),0)

	def test_calculate_remainder(self):
		warnings.filterwarnings("ignore")

		# Check that the remainder is correctly calculated.
		n_wavs = 1000
		n_freqs = 8
		n_sources = 5

		# Start by generating a our A, S, and X matrices
		S = np.ones((n_sources,n_wavs))*np.sqrt(n_freqs)
		A = np.ones((n_freqs,n_sources))/np.sqrt(n_freqs)
		X = np.ones((n_freqs,n_wavs))*n_sources

		# Create the remainder matrix
		R_i = np.zeros((n_freqs,n_wavs))

		# Create AS for computation
		AS = np.zeros(X.shape)

		# Check that no matter the source, we get the correct remainder.
		for i in range(n_sources):
			gmca_core.calculate_remainder(X,S,A,AS,R_i,i)
			self.assertAlmostEqual(np.max(np.abs(R_i-1)),0)

		# Repeat the test with random matrices.
		S = np.random.randn(n_sources*n_wavs).reshape((n_sources,n_wavs))
		# Check that no matter the source, we get the correct remainder.
		for i in range(n_sources):
			gmca_core.calculate_remainder(X,S,A,AS,R_i,i)
			check_Ri = np.copy(X)
			for j in range(n_sources):
				if i == j:
					continue
				check_Ri -= np.outer(A[:,j],S[j])
			self.assertAlmostEqual(np.max(np.abs(R_i-check_Ri)),0)

	def test_ret_min_rmse(self):
		warnings.filterwarnings("ignore")

		# Check that the minimum RMSE solution is returned
		freq_dim = 10
		pix_dim = 100
		n_iterations = 50
		n_sources = 5
		lam_p = [0.0]*5

		# Generate ground truth A and S
		A_org = np.random.normal(size=(freq_dim,n_sources))
		S_org = np.random.normal(size=(n_sources,pix_dim))
		X = np.dot(A_org,S_org)

		# Initialize A and S for GMCA
		A_p = np.ones(A_org.shape)
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Run GMCA
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			ret_min_rmse=True)

		# Check that GMCA returns the minimum RMSE solution
		self.assertAlmostEqual(np.sum(S),np.sum(np.dot(np.linalg.pinv(A),X)))

		# Reset A and S
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Re-run GMCA without ret_min_rmse
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			ret_min_rmse=False)

		# Check that GMCA does not return the min_rmse solution
		self.assertNotEqual(np.sum(S),np.sum(np.dot(np.linalg.pinv(A),X)))

	def test_min_rmse_rate(self):
		warnings.filterwarnings("ignore")

		# Check that the minimum RMSE solution is returned

		freq_dim = 10
		pix_dim = 100
		n_iterations = 50
		n_sources = 5
		lam_p = [0.0]*5

		# Generate ground truth A and S
		A_org = np.random.normal(size=(freq_dim,n_sources))
		S_org = np.random.normal(size=(n_sources,pix_dim))
		X = np.dot(A_org,S_org)

		# Initialize A and S for GMCA
		A_p = np.ones(A_org.shape)
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Run GMCA
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			ret_min_rmse=False, min_rmse_rate=n_iterations)

		# Check that GMCA returns the minimum RMSE solution
		self.assertAlmostEqual(np.max(np.abs(S-np.dot(np.linalg.pinv(A),X))),
			0)

		# Reset A and S
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Re-run GMCA without ret_min_rmse
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			ret_min_rmse=False, min_rmse_rate=n_iterations-1)

		# Check that GMCA does not return the min_rmse solution
		self.assertGreater(np.mean(np.abs(S-np.dot(np.linalg.pinv(A),X))),
			1e-4)

	def test_gmca_end_to_end(self):
		warnings.filterwarnings("ignore")

		# Test that gmca works end to end, returning reasonable results.
		rseed = 5
		freq_dim = 10
		pix_dim = 100
		n_iterations = 50
		n_sources = 5
		lam_s = 1
		lam_p = [0.0]*n_sources

		# Generate ground truth A and S
		A_org = np.abs(np.random.normal(size=(freq_dim,n_sources)))
		S_org = np.random.normal(loc=1000,size=(n_sources,pix_dim))
		X = np.dot(A_org,S_org)

		# Initialize A and S for GMCA
		A_p = np.ones(A_org.shape)
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Run GMCA
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			lam_s=lam_s, ret_min_rmse=False, min_rmse_rate=2*n_iterations,
			enforce_nn_A=False,seed=rseed)

		# Save sparsity of S for later test
		sparsity_1 = np.sum(np.abs(S))
		err1 = np.sum(np.abs(np.dot(A,S)-X))

		# Continue GMCA
		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			lam_s=lam_s,ret_min_rmse=False, min_rmse_rate=2*n_iterations,
			enforce_nn_A=False,seed=rseed)
		err2 = np.sum(np.abs(np.dot(A,S)-X))

		self.assertGreater(err1,err2)

		gmca_core.gmca_numba(X, n_sources, 200, A, S, A_p, lam_p, lam_s=lam_s,
			ret_min_rmse=False, min_rmse_rate=2*n_iterations,
			enforce_nn_A=False, seed=rseed)

		self.assertLess(np.sum(np.abs(np.dot(A,S)-X)),1e-3)

		# Test that lam_s enforces sparsity end_to_end
		lam_s = 10

		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			lam_s=lam_s, ret_min_rmse=False, min_rmse_rate=2*n_iterations,
			enforce_nn_A=False, seed=rseed)

		# Save closeness to prior for later test
		A_p_val = np.sum(np.abs(A-A_p))

		self.assertLess(np.sum(np.abs(S)), sparsity_1)
		# Test that lam_p enforcces prior end_to_end

		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)
		lam_p = [100.0]*n_sources

		gmca_core.gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p,
			lam_s=lam_s, ret_min_rmse=False, min_rmse_rate=2*n_iterations,
			enforce_nn_A=False, seed=rseed)

		self.assertLess(np.sum(np.abs(A-A_p)),A_p_val)

	def test_random_seed(self):
		warnings.filterwarnings("ignore")

		# Test that setting the random seed leads to consistent results.
		freq_dim = 10
		pix_dim = 100
		n_iterations = 1
		n_sources = 5
		lam_p = [0.0]*5

		# Generate ground truth A and S
		A_org = np.abs(np.random.normal(size=(freq_dim,n_sources)))
		S_org = np.random.normal(loc=1000,size=(n_sources,pix_dim))
		X = np.dot(A_org,S_org)

		# Initialize A and S for GMCA
		A_p = np.ones(A_org.shape)
		A1 = np.ones(A_org.shape)
		S1 = np.ones(S_org.shape)
		A2 = np.ones(A_org.shape)
		S2 = np.ones(S_org.shape)
		A3 = np.ones(A_org.shape)
		S3 = np.ones(S_org.shape)

		# Run GMCA with different seeds.
		gmca_core.gmca_numba(X, n_sources, n_iterations, A1, S1, A_p, lam_p,
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False,
			seed=1)
		gmca_core.gmca_numba(X, n_sources, n_iterations, A2, S2, A_p, lam_p,
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False,
			seed=1)
		gmca_core.gmca_numba(X, n_sources, n_iterations, A3, S3, A_p, lam_p,
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False,
			seed=2)

		# Make sure the two runs with the same random seed give the same
		# answer. Given that we only ran for 1 iteration, make sure that
		# different random seeds do not give the same answer.
		self.assertAlmostEqual(np.max(np.abs(A1-A2)),0)
		self.assertAlmostEqual(np.max(np.abs(S1-S2)),0)

		self.assertGreater(np.mean(np.abs(A1-A3)),1e-4)
		self.assertGreater(np.mean(np.abs(S1-S3)),1e-4)

	def test_wrapper(self):
		# Test that the wrapper returns the same results as the numba
		# implementation
		freq_dim = 10
		pix_dim = 100
		n_iterations = 10
		n_sources = 5
		lam_p = [0.0]*5

		X = np.random.normal(loc=1000,size=(freq_dim,pix_dim))

		A_numba = np.random.normal(size=(freq_dim,n_sources))
		helpers.A_norm(A_numba)
		S_numba = np.ones((n_sources,pix_dim))

		A_p = np.random.normal(size=(freq_dim,n_sources))
		helpers.A_norm(A_p)

		lam_s_vals = [0,10]
		lam_p_vals = [0,1000]
		min_rmse_rates = [0,2]
		ret_min_rmse_vals = [True,False]
		enforce_nn_A = True

		for lam_s in lam_s_vals:
			for lam_p_val in lam_p_vals:
				lam_p = [lam_p_val] * n_sources
				for min_rmse_rate in min_rmse_rates:
					for ret_min_rmse in ret_min_rmse_vals:
						A_init = np.copy(A_numba)
						S_init = np.copy(S_numba)
						A,S = gmca_core.gmca(X, n_sources, n_iterations,
							A_init,S_init, A_p, lam_p, enforce_nn_A, lam_s,
							ret_min_rmse,min_rmse_rate, seed=2)
						gmca_core.gmca_numba(X, n_sources, n_iterations,
							A_numba,S_numba, A_p, lam_p, enforce_nn_A, lam_s,
							ret_min_rmse, min_rmse_rate, seed=2)
						self.assertAlmostEqual(np.max(np.abs(A_numba-A)),0)
						self.assertAlmostEqual(np.max(np.abs(S_numba-S)),0)

	def test_mgmca(self):
		# Test that mgmca runs the analysis and returns a viable map
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
		output_maps_prefix = self.root_path + 's2dw_test'
		analysis_type = 'mgmca'
		scale_int = 2
		j_min = 1
		lam_s=0
		wav_analysis_maps = wavelets.multifrequency_wavelet_maps(
			input_maps_dict,output_maps_prefix,analysis_type,scale_int,j_min)

		# Run mgmca on the wavelet analysis maps
		max_n_sources = 6
		n_iterations = 10
		lam_s = 0
		mgmca_dict = gmca_core.mgmca(wav_analysis_maps,max_n_sources,
			n_iterations,lam_s=lam_s)
		np.testing.assert_almost_equal(wav_analysis_maps['0'],
			np.dot(mgmca_dict['0']['A'],mgmca_dict['0']['S']),decimal=4)
		np.testing.assert_almost_equal(wav_analysis_maps['1'],
			np.dot(mgmca_dict['1']['A'],mgmca_dict['1']['S']),decimal=4)
