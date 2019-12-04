from hgmca.gmca import update_S, update_A, gmca_numba
from hgmca.helpers import A_norm
import numpy as np
import unittest


class GmcaTests(unittest.TestCase):
	# A set of tests to verify that the basic functionality of gmca is working
	# as expected.

	def test_update_S(self):
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
				update_S(S,A,A_R,R_i,A_i,lam_s,i)

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
				update_S(S,A,A_R,R_i,A_i,lam_s,i)
				S_check = np.dot(A[:,i],R_i)
				S_check[np.abs(S_check)<lam_s] = 0
				S_check -= lam_s*np.sign(S_check)
				self.assertAlmostEqual(np.max(np.abs(S[i]-S_check)),0)
		
	def test_update_A(self):
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
		A_norm(A_p)
		enforce_nn_A = False

		# Test results for various values of lam_p
		lam_p_tests = [0.0,0.1,1,2,100,200,1000,1e8]
		for lam_p_val in lam_p_tests:
			lam_p = [lam_p_val] * n_sources
			for i in range(n_sources):
				update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i)

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
				update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i)

				# Make sure that all the columns have the correct value.
				check_A = np.ones(n_freqs) * n_wavs
				check_A += lam_p_val*A_p[:,i]
				check_A[check_A<0] = 0
				if np.sum(check_A) > 0:
					check_A /= np.linalg.norm(check_A)
				self.assertAlmostEqual(np.max(np.abs(A[:,i] - check_A)),0)

	def test_ret_min_rmse(self):
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
		gmca_numba(np.array(X), n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=True)

		# Check that GMCA returns the minimum RMSE solution
		self.assertAlmostEqual(np.sum(S),np.sum(np.dot(np.linalg.pinv(A),X)))

		# Reset A and S
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Re-run GMCA without ret_min_rmse
		gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=False)

		# Check that GMCA does not return the min_rmse solution
		self.assertNotEqual(np.sum(S),np.sum(np.dot(np.linalg.pinv(A),X)))

	def test_min_rmse_rate(self):
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
		gmca_numba(np.array(X), n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=False, min_rmse_rate=n_iterations)

		# Check that GMCA returns the minimum RMSE solution
		self.assertAlmostEqual(np.max(np.abs(S-np.dot(np.linalg.pinv(A),X))),
			0)

		# Reset A and S
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Re-run GMCA without ret_min_rmse
		gmca_numba(np.array(X), n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=False, min_rmse_rate=n_iterations-1)

		# Check that GMCA does not return the min_rmse solution
		self.assertGreater(np.mean(np.abs(S-np.dot(np.linalg.pinv(A),X))),0.1)

	def test_gmca_end_to_end(self):
		# test that gmca works end to end, returning reasonable results
		np.random.seed(5)
		freq_dim = 10
		pix_dim = 100
		n_iterations = 50
		n_sources = 5
		lam_p = [0.0]*5

		# Generate ground truth A and S 
		A_org = np.abs(np.random.normal(size=(freq_dim,n_sources)))
		S_org = np.random.normal(loc=1000,size=(n_sources,pix_dim))
		X = np.dot(A_org,S_org)

		# Initialize A and S for GMCA
		A_p = np.ones(A_org.shape)
		A = np.ones(A_org.shape)
		S = np.ones(S_org.shape)

		# Run GMCA
		gmca_numba(np.array(X), n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False)

		err1 = np.sum(np.abs(np.dot(A,S)-X))
		# Continue GMCA
		gmca_numba(np.array(X), n_sources, n_iterations, A, S, A_p, lam_p, 
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False)
		err2 = np.sum(np.abs(np.dot(A,S)-X))

		self.assertGreater(err1,err2)

		gmca_numba(np.array(X), n_sources, 200, A, S, A_p, lam_p, 
			ret_min_rmse=False, min_rmse_rate=n_iterations, enforce_nn_A=False)

		self.assertLess(np.sum(np.abs(np.dot(A,S)-X)),1e-3)
