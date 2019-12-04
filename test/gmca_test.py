from hgmca.gmca import update_S
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
		
		return






