from hgmca import helpers
import numpy as np
import unittest


class HelpersTests(unittest.TestCase):
	# A set of tests to verify that the basic functionality of gmca is working
	# as expected.

	def test_A_norm(self):
		# Check that for multiple random values A_norm behaves as desired
		n_A_test = 10
		n_freqs = 8
		n_sources = 5

		for _ in range(n_A_test):
			A = np.random.randn(n_freqs*n_sources).reshape((n_freqs,n_sources))
			helpers.A_norm(A)
			for i in range(n_sources):
				self.assertAlmostEqual(np.sum(np.square(A[:,i])),1)

	def test_nan_to_num(self):
		# Check that to nans remain
		mat = np.ones((100,100))
		for i in range(len(mat)):
			mat[i,np.random.randint(0,mat.shape[1])] = np.nan

		helpers.nan_to_num(mat)
		self.assertEqual(np.sum(np.isnan(mat)),0)
		self.assertEqual(np.sum(mat),9900)
