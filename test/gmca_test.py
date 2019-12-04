from gmca import gmca, gmca_numba
import numpy as np
from test_helpers import toy_data, cmb_test_data
import scipy.linalg as lng 
import unittest
import wavelet, helpers


class GmcaTests(unittest.TestCase):

	def test_sparsity(self):
		n_iterations = 1
		n_sources = 5
		enforce_nn_A = True
		for i in range(1,4):
			X,A_org,S_org = toy_data((10,1000),n_sources,sparsity=0.01, 
				ret_A=True,ret_S=True, seed=i)
			low_lam_s = [0.1,0.2,0.3]
			low_sparsity_tot = 0
			low_A_mat_dif = 0
			for lam_s in low_lam_s:
				np.random.seed(i)
				A, S, _ = gmca(X, n_sources, n_iterations=n_iterations*1500,
					enforce_nn_A=enforce_nn_A, ret_loss = True, lam_s=lam_s)
				low_sparsity_tot += np.sum(np.abs(S))
				low_A_mat_dif = np.linalg.norm(A-A_org)
			
			large_lam_s = [15,17,19]
			large_sparsity_tot = 0
			large_A_mat_dif = 0
			for lam_s in large_lam_s:
				np.random.seed(i)
				A, S, _ = gmca(X, n_sources, n_iterations=n_iterations*1500,
					enforce_nn_A=enforce_nn_A, ret_loss = True, lam_s=lam_s)
				large_sparsity_tot += np.sum(np.abs(S))
				large_A_mat_dif = np.linalg.norm(A-A_org)


			self.assertGreater(low_sparsity_tot,large_sparsity_tot)
			self.assertGreater(low_A_mat_dif,large_A_mat_dif)





