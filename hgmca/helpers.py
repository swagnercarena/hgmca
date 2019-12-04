import numpy as np
import numba

@numba.jit(nopython=True)
def A_norm(A):
	""" A quick helper function that takes a mixing matrix A and normalizes each
		of the columns in A
		Paramters:
			A: The mixing matrix to be normalized. Must be a numpy array.
	"""
	for i in range(len(A[0])):
		if np.sum(A[:,i]) != 0:
			A[:,i] = A[:,i]/np.linalg.norm(A[:,i])