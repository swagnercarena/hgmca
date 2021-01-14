import numpy as np
import numba


@numba.njit
def A_norm(A):
	"""Normalizes each column of the matrix A.

	Parameters:
		A (np.array): The mixing matrix to be normalized. Must be a numpy array.
	"""
	for i in range(len(A[0])):
		if np.sum(A[:,i]) != 0:
			A[:,i] = A[:,i]/np.linalg.norm(A[:,i])


@numba.njit
def nan_to_num(mat):
	"""Converts all of the nan values in the matrix to 0.

	Parameters:
		mat (np.array): The matrix to remove the nans from.
	"""
	for i in range(len(mat)):
		for j in range(mat.shape[1]):
			if np.isnan(mat[i,j]):
				mat[i,j] = 0
