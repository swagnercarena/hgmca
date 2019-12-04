import numpy as np
import numba

@numba.jit(nopython=True)
def update_S(S,A,A_R,R_i,A_i,lam_s,i):
	"""	Update a row of S according to the closed form solution.
	
		Paramters:
			S (np.array): The current value of the matrix S. The row i will be
				updated.
			A (np.array): The current value of the matrix A.
			A_R (np.array): A pre-allocated array that will be used for the
				A matrix calculation. Must have the shape (1,X.shape[1]).
			R_i (np.array): The remainder of the data after removing the rest
				of the sources.
			A_i (np.array): A pre-allocated array that will be used for the
				A matrix calculation. Must have the shape (A.shape[0],1).
			lam_s (float): The lambda parameter for the sparsity l1 norm.
			i (int): The index of the source to be updated.

		Notes:
			The S matrix row will be updated in place.
	"""
	# See paper for derivation of the update formula.
	A_i += np.expand_dims(A[:,i], axis=1)
	np.dot(A_i.T,R_i,out=A_R)
	# We do a soft thresholding to deal with the fact that there is no
	# gradient for the l1 norm.
	# This is fine to do with for loops inside jit.
	for j in range(A_R.shape[1]):
		if abs(A_R[0,j]) < lam_s:
			A_R[0,j] = 0
	S[i] = A_R - lam_s*np.sign(A_R)
	# Reset A_i
	A_i *= 0

@numba.jit(nopython=True)
def update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i):
	"""	Update a column of A according to the closed form solution.
	
		Paramters:
			S (np.array): The current value of the matrix S. 
			A (np.array): The current value of the matrix A. The column i 
				will be updated.
			R_i (np.array): The remainder of the data after removing the rest
				of the sources.
			lam_p ([float,...]): A n_sources long array of prior for each of 
				the columns of A_p. This allows for a different lam_p to be 
				applied to different columns of A_p.
			A_p (np.array): A matrix prior for the CGMCA calculation. 
			enforce_nn_A (bool): a boolean that determines if the mixing matrix 
				will be forced to only have non-negative values.
			i (int): The index of the source to be updated.

		Notes:
			The A matrix column will be updated in place.
	"""
	# See the paper for the update formula
	S_i = np.expand_dims(S[i],axis=1)
	A[:,i] = np.dot(R_i,S_i)[:,0] + lam_p[i] * A_p[:,i]
	
	if enforce_nn_A:
		A *= (A>0)
	if np.sum(A[:,i]) != 0:
		# Rescale A norm of A. Don't bother rescaling S since we are 
		# about to recalculate it. 
		A[:,i] = A[:,i]/np.linalg.norm(A[:,i])

@numba.jit(nopython=True)
def gmca_numba(X, n_sources, n_iterations, A, S, A_p, lam_p, 
	enforce_nn_A = True, lam_s = 1, ret_min_rmse=True, min_rmse_rate=0,seed=0):
	""" Run the base gmca algorithm on X using lasso shooting to solve for the
		closed form of the L1 sparsity term.

		Parameters:
			X (np.array): A numpy array with dimensions number_of_maps (or 
				frequencies) x number of data points (wavelet coefficients) per 
				map.
			n_sources (int): the number of sources to attempt to extract from the
				data.
			n_iterations (int): the number of iterations of coordinate descent to 
				conduct.
			A (np.array): an initial value for the matrix A. Will be 
				overwritten.
			S (np.array): an initial value for the matrix S. Will be 
				overwritten.
			A_p (np.array): A matrix prior for the CGMCA calculation. 
			lam_p ([float,...]): A n_sources long array of prior for each of 
				the columns of A_p. This allows for a different lam_p to be 
				applied to different columns of A_p.
			enforce_nn_A (bool): a boolean that determines if the mixing matrix 
				will be forced to only have non-negative values.
			lam_s (float): The lambda parameter for the sparsity l1 norm.
			ret_min_rmse (bool): A boolean parameter that decides if the minimum 
				rmse error solution for S will be returned. This will give best 
				CMB reconstruction but will not return the minimum of the loss
				function.
			min_rmse_rate (int): How often the source matrix will be set to the 
				minimum rmse solution. 0 will never return min_rmse within the 
				gmca optimization.
		
		Notes:
			A and S will be updated in place.
	"""

	# Set the random seed for reproducability
	if seed>0:
		np.random.seed(seed)

	# Pre-allocate R_i, AS, A_R, and A_i to speed up computation
	R_i = np.zeros(X.shape)
	AS = np.zeros(X.shape)
	A_R = np.zeros((1,X.shape[1]))
	A_i = np.zeros((A.shape[0],1))

	for iteration in range(n_iterations):
		# At each iteration we select a random ordering to update the sources
		source_perm = np.random.permutation(n_sources)
		for i in source_perm:
			# Calculate a remainder term for the data yet unexplained by 
			# the other sources. Avoid allocating new memory.
			np.outer(A[:,i],S[i],out=R_i)
			R_i+=X
			np.dot(A,S,out=AS)
			R_i-=AS

			# Carry out optimization calculation for column of A
			update_A(S,A,R_i,lam_p,A_p,enforce_nn_A,i)

			# Carry out the S update step.
			update_S(S,A,A_R,R_i,A_i,lam_s,i)

		if min_rmse_rate and iteration%min_rmse_rate == 0:
			np.dot(np.linalg.pinv(A),X,out=S)

	# Only in the context of HGMCA - where the output will be the input to a
	# future GMCA call - should this be set to false.
	if ret_min_rmse:
		np.dot(np.linalg.pinv(A),X,out=S)

