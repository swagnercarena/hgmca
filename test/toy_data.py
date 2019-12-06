import numpy as np
from hgmca import helpers

# A simple code that generates some random toy data.
def toy_data(dim, num_sources, sparsity=0.1, ret_A=False, ret_S=False, 
	seed=None):
	if seed is not None:
		np.random.seed(seed)
	A = np.random.rand(dim[0],num_sources)
	S = np.zeros((num_sources,dim[1]))
	wav_ind = list(range(dim[1]))
	for s in range(num_sources):
		s_ind = np.random.choice(wav_ind,int(dim[1]*sparsity),replace=False)
		S[s,s_ind] = (np.random.randn(int(dim[1]*sparsity))+
			np.random.randn(1)*4)*200
	helpers.A_norm(A)
	#print(A)
	if ret_A and ret_S:
		return np.dot(A,S),A,S
	if ret_A:
		return np.dot(A,S),A
	if ret_S:
		return np.dot(A,S),S
	return np.dot(A,S)

# Generates toy data with significant local variation in the mixing matrix.
def toy_data_local(dim, num_sources, lev, cmb_c, scales=1, sparsity=0.1, 
	seed=None):
	if seed is not None:
		np.random.seed(seed)
	# Begin by initializing the sources using random indices and respecting the
	# sparsity request.
	S_scale = [np.zeros((num_sources,dim[1])) for _ in range(scales)]
	for scale in range(scales):
		wav_ind = list(range(dim[1]))
		for s in range(num_sources):
			s_ind = np.random.choice(wav_ind,int(dim[1]*sparsity),replace=True)
			S_scale[scale][s,s_ind] = (np.random.randn(int(dim[1]*sparsity))+
				np.random.rand(1)*4)*200

	# Initialize the data and the true reconstruction of the cmb to zero.
	X_scale = [np.zeros(dim) for _ in range(scales)]
	true_recon = [np.zeros(dim) for _ in range(scales)]

	# We want the cmb column to be consistent so we set it here.
	A_cmb = np.random.rand(dim[0])

	# Move through the map and build both our true reconstruction and our
	# data using a varying mixing matrix.
	n_patches = 12*(4**(lev-1))
	pix_per_patch = int(dim[1]/n_patches)
	start = 0
	A_list = list()
	for patch in range(n_patches): 
		A = np.random.rand(dim[0],num_sources)
		# Enforce that the cmb column never changes.
		A[:,cmb_c] = A_cmb
		helpers.A_norm(A)
		for scale in range(scales):
			X_scale[scale][:,start:start+pix_per_patch] = np.dot(
				A,S_scale[scale][:,start:start+pix_per_patch])
			true_recon[scale][:,start:start+pix_per_patch] = np.outer(
				A[:,cmb_c],S_scale[scale][cmb_c,start:start+pix_per_patch])
		A_list.append(A)
		start = start + pix_per_patch
	
	return X_scale,A_list,S_scale,true_recon

# Generates toy data with significant local variation in the mixing matrix but
# similarity among certain clusters of data.
def toy_data_grouped_local(dim, num_sources, l_divs, n_clusters, cmb_c, 
	sparsity=0.1):
	A_list = list()
	S = np.zeros((num_sources,dim[1]))
	wav_ind = list(range(dim[1]))
	for s in range(num_sources):
		s_ind = np.random.choice(wav_ind,int(dim[1]*sparsity),replace=False)
		S[s,s_ind] = (np.random.randn(int(dim[1]*sparsity))+
			np.random.randn(1)*4)*200
	X = np.zeros(dim)
	true_recon = np.zeros(dim)
	start = 0
	# We want the cmb column to be consistent
	A_cmb = np.random.rand(dim[0])
	# We generate the matrices from which the individual clusters will be 
	# pulled
	A_clust_seed = np.random.rand(n_clusters,dim[0],num_sources)
	clust = 0
	mat_per_clust = int(np.ceil(len(l_divs)/n_clusters))
	for div in l_divs:
		A = (A_clust_seed[clust//mat_per_clust]+np.random.rand(dim[0],
			num_sources))
		A[:,cmb_c] = A_cmb
		helpers.A_norm(A)
		X[:,start:start+div] = np.dot(A,S[:,start:start+div])
		true_recon[:,start:start+div] = np.outer(A[:,cmb_c],
			S[cmb_c,start:start+div])
		A_list.append(A)
		start = start + div
		clust = clust + 1
	
	return X,A_list,S,true_recon


