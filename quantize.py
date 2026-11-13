import numpy as np
import fpylll

def taps_to_coeff(taps, dc_gain):
    h = np.asarray(taps)
    N = len(h)
    if N % 2 == 0:
        K = N//2 - 1
        a = np.zeros(K+1, dtype=h.dtype)
        a[K] = h[0]
        for k in range(1, K+1):
            a[K-k] = h[k] - a[K-k+1]
    else:
        K = (N-1)//2
        a = h[K::-1].copy() # a[k] = h[K-k] for k=0...K
    a = a * dc_gain
    return a[1:]

def coeff_to_taps(coeff, even=False):
    q = np.asarray(coeff)
    qk = q[1:]
    if even:
        a = np.concatenate(([q[0]//2 - 2*qk.sum()], qk))
        rev = a[::-1]                                # a_K, ..., a_0
        half = rev + np.concatenate(([0], rev[:-1])) # h_0, ..., h_K
        return np.concatenate((half, half[::-1]))
    else:
        half = np.concatenate((qk[::-1], [q[0] - 2*qk.sum()])) # h_0, ..., h_K
        return np.concatenate((half, half[-2::-1]))

def psi(f, k, even=False):
    if not even:
        return 2 * (np.cos(2*np.pi*k*f) - 1)
    else:
        return 4 * np.cos(np.pi*f) * (np.cos(2*np.pi*k*f) - 1)

def gram_matrix(K, W, f, even=False):
    r'''G_{i,j} = 2\int_{0}^{1/2}W(f)\psi_i(f)\psi_j(f)df '''
    return np.array(
        [[((W * psi(f,i,even) * psi(f,j,even)).sum() * (f[1]-f[0]))
        for j in range(1,K+1)] for i in range(1,K+1)]
    )

def sphere_decoding(taps, W, f, q0):

    # compute the lattice and LLL reduce it
    N = len(taps)
    K = N//2 - 1 if N % 2 == 0 else (N-1)//2
    G = gram_matrix(K, W, f, even=(N % 2 == 0)) # TODO: decide on inputs
    L = np.linalg.cholesky(G) # G = LL^T
    B = fpylll.IntegerMatrix.from_matrix(
        [[int(v) for v in row] for row in np.rint(L * 2**30)]
    ) # cast L into an int matrix for fpylll
    U = fpylll.IntegerMatrix.identity(K)
    fpylll.LLL.reduction(B, U)

    # solve the CVP in error units
    a = taps_to_coeff(taps, dc_gain=q0)
    a0 = np.round(a)
    target = (a-a0) @ L
    lattice_point = np.array(fpylll.CVP.closest_vector(B, tuple(int(x * 2**30) for x in target)), dtype=float)

    # convert back to full scale units
    B = np.array([[B[i, j] for j in range(K)] for i in range(K)], dtype=float)
    U = np.array([[U[i, j] for j in range(K)] for i in range(K)], dtype=np.int64)
    z = np.rint(np.linalg.solve(B.T, lattice_point)).astype(np.int64)
    q = a0.astype(np.int64) + z @ U
    return q

def quantize(taps, W, f, bits):
    '''quantized floating point taps to some number of signed bits'''
    q0 = 2**bits
    q = sphere_decoding(taps, W, f, q0) # q_1, ..., q_K
    q = np.concatenate(([q0], q))       # q_0 = DC gain
    return coeff_to_taps(q, even=len(taps) % 2 == 0)
