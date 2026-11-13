import numpy as np

def cic(f, M):
    '''Transfer function for a 1st order CIC decimator
    Args:
        f: low rate normalized frequency [0,0.5]
        M: decimation ratio
    '''
    f = np.asarray(f, float)
    num = np.where(f != 0, np.sin(np.pi*f),     1)
    den = np.where(f != 0, M*np.sin(np.pi*f/M), 1)
    return num/den

def chebyshev(x, n):
    '''Nth order chebyshev polynomial in x'''
    x = np.asarray(x, float)
    t0, t1 = np.ones_like(x), x
    if n == 0:
        return t0
    for _ in range(n-1):
        t0, t1 = t1, 2*x*t1 - t0
    return t1

def chebyshev_cic(f, M, N, gamma):
    return chebyshev(gamma*cic(f,M), N) / chebyshev(gamma, N)

def none(f):
    return 1
