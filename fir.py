import numpy as np

def fir_tf(h, f):
    r'''FIR transfer function with taps h: |\sum_{n=0}^{N-1} h_n e^{-2\pi jnf}|'''
    n = np.arange(len(h))
    return np.abs(np.exp(-2*np.pi*1j * np.outer(f,n)) @ h)


