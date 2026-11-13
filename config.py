import numpy as np
import pre_filters

attenuation = -144 # 24 bits in dB
ntaps = 32         # length of FIR taps h

# normalized frequency [0, 0.5] keep a gap between edges
f_pass = 0.045 # 0.09
f_trans_l = 0.14
f_trans_r = 0.20
f_stop = 0.25

def H(f):
    '''The selected pre-filter transfer function'''
    return pre_filters.chebyshev_cic(f, M=64, N=7, gamma=8)

def passband_mag(f):
    return 1/H(f)

def transitionband_mag(f):
    return 0

def stopband_mag(f):
    return 0

def passband_weight(f):
    return 1000

def transitionband_weight(f):
    return H(1-f)/10**(attenuation/20)

def stopband_weight(f):
    return H(f)/10**(attenuation/20)

bands = (
    [0,         f_pass,    passband_mag,       passband_weight],
    [f_trans_l, f_trans_r, transitionband_mag, transitionband_weight],
    [f_stop,    0.5,       stopband_mag,       stopband_weight]
)

bits = 31

def quantizer_weights(f):
    return np.where(f <= f_stop, 1,
           np.where(f <= 0.27, 5e10,
           np.where(f <= 0.30, 4e10,
           np.where(f <= 0.35, 2e10,
           np.where(f <= 0.42, 7e9,
                               1e9)))))
