import numpy as np
import matplotlib.pyplot as plt

import firpm
import config
import fir
import quantize

def dB(x):
    with np.errstate(divide='ignore'):
        return 20*np.log10(np.abs(x))

def main():
    h = firpm.remez(config.ntaps, config.bands)
    h = np.array(h)
    h = h / h.sum()
    np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)
    print("h = ", ', '.join(np.format_float_positional(x) for x in h))

    f = np.arange(0, 0.5, 1e-6)
    fir_f = fir.fir_tf(h=h, f=f)
    H_f = config.H(f)
    H_a = config.H(1-f)

    ripple = np.max(np.abs(fir_f[f<config.f_pass] * H_f[f<config.f_pass] - 1))+1
    atten =  np.max(np.abs(fir_f[f>config.f_stop] * H_f[f>config.f_stop]))
    print(f"passband ripple: {dB(ripple):.2f} dB")
    print(f"stopband attenuation: {dB(atten):.1f} dB")

    #q = quantize.quantize(h, config.quantizer_weights(f), f, config.bits)
    q = np.round(2**30 * h)
    print("q = ", ", ".join(str(int(x)) for x in q))
    print(f"sum q = {q.sum()}")
    print(f"log2 sum |q| = {np.log2(np.sum(np.abs(q)))}")
    print(f"max |q| = {np.max(np.abs(q))}")
    fir_q = fir.fir_tf(h=q/q.sum(), f=f)
    fir_r = fir.fir_tf(h=np.clip(2**-17*np.round(h*2**17), -2**17, 2**17-1), f=f)

    ripple = np.max(np.abs(fir_q[f<config.f_pass] * H_f[f<config.f_pass] - 1))+1
    atten =  np.max(np.abs(fir_q[f>config.f_stop] * H_f[f>config.f_stop]))
    print(f"passband ripple: {dB(ripple):.2f} dB")
    print(f"stopband attenuation: {dB(atten):.1f} dB")


    ### PLOT ###
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 5), width_ratios=[2, 1])
    plt.sca(ax)
    #plt.plot(f, dB(fir_f), label='FIR', color='C2', linestyle='-')
    #plt.plot(f, dB(fir_r), label='FIR Rounded', color='C4', linestyle='-')
    #plt.plot(f, dB(fir_q), label='FIR Quantized', color='C4', linestyle='-')
    plt.plot(f, dB(H_f), label='H', color='C0', linestyle='-')
    plt.plot(f, dB(H_a), label='H alias', color='C0', linestyle='--')
    plt.plot(f, dB(H_f*fir_f), label='H * FIR', color='C1', linestyle='-')
    plt.plot(f, dB(H_a*fir_f), label='H alias * FIR', color='C1', linestyle='--')
    #plt.plot(f, dB(H_f*fir_r), label='H * rounded FIR', color='C3', linestyle='-')
    #plt.plot(f, dB(H_a*fir_r), label='H alias * rounded FIR', color='C3', linestyle='--')
    plt.plot(f, dB(H_f*fir_q), label='H * qFIR', color='C5', linestyle='-')
    plt.plot(f, dB(H_a*fir_q), label='H alias * qFIR', color='C5', linestyle='--')
    plt.axhline(config.attenuation, label=f'{config.attenuation}dB', color='gray', linestyle='--')
    plt.axvline(0.25, color='gray', linestyle='--')
    plt.axvspan(0,                config.f_pass,    color='gray', alpha=0.1, linewidth=0)
    plt.axvspan(config.f_trans_l, config.f_trans_r, color='gray', alpha=0.1, linewidth=0)
    plt.axvspan(config.f_stop,    0.5,              color='gray', alpha=0.1, linewidth=0)
    plt.ylim(-200, 20)
    plt.xlim(0, 0.5)
    plt.legend(loc='upper right')
    plt.xlabel('Normalized Frequency (f/f_s)')
    plt.ylabel('Magnitude (dB)')
    plt.title(f'{config.ntaps}-tap FIR')
    axz.axhline(0, color='gray')
    p = f < config.f_pass
    for l in ax.get_lines()[:-2]:  # skip the axhline
        axz.plot(f[p], l.get_ydata()[p], color=l.get_color(), linestyle=l.get_linestyle())
    pb = [dB(x[p]) for x in (fir_f, H_f, H_f*fir_f)]
    axz.set_xlim(0, config.f_pass)
    axz.set_ylim(min(map(np.min, pb)) - 0.1, max(map(np.max, pb)) + 0.1)
    axz.set_title('passband')
    plt.savefig('plot.png', dpi=100)
    plt.close()

if __name__ == "__main__":
    main()
