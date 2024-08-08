import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications, SCS

def clock_offset(signal, sample_rate, offset_fraction):
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction
    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset 
    x_shifted = interpolator(t_shifted)
    return x_shifted

def plot_complex_points(complex_array, constellation):
    plt.plot([point.real for point in complex_array], [point.imag for point in complex_array], 'ro', label='Received Points')
    for point, label in constellation:
        plt.plot(point.real, point.imag, 'b+', markersize=10)
        plt.text(point.real, point.imag, f' {label}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Complex Constellation Plot')
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

# SYSTEM PARAMETERS
###################################################################################################
qpsk_constellation = [[complex( np.sqrt(1) +  np.sqrt(1)*1j), 3], 
                      [complex( np.sqrt(1) + -np.sqrt(1)*1j), 2], 
                      [complex(-np.sqrt(1) + -np.sqrt(1)*1j), 0], 
                      [complex(-np.sqrt(1) +  np.sqrt(1)*1j), 1]]
fs = 8 # sample rate
input_message_symbols = [0, 1, 2, 3, 2, 1, 0]

bits_to_amplitude = {bit: amplitude for amplitude, bit in qpsk_constellation}

# inphase channel symbol mapping
xk = np.real([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# quadrature channel symbol mapping
yk = np.imag([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# adding header to each channel
header = np.ones(25)
xk = np.concatenate([header, xk])
yk = np.concatenate([header, yk])


# UPSAMPLING
# ###################################################################################################
xk_upsampled = DSP.upsample(xk, fs, interpolate_flag=False)
yk_upsampled = DSP.upsample(yk, fs, interpolate_flag=False)

# plot_complex_points((xk_upsampled + 1j * yk_upsampled), constellation=qpsk_constellation)

# INTRODUCE TIMING OFFSET
###################################################################################################
timing_offset = 0.1
sample_shift = int(fs/3)
print(sample_shift)

xk_upsampled_offset = clock_offset(xk_upsampled, fs, timing_offset)[sample_shift:]
yk_upsampled_offset = clock_offset(yk_upsampled, fs, timing_offset)[sample_shift:]

plot_complex_points((xk_upsampled_offset + 1j * yk_upsampled_offset), constellation=qpsk_constellation)

# PULSE SHAPE
###################################################################################################
length = 64
alpha = 0.10
pulse_shape = communications.srrc(alpha, fs, length)

yk_pulse_shaped = np.real(np.roll(DSP.convolve(yk_upsampled, pulse_shape, mode="same"), -1))
yk_pulse_shaped_offset = np.real(np.roll(DSP.convolve(yk_upsampled_offset, pulse_shape, mode="same"), -1))

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].stem(yk_pulse_shaped[len(header)*fs:])
axs[0].set_title("Pulse Shaped")
axs[1].stem(yk_pulse_shaped_offset[len(header)*fs:])
axs[1].set_title("Pulse Shaped w/ Offset")

plt.tight_layout()
plt.show()