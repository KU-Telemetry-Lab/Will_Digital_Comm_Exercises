import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications, SCS


def apply_clock_offset(signal, sample_rate, offset_fraction):
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction
    print(1/sample_rate)
    print(clock_offset)

    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset
    x_shifted = interpolator(t_shifted)
    return x_shifted


# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25 * sample_rate
symbol_clock_offset = .1
qpsk_constellation = [[complex(np.sqrt(1) + np.sqrt(1) * 1j), 3],
                      [complex(np.sqrt(1) + -np.sqrt(1) * 1j), 2],
                      [complex(-np.sqrt(1) + -np.sqrt(1) * 1j), 0],
                      [complex(-np.sqrt(1) + np.sqrt(1) * 1j), 1]]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))
test_input_1 = [1, 0, 1, 0]


# SYNCHRONIZATION PARAMETERS
timing_offset = 0.10 # fractional offset (0-1)

# UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_1
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate_flag=False)
a_k_upsampled_real = np.real(a_k_upsampled)
a_k_upsampled_imag = np.imag(a_k_upsampled)

# TIMING OFFSET
a_k_upsampled_real_offset = apply_clock_offset(a_k_upsampled_real, sample_rate, timing_offset)
a_k_upsampled_imag_offset = apply_clock_offset(a_k_upsampled_imag, sample_rate, timing_offset)


# PULSE SHAPE
length = 64
alpha = 0.5
pulse_shape = communications.srrc(alpha, sample_rate, length)

s_nT_real = np.real(np.roll(DSP.convolve(a_k_upsampled_real, pulse_shape, mode="same"), -1))
s_nT_imag = np.real(np.roll(DSP.convolve(a_k_upsampled_imag, pulse_shape, mode="same"), -1))

s_nT_real_offset = np.real(np.roll(DSP.convolve(a_k_upsampled_real_offset, pulse_shape, mode="same"), -1))
s_nT_imag_offset = np.real(np.roll(DSP.convolve(a_k_upsampled_imag_offset, pulse_shape, mode="same"), -1))


# PLOTTING
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

axs[0].stem(s_nT_imag)
axs[0].set_title("Pulse Shaped")

axs[1].stem(s_nT_imag_offset)
axs[1].set_title("Pulse Shaped w/ Offset")

plt.tight_layout()
plt.show()