import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import MatLab
from KUSignalLib import communications


def modulate_by_exponential(x, f_c, f_s, phase_offset=0):
    y = []
    for i, value in enumerate(x):
        modulation_factor = np.exp(1j * 2 * np.pi * f_c * i / f_s + phase_offset)
        y.append(value * modulation_factor)
    return np.array(y)

def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    large_len = len(large_array)
    for i in range(large_len - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1

def string_to_ascii_binary(string, num_bits=7):
    ascii_binary_strings = []
    for char in string:
        ascii_binary = bin(ord(char))[2:].zfill(num_bits)
        ascii_binary_strings.append(ascii_binary)
    return ascii_binary_strings


sample_rate = 8
pulse_shape = "SRRC"
unique_word = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
header = [1, 1, 1, 1]

# transmit and receive local oscillators slightly off sync
receive_lo_frequnecy = sample_rate * 0.2
receive_lo_phase = 0
transmit_lo_frequnecy = .90 * receive_lo_frequnecy 
transmit_lo_phase = np.pi

bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]

bits = [i[1] for i in bpsk_constellation]
bits_str = ['0', '1']
amplitudes = [i[0] for i in bpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


test_input_1 = [1, 0, 0, 1]
test_input_2 = [1, 1, 0, 0, 1, 1, 0, 0]
string_input = "will is cool, this is a test"
test_input_3 = [int(num) for num in ''.join(string_to_ascii_binary(string_input))]

# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = header + unique_word + test_input_1
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate=False)
a_k_upsampled_real = np.real(a_k_upsampled)
a_k_upsampled_imag = np.imag(a_k_upsampled)

# 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)
s_nT_real = np.real(np.roll(DSP.convolve(a_k_upsampled_real, pulse_shape, mode="same"), -1))
s_nT_imag = np.real(np.roll(DSP.convolve(a_k_upsampled_imag, pulse_shape, mode="same"), -1))

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR
s_nT_modulated = (
    np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_real, transmit_lo_frequnecy, sample_rate, phase=transmit_lo_phase)) +
    np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_imag, transmit_lo_frequnecy, sample_rate, phase=transmit_lo_phase))
)

# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT_real = np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_modulated, receive_lo_frequnecy, sample_rate, phase=receive_lo_phase))
r_nT_imag = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_modulated, receive_lo_frequnecy, sample_rate, phase=receive_lo_phase))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs_real = np.array(DSP.downsample(x_nT_real, sample_rate))
x_kTs_imag = np.array(DSP.downsample(x_nT_imag, sample_rate))

B = 0.02 * sample_rate
fs = sample_rate
fc = receive_lo_frequnecy
zeta = 1 / np.sqrt(2)
K0 = 1
Kp = 1
K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / fs)) / K0)
K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / fs)) ** 2) / K0)
pll = PLL.PLL(Kp, K0, K1, K2, fc, fs)

detected_bits = []
dds_output = 0 + 1j*0
sample_indexes = np.arange(len(x_kTs_real))

for sample_index in sample_indexes:
    # ccw rotation via mixing
    x_kTs_real_ccwr = modulate_by_exponential([x_kTs_real[sample_index]], 0, sample_rate, np.real(dds_output))
    x_kTs_imag_ccwr = modulate_by_exponential([x_kTs_imag[sample_index]], 0, sample_rate, np.imag(dds_output))

    # decision
    detected_bit_real = communications.nearest_neighbor(x_kTs_real_ccwr, bpsk_constellation)[0]
    detcted_bit_imag = communications.nearest_neighbor(x_kTs_imag_ccwr, bpsk_constellation)[0]
    detected_bits.append(detected_bit_real)

    # input to pll
    phase_detector_output = pll.phase_detector(x_kTs_real_ccwr + 1j*x_kTs_imag_ccwr, detected_bit_real + 1j*detcted_bit_imag)
    loop_filter_output = pll.loop_filter(phase_detector_output)
    dds_output = pll.DDS(sample_index, loop_filter_output[0])

print(detected_bits)

# uw_start_index = find_subarray_index(unique_word, detected_bits)
# detected_bits = detected_bits[uw_start_index + len(unique_word): uw_start_index + len(unique_word) + len(b_k)]
# message = communications.bin_to_char(detected_bits)
# print(message)