import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import MatLab
from KUSignalLib import communications

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
carrier_frequency = sample_rate * 0.2
pulse_shape = "SRRC"
unique_word = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]
header = [1, 1, 1, 1]

A = 1
amplitudes = [-1, 1]
bits = [0, 1]
bin_strs = ['0', '1']
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bin_str = dict(zip(bits, bin_strs))

# TEST ON GIVEN ASACII DATA
sample_rate
test_file = "bpskcruwdata.mat"
data_offset = 16
input_message_length = 2247
modulated_data = header + list(MatLab.load_matlab_file(test_file)[1])
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)

r_nT_real = np.sqrt(2) * np.real(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))
r_nT_imag = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))
x_kTs_real = np.array(DSP.downsample(x_nT_real, sample_rate, offset=0))
x_kTs_imag = np.array(DSP.downsample(x_nT_imag, sample_rate, offset=0))

B = 0.02 * sample_rate
fs = sample_rate
fc = carrier_frequency
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
    x_kTs_real_ccwr = DSP.modulate_by_exponential([x_kTs_real[sample_index]], np.real(dds_output), sample_rate)
    x_kTs_imag_ccwr = DSP.modulate_by_exponential([x_kTs_imag[sample_index]], np.imag(dds_output), sample_rate)

    # decision
    bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]
    detected_bit_real = communications.nearest_neighbor(x_kTs_real_ccwr, bpsk_constellation)[0]
    detcted_bit_imag = communications.nearest_neighbor(x_kTs_imag_ccwr, bpsk_constellation)[0]
    detected_bits.append(detected_bit_real)

    # input to pll
    phase_detector_output = pll.phase_detector(x_kTs_real_ccwr + 1j*x_kTs_imag_ccwr, detected_bit_real + 1j*detcted_bit_imag)
    loop_filter_output = pll.loop_filter(phase_detector_output)
    dds_output = pll.DDS(sample_index, loop_filter_output[0])

uw_start_index = find_subarray_index(unique_word, detected_bits)
if uw_start_index == -1:
    uw_start_index = find_subarray_index(unique_word[::-1], detected_bits)
    

# detected_bits = detected_bits[uw_start_index + len(unique_word): uw_start_index + len(unique_word) + input_message_length]
# message = communications.bin_to_char(detected_bits)
# print(message)