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
carrier_frequency = 0.3*sample_rate
symbol_clock_offset = 0
qpsk_constellation = [[complex( np.sqrt(1)+ np.sqrt(1)*1j), 3], 
                      [complex( np.sqrt(1)+-np.sqrt(1)*1j), 2], 
                      [complex(-np.sqrt(1)+-np.sqrt(1)*1j), 0], 
                      [complex(-np.sqrt(1)+ np.sqrt(1)*1j), 1]]

unique_word_symbols = [2, 2, 2, 2, 1, 1, 1, 1]

phase_ambiguities = {
    "22221111": 0,
    "33330000": np.pi/2,
    "11112222": np.pi,
    "00003333": 3*np.pi/2
}

header = [3, 3, 3, 3]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

# TEST ON GIVEN ASACII DATA
test_file = "qpskcruwdata.mat"
data_offset = 16
input_message_length = 2247
modulated_data = header + list(MatLab.load_matlab_file(test_file)[1])
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)

# demodulate
r_nT_real = np.sqrt(2) * np.real(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))
r_nT_imag = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))

# match filter
x_nT = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1)) + 1j * np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))

# down sample (and remove header)
x_kTs = (np.array(DSP.downsample(x_nT, sample_rate, offset=0)))[len(header):]

B = 0.02 * sample_rate
zeta = 1 / np.sqrt(2)
K0 = 1
Kp = 1
K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / sample_rate)) / K0)
K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / sample_rate)) ** 2) / K0)
pll = PLL.PLL(Kp, K0, K1, K2, carrier_frequency, sample_rate)

dds_output = np.exp(1j * 0)
rotated_constellations = []
detected_constellations = []
pll_error = []

for i in range(len(x_kTs)):
    # perform ccw rotation
    x_kTs_ccwr = x_kTs[i] * dds_output
    rotated_constellations.append(x_kTs_ccwr)
    
    # find nearest neighbor constellation
    detected_int = communications.nearest_neighbor([x_kTs_ccwr], qpsk_constellation)[0]
    detected_constellation = bits_to_amplitude[detected_int]
    detected_constellations.append(detected_constellation)

    # calculate phase error
    phase_error = pll.phase_detector(x_kTs_ccwr, detected_constellation)
    pll_error.append(phase_error)

    # feed into loop filter
    loop_filter_output = pll.loop_filter(phase_error)
    
    # generate next dds output
    dds_output = np.exp(1j * loop_filter_output)

# plt.plot(np.real(x_kTs), np.imag(x_kTs), 'ro')
# plt.plot(np.real(rotated_constellations), np.imag(rotated_constellations), 'ro')
# plt.plot(np.real(detected_constellations), np.imag(detected_constellations), 'bo')
# plt.show()

# unique word resolution
received_unique_word = communications.nearest_neighbor(detected_constellations[: len(unique_word_symbols)], qpsk_constellation)
received_unique_word_string = ""
for symbol in received_unique_word:
    received_unique_word_string += str(symbol)

print(received_unique_word_string)


# uw_start_index = find_subarray_index(unique_word, detected_bits)
# if uw_start_index == -1:
#     uw_start_index = find_subarray_index(unique_word[::-1], detected_bits)
    

# # # detected_bits = detected_bits[uw_start_index + len(unique_word): uw_start_index + len(unique_word) + input_message_length]
# # # message = communications.bin_to_char(detected_bits)
# # # print(message)