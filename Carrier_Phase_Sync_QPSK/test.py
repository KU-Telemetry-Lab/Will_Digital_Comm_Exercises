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

header = np.tile([3, 0], 50).tolist()

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

# down sample
x_kTs = (np.array(DSP.downsample(x_nT, sample_rate, offset=0)))

B = 0.02 * sample_rate
zeta = 1 / np.sqrt(2)
pll = PLL.PLL(sample_rate, loop_bandwidth=B, damping_factor=zeta)

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

plt.plot(pll_error, label='Phase error')
# plt.plot(np.real(internalSignal), label='internal Signal')
# plt.plot(np.real(incomingSignal), label='incoming Signal')
plt.legend()
plt.show()

# DSP.plot_complex_points(detected_constellations, referencePoints=amplitudes) 

# detected_ints = communications.nearest_neighbor(detected_constellations, qpsk_constellation)

# unique_word_found = None
# unique_word_index = -1

# for unique_word in phase_ambiguities.keys():
#     unique_word_list = [int(i) for i in unique_word]
#     unique_word_index = find_subarray_index(unique_word_list, detected_ints)
#     if unique_word_index != -1:
#         unique_word_found = unique_word
#         break

# detected_constellations = np.array(detected_constellations[unique_word_index:]) * np.exp(-1j * phase_ambiguities[unique_word_found])
# detected_ints = communications.nearest_neighbor(detected_constellations[len(unique_word):], qpsk_constellation)
# print(f"Transmission Symbol Errors: {error_count(b_k[len(header) + len(unique_word):], detected_ints)}")

# # 4.2 CONSTELLATION PLOTTING
# plt.plot(np.real(rotated_constellations), np.imag(rotated_constellations), 'ro')
# plt.plot(np.real(detected_constellations), np.imag(detected_constellations), 'bo')
# plt.show()

# # 4.3 MESSAGE DECODING (BIN TO ASCII)
# detected_bits = []
# for symbol in detected_ints:
#     detected_bits += ([*bin(symbol)[2:].zfill(2)])

# message = communications.bin_to_char(detected_bits)
# print(message)