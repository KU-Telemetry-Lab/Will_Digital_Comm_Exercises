import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
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

def error_count(x, y):
    count = 0
    for i in range(len(x)):
        if (x[i] != y[i]):
            count += 1
    return count

def invert_array(x):
    return np.array([0 if value == 1 else 1 for value in x])


# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0
qpsk_constellation = [[complex( np.sqrt(1)+ np.sqrt(1)*1j), 3], 
                      [complex( np.sqrt(1)+-np.sqrt(1)*1j), 2], 
                      [complex(-np.sqrt(1)+-np.sqrt(1)*1j), 0], 
                      [complex(-np.sqrt(1)+ np.sqrt(1)*1j), 1]]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

test_input_1 = [1, 0, 0, 1]
test_input_2 = [3, 2, 1, 0, 1, 2, 3, 3, 2, 1, 0, 1, 2]
string_input = "\nthis is a decision directed carrier phase synchronization test using 2 bit quadrture amplitude modulation!\n"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+2] for i in range(0, len(string_input_bin), 2)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]


# SYNCHRONIZATION PARAMETERS
header = np.tile([3, 0], 50).tolist()
unique_word = [0, 1, 2, 3, 0, 1, 2, 3]
phase_ambiguities = {
    "01230123": 0,
    "20312031": np.pi/2,
    "32103210": np.pi,
    "13021302": 3*np.pi/2
}

transmitter_phase_offset = np.pi
transmitter_freq_offset = carrier_frequency * 0.0001


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = header + unique_word + test_input_3
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate_flag=False)
a_k_upsampled_real = np.real(a_k_upsampled)
a_k_upsampled_imag = np.imag(a_k_upsampled)

# 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)
s_nT_real = np.real(np.roll(DSP.convolve(a_k_upsampled_real, pulse_shape, mode="same"), -1))
s_nT_imag = np.real(np.roll(DSP.convolve(a_k_upsampled_imag, pulse_shape, mode="same"), -1))

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR (ADD PHASE OFFSET)
s_nT_modulated = (
    np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_real, carrier_frequency + transmitter_freq_offset, sample_rate))) +
    np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_imag, carrier_frequency + transmitter_freq_offset, sample_rate)))
) * np.exp(1j * transmitter_phase_offset)


# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT_real = np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))
r_nT_imag = np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs_real = np.array(DSP.downsample(x_nT_real, sample_rate))
x_kTs_imag = np.array(DSP.downsample(x_nT_imag, sample_rate))
x_kTs = (x_kTs_real + 1j * x_kTs_imag)


# 3.1 PLL SYSTEM PARAMETERS
B = 0.02 * sample_rate
zeta = 1 / np.sqrt(2)
pll = PLL.PLL(sample_rate, loop_bandwidth=B, damping_factor=zeta)

dds_output = np.exp(1j * 0)
rotated_constellations = []
detected_constellations = []
pll_error = []
loop_filter_outputs = []
dds_outputs = []

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
    loop_filter_outputs.append(loop_filter_output)

    # generate next dds output
    dds_output = np.exp(1j * loop_filter_output)
    dds_outputs.append(dds_output)

plt.plot(pll_error, label="Phase Error")
# plt.plot(np.real(dds_outputs), label="DDS Ouptut")
plt.legend()
plt.show()

detected_ints = communications.nearest_neighbor(detected_constellations, qpsk_constellation)

# 3.2 UNIQUE WORD RESOLUTION
unique_word_found = None
unique_word_index = -1

for unique_word in phase_ambiguities.keys():
    unique_word_list = [int(i) for i in unique_word]
    unique_word_index = find_subarray_index(unique_word_list, detected_ints)
    if unique_word_index != -1:
        unique_word_found = unique_word
        break

# 3.3 UNIQUE WORD OFFSET ADJUSTMENT
detected_constellations = np.array(detected_constellations[unique_word_index:]) * np.exp(-1j * phase_ambiguities[unique_word_found])

# 4.1 SYMBOL DETECTION AND ERROR CALCULATION
detected_ints = communications.nearest_neighbor(detected_constellations[len(unique_word):], qpsk_constellation)
print(f"Transmission Symbol Errors: {error_count(b_k[len(header) + len(unique_word):], detected_ints)}")

# 4.2 CONSTELLATION PLOTTING
plt.plot(np.real(rotated_constellations), np.imag(rotated_constellations), 'ro')
plt.plot(np.real(detected_constellations), np.imag(detected_constellations), 'bo')
plt.show()

# 4.3 MESSAGE DECODING (BIN TO ASCII)
detected_bits = []
for symbol in detected_ints:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)
