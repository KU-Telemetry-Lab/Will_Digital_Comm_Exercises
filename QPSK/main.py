import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications


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

sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0

bits = [3, 2, 0, 1]
bits_str = ['11', '10', '00', '01']
amplitudes = [complex( 1+ 1j), complex( 1+-1j), complex(-1+-1j), complex(-1+ 1j)]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

header = [1, 1, 1, 1]
test_input_1 = [1, 0, 0, 1, 0, 0]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "will is cool, this is a test"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+2] for i in range(0, len(string_input_bin), 2)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = header + test_input_3
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
    np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_real, carrier_frequency, sample_rate))) +
    np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_imag, carrier_frequency, sample_rate)))
)

# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT_real = np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))
r_nT_imag = np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs_real = np.array(DSP.downsample(x_nT_real, sample_rate))
x_kTs_imag = np.array(DSP.downsample(x_nT_imag, sample_rate))
x_kTs = x_kTs_real + 1j * x_kTs_imag

# 2.5 MAKE A DECISION FOR EACH PULSE
qpsk_constellation = [[complex( 1+ 1j), 3], [complex( 1+-1j), 2], [complex(-1+-1j), 0], [complex(-1+ 1j), 1]]
detected_ints = communications.nearest_neighbor(x_kTs[len(header):], qpsk_constellation)
print(f"Transmission Symbol Errors: {error_count(b_k[len(header):], detected_ints)}")

# 2.6 CONVERT BINARY TO ASCII
detected_bits = []
for symbol in detected_ints:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)

# # Plot original symbols
# plt.figure()
# plt.stem(np.imag(a_k))
# plt.title("Original Symbols")

# # Plot upsampled symbols
# plt.figure()
# plt.stem(np.real(a_k_upsampled))
# plt.title("Upsampled Symbols")

# # Plot modulated signal
# plt.figure()
# plt.stem(s_nT_modulated)
# plt.title("Modulated Signal")

# # Plot demodulated signal
# plt.figure()
# plt.stem(r_nT)
# plt.title("Demodulated Signal")

# # Plot match filtered signal
# plt.figure()
# plt.stem(x_nT)
# plt.title("Match Filtered Signal")

# # Plot downsampled signal
# plt.figure()
# plt.stem(np.imag(x_kTs))
# plt.title("Downsampled Signal")
# plt.show()


