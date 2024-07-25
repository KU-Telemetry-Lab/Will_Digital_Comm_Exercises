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


# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0

mpsk_constellation = [
    [complex(3 + 0*1j), 0],
    [complex(np.sqrt(4.5) + np.sqrt(4.5)*1j), 1],
    [complex(0 + 3*1j), 3],
    [complex(-np.sqrt(4.5) + np.sqrt(4.5)*1j), 2],
    [complex(-3 + 0*1j), 6],
    [complex(-np.sqrt(4.5) + -np.sqrt(4.5)*1j), 7],
    [complex(0 + -3*1j), 5],
    [complex(np.sqrt(4.5) + -np.sqrt(4.5)*1j), 4]
]

bits = [i[1] for i in mpsk_constellation]
bits_str = ['000', '001', '011', '010', '110', '111', '101', '100']
amplitudes = [i[0] for i in mpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

header = [1, 1, 1, 1]
test_input_1 = [1, 0, 0, 1, 0, 0]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "will is cool, this is a test"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+3] for i in range(0, len(string_input_bin), 3)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = header + test_input_3
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
detected_ints = communications.nearest_neighbor(x_kTs[len(header):], mpsk_constellation)
print(f"Transmission Symbol Errors: {error_count(b_k[len(header):], detected_ints)}")

# # 2.6 CONVERT BINARY TO ASCII
detected_bits = []
for symbol in detected_ints:
    detected_bits += list(bits_to_bits_str[symbol])
message = communications.bin_to_char(detected_bits)
print(message)