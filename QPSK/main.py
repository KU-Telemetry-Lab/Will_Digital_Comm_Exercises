import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import MatLab
from KUSignalLib import communications


def local_oscillator_cos(input, f_c, f_s):
    output = []
    for i in range(len(input)):
        output.append(input[i] * np.sqrt(2) * np.cos(2*np.pi*f_c*i/f_s))
    return output

def local_oscillator_sin(input, f_c, f_s):
    output = []
    for i in range(len(input)):
        output.append(input[i] * -1 * np.sqrt(2) * np.sin(2*np.pi*f_c*i/f_s))
    return output

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
pulse_shape = "NRZ" # change to SRRC (50% excess bandwidth)
symbol_clock_offset = 0

bits = [3, 2, 0, 1]
bits_str = ['11', '10', '00', '01']
amplitudes = [complex( 1+ 1j), complex( 1+-1j), complex(-1+-1j), complex(-1+ 1j)]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


test_input_1 = [0, 1, 2, 3]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "will is cool"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+2] for i in range(0, len(string_input_bin), 2)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_1
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.Upsample(a_k, sample_rate, interpolate=False)

# 1.2 NRZ FILTER THE UPSAMPLED SIGNAL (PULSE SHAPING)
filter_num = [.25/2 for i in range(sample_rate)]
filter_denom = [1]
s_nT = np.array(DSP.DirectForm2(filter_num, filter_denom, np.real(a_k_upsampled)) + 1j*DSP.DirectForm2(filter_num, filter_denom, np.imag(a_k_upsampled)))

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR 
carrier_frequency = 2
s_nT_modulated = np.array(local_oscillator_cos(np.real(s_nT), carrier_frequency, sample_rate) + local_oscillator_sin(np.imag(s_nT), carrier_frequency, sample_rate))

# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT = np.array(local_oscillator_cos(s_nT_modulated, carrier_frequency, sample_rate) + 1j*np.array(local_oscillator_sin(s_nT_modulated, carrier_frequency, sample_rate)))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT = np.array(DSP.DirectForm2(filter_num, filter_denom, np.real(r_nT)) + 1j*DSP.DirectForm2(filter_num, filter_denom, np.imag(r_nT)))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs = DSP.Downsample(x_nT[sample_rate:], sample_rate)
print(np.real(a_k))
print(np.imag(a_k))
print(np.real(x_kTs))
print(np.imag(x_kTs))

# 2.5 MAKE A DECISION FOR EACH PULSE
qpsk = [[complex( 1+ 1j), 3], [complex( 1+-1j), 2], [complex(-1+-1j), 0], [complex(-1+ 1j), 1]]
detected_bits = communications.nearest_neighbor(x_kTs, qpsk)
print(f"Transmission Bit Errors: {error_count(b_k, detected_bits)}")
print(detected_bits)

