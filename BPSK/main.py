import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
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
bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]

bits = [i[1] for i in bpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in bpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

test_input_1 = [1, 0, 0, 1]
test_input_2 = [1, 1, 0, 0, 1, 1, 0, 0]
string_input = "...will is cool, this is a test..."
test_input_3 = [int(num) for num in ''.join(string_to_ascii_binary(string_input))]

# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_3
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate_flag=False)

# 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)
s_nT = np.real(np.array(np.roll(np.real(DSP.convolve(a_k_upsampled, pulse_shape, mode="same")), -1)))

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR 
s_nT_modulated = np.real(np.array(np.sqrt(2) * DSP.modulate_by_exponential(s_nT, carrier_frequency, sample_rate)))


# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT = np.real(np.array(np.sqrt(2) * DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT = np.real(np.array(np.roll(np.real(DSP.convolve(r_nT, pulse_shape, mode="same")), -1)))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs = DSP.downsample(x_nT, sample_rate)

# 2.4 MAKE A DECISON FOR EACH PULSE
detected_bits = communications.nearest_neighbor(x_kTs, bpsk_constellation)
print(f"Transmission Bit Errors: {error_count(b_k, detected_bits)}")

# 2.3 CONVERT BINARY TO ASCII
message = communications.bin_to_char(detected_bits)
print(message)

