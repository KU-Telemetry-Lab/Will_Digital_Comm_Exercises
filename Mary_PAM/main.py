import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications as Communications


def int_to_three_bit_binary(num):
    binary_str = bin(num)[2:]  # Convert integer to binary string
    padded_binary_str = binary_str.zfill(3)  # Pad with zeros to ensure three place values
    return padded_binary_str

def string_to_ascii_binary(string, num_bits=7):
    ascii_binary_strings = []
    for char in string:
        ascii_binary = bin(ord(char))[2:].zfill(num_bits)
        ascii_binary_strings.append(ascii_binary)
    return ascii_binary_strings


sample_rate = 16 # samples/bit
pulse_shape = "NRZ" # non return to zero pulse shaping
symbol_clock_offset = 0

amplitudes = [-7, -5, -3, -1, 1, 3, 5, 7]
bits = [0, 1, 3, 2, 5, 7, 6, 4]
bin_strs = ['000', '001', '011', '010', '101', '111', '110', '100']
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bin_str = dict(zip(bits, bin_strs))
mary_pam = [
    [complex(-7+0j), 0], 
    [complex(-5+0j), 1], 
    [complex(-3+0j), 3], 
    [complex(-1+0j), 2], 
    [complex(1+0j), 5], 
    [complex(3+0j), 7], 
    [complex(5+0j), 6], 
    [complex(7+0j), 4]]


test_input_1 = [0, 1]
test_input_2 = [1, 3, 5, 7, 6, 4, 2, 0]
string_input = "will is cool"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+3] for i in range(0, len(string_input_bin), 3)]
test_input_3 = [int(bin3, 2) for bin3 in input_bin_blocks]


# 1.1 UPSAMPLE THE BASEBAND BINARY SIGNAL
b_k = test_input_3
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.Upsample(a_k, sample_rate, interpolate=False)

# 1.2 NRZ FILTER THE UPSAMPLED SIGNAL (PULSE SHAPING)
filter_num = [.25 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_num))
s_nT = DSP.DirectForm2(filter_num, filter_denom, a_k_upsampled)

# 2.1 MATCH FILTER THE RECIEVED SIGNAL
x_nT = DSP.DirectForm2(filter_num, filter_denom, s_nT) 

# 2.2 DOWNSAMPLE EACH PULSE
x_kTs = DSP.Downsample(x_nT[sample_rate:], sample_rate)

# 2.3 MAKE A DECISION FOR EACH PULSE (MODULAR)
detected_ints = Communications.nearest_neighbor(x_kTs, mary_pam)

detected_bits = []
for num in detected_ints:
    bin3 = list(str(bin(num))[2:])
    if len(bin3) == 1:
        bin3 = ''.join(['0', '0'] + list(bin3))
    if len(bin3) == 2:
        bin3 = ''.join(['0'] + list(bin3))
    detected_bits += bin3

char_message = Communications.bin_to_char(detected_bits)
print(char_message)

# 2.4 PlOTTING SIMULATION RESULTS
plt.figure()
plt.stem(a_k)
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(a_k_upsampled)
plt.title('a_k_upsampled')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(s_nT)
plt.title('s_nT')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(x_nT)
plt.title('x_nT')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(x_kTs)
plt.title('x_kTs')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(detected_bits)
plt.title('x_nT')
plt.xlabel('Index')
plt.ylabel('Value')

plt.show()


# 3 TEST SYSTEM ON GIVEN ASCII DATA
test_file = "bb8data.mat"
input_message_length = 140 # symbols = 420 bits = 60 ascii chars
r_nT = MatLab.loadMatLabFile(test_file)[1]

filter_num = [.25/3 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_denom))
x_nT = DSP.DirectForm2(np.array(filter_num), np.array(filter_denom), r_nT)

x_kTs = DSP.Downsample(x_nT[sample_rate:], sample_rate)

detected_ints = Communications.nearest_neighbor(x_kTs, mary_pam)

char_message = Communications.bin_to_ascii(detected_bits)
print(char_message)