import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import Communications

A = 1  # incoming signal amplitude
amplitude_to_bits = {-A: 0, A: 1} # binary PAM symbol to bits map
bits_to_amplitude = {value: key for key, value in amplitude_to_bits.items()} # binary PAM bits to symbol map

sample_rate = 16 # samples/bit
pulse_shape = "NRZ" # non return to zero pulse shaping
symbol_clock_offset = 0

# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "bb2data.mat" # binay input file with N messages
input_message_length = 126 # bits (18 ASCII characters)
r_nT = MatLab.loadMatLabFile(test_file)[1]

filter_num = [.25*1 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_denom))
x_nT = DSP.DirectForm2(filter_num, filter_denom, r_nT)

x_kTs = DSP.Downsample(x_nT[sample_rate:], sample_rate)

detected_bits = [amplitude_to_bits[round(symbol)] for symbol in x_kTs]

char_message = Communications.bin_to_ascii(detected_bits)

print(char_message)
