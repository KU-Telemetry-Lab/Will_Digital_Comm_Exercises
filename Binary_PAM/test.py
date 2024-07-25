import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications


# SYSTEM PARAMETERS
sample_rate = 16
pulse_shape = "NRZ"
symbol_clock_offset = 0

pam_constellation = [
    [complex(-1), 0],
    [complex(1), 1]
]
amplitude_to_bits = {-1: 0, 1: 1} # binary PAM symbol to bits map
bits_to_amplitude = {value: key for key, value in amplitude_to_bits.items()} # binary PAM bits to symbol map


# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "bb2data.mat" 
input_message_length = 126 
r_nT = MatLab.load_matlab_file(test_file)[1]

filter_num = [.25*1 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_denom))

x_nT = DSP.direct_form_2(filter_num, filter_denom, r_nT)
x_kTs = DSP.downsample(x_nT[sample_rate:], sample_rate)
detected_ints = communications.nearest_neighbor(x_kTs, pam_constellation)

char_message = communications.bin_to_char(detected_ints)
print(char_message)
