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

test_input_1 = [1]
test_input_2 = [1, 0, 0, 1]
test_input_3 = [1, 1, 1, 1, 0, 0, 0, 0]

# 1.1 UPSAMPLE THE BASEBAND BINARY SIGNAL
b_k = test_input_2 # input binary signal
a_k = [bits_to_amplitude[bit] for bit in b_k] # symbol mapping
a_k_upsampled = DSP.Upsample(a_k, sample_rate, interpolate=False) # upsampling

# 1.2 NRZ FILTER THE UPSDAMPLED SIGNAL
filter_num = [.25*1 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_denom))
s_nT = DSP.DirectForm2(filter_num, filter_denom, a_k_upsampled)

# 2.1 MATCH FILTER THE RECIEVED SIGNAL
x_nT = DSP.DirectForm2(filter_num, filter_denom, s_nT) 

# 2.2 DOWNSAMPLE EACH PULSE
x_kTs = DSP.Downsample(x_nT[sample_rate+1:], sample_rate)

# 2.3 MAKE A DECISION FOR EACH PULSE
detected_bits = [amplitude_to_bits[round(symbol)] for symbol in x_kTs]

# 2.4 PlOTTING SIMULATION RESULTS
plt.figure()
plt.stem(a_k)
plt.title('a_k')
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

# plt.show()


