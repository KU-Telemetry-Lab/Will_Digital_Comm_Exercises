import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications

def error_count(x, y):
    count = 0
    for i in range(len(x)):
        if (x[i] != y[i]):
            count += 1
    return count

# SYSTEM PARAMETERS
sample_rate = 16

amplitude_to_bits = {-1: 0, 1: 1}
bits_to_amplitude = {value: key for key, value in amplitude_to_bits.items()} 

test_input_1 = [1]
test_input_2 = [1, 0, 0, 1]
test_input_3 = [1, 1, 1, 1, 0, 0, 0, 0]


# 1.1 UPSAMPLE THE BASEBAND BINARY SIGNAL
b_k = test_input_3
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate_flag=False)

# 1.2 NRZ FILTER THE UPSDAMPLED SIGNAL
filter_num = [.25*1 for i in range(sample_rate)]
filter_denom = [1]
padding_length = max(len(filter_num), len(filter_denom))
s_nT = DSP.direct_form_2(filter_num, filter_denom, a_k_upsampled)


# 2.1 MATCH FILTER THE RECIEVED SIGNAL
x_nT = DSP.direct_form_2(filter_num, filter_denom, s_nT) 

# 2.2 DOWNSAMPLE EACH PULSE
x_kTs = DSP.downsample(x_nT[sample_rate+1:], sample_rate)

# 2.3 MAKE A DECISION FOR EACH PULSE
detected_ints = communications.nearest_neighbor(x_kTs, pam_constellation)
print(f"Transmission Symbol Errors: {error_count(b_k, detected_ints)}")


# 3.1 PlOTTING SIMULATION RESULTS
plt.figure()
plt.stem(np.real(a_k))
plt.title('a_k')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(np.real(a_k_upsampled))
plt.title('a_k_upsampled')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(np.real(s_nT))
plt.title('s_nT')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(np.real(x_nT))
plt.title('x_nT')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(np.real(x_kTs))
plt.title('x_kTs')
plt.xlabel('Index')
plt.ylabel('Value')

plt.figure()
plt.stem(detected_ints)
plt.title('x_nT')
plt.xlabel('Index')
plt.ylabel('Value')

# plt.show()


