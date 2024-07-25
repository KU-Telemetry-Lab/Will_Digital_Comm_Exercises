import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications


# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.125*sample_rate
bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]

bits = [i[1] for i in bpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in bpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "bpskdata.mat"
modulated_data = MatLab.load_matlab_file(test_file)[1]
input_message_length = 805 # symbols (115 ascii characters)
data_offset = 16
pulse_shape = communications.srrc(.5, sample_rate, 32)
bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]

r_nT = np.real(np.array(np.sqrt(2) * DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate)))
x_nT = np.real(np.array(np.roll(np.real(DSP.convolve(r_nT, pulse_shape, mode="full")), -1)))
x_kTs = DSP.downsample(x_nT, sample_rate, offset=sample_rate)
detected_ints = communications.nearest_neighbor(x_kTs, bpsk_constellation)

message = communications.bin_to_char(detected_ints[data_offset:])
print(message)