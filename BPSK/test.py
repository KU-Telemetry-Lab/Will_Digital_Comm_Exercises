import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications

sample_rate = 8
carrier_frequnecy = .125*8
symbol_clock_offset = 0

A = 1 # incoming signal base amplitude
amplitudes = [-1, 1]
bits = [0, 1]
bin_strs = ['0', '1']
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bin_str = dict(zip(bits, bin_strs))

# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "bpskdata.mat"
modulated_data = MatLab.load_matlab_file(test_file)[1]
input_message_length = 805 # symbols (115 ascii characters)
data_offset = 16
carrier_frequency = 1
sample_rate = 8
pulse_shape = communications.srrc(.5, sample_rate, 32)
bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]

r_nT = np.real(np.array(np.sqrt(2) * DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate)))
x_nT = np.real(np.array(np.roll(np.real(DSP.convolve(r_nT, pulse_shape, mode="full")), -1)))
x_kTs = DSP.downsample(x_nT, sample_rate, offset=sample_rate)
detected_ints = communications.nearest_neighbor(x_kTs, bpsk_constellation)
message = communications.bin_to_char(detected_ints[data_offset:])
print(message)