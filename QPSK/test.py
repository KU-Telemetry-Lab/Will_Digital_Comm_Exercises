import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications

sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0
qpsk_constellation = [[complex( 1+ 1j), 3], [complex( 1+-1j), 2], [complex(-1+-1j), 0], [complex(-1+ 1j), 1]]

bits = [3, 2, 0, 1]
bits_str = ['11', '10', '00', '01']
amplitudes = [complex( 1+ 1j), complex( 1+-1j), complex(-1+-1j), complex(-1+ 1j)]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "qpskdata.mat"
modulated_data = MatLab.load_matlab_file(test_file)[1]
input_message_length = 0
data_offset = 12
carrier_frequency = 2
sample_rate = 8
pulse_shape = communications.srrc(.5, sample_rate, 32)

r_nT = np.array(np.sqrt(2) * DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate), dtype=complex)
x_nT = np.array(np.roll(np.convolve(r_nT, pulse_shape, mode="full"), -1), dtype=complex)
x_kTs = DSP.downsample(x_nT, sample_rate, offset=sample_rate)
detected_symbols = communications.nearest_neighbor(x_kTs, qpsk_constellation)
detected_bits = []
for symbol in detected_symbols:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])
message = communications.bin_to_char(detected_bits[data_offset:])
print(message)