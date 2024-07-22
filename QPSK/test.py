import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications

# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0

qpsk_constellation = [
    [complex( np.sqrt(4.5)+ np.sqrt(4.5)*1j), 3], 
    [complex( np.sqrt(4.5)+-np.sqrt(4.5)*1j), 2], 
    [complex(-np.sqrt(4.5)+-np.sqrt(4.5)*1j), 0], 
    [complex(-np.sqrt(4.5)+ np.sqrt(4.5)*1j), 1]
]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "qpskdata.mat"
modulated_data = MatLab.load_matlab_file(test_file)[1]
data_offset = 12
sample_rate = 8
carrier_frequency = .25 * sample_rate
pulse_shape = communications.srrc(.5, sample_rate, 32)

r_nT_real = np.sqrt(2) * np.real(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))
r_nT_imag = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate))
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))
x_kTs_real = np.array(DSP.downsample(x_nT_real, sample_rate))
x_kTs_imag = np.array(DSP.downsample(x_nT_imag, sample_rate))
x_kTs = x_kTs_real + 1j * x_kTs_imag

DSP.plot_complex_points(x_kTs, referencePoints=amplitudes) # plotting received constellations

detected_symbols = communications.nearest_neighbor(x_kTs, qpsk_constellation)
detected_bits = []
for symbol in detected_symbols:
    detected_bits += list(bits_to_bits_str[symbol])
message = communications.bin_to_char(detected_bits[data_offset:])
print(message)