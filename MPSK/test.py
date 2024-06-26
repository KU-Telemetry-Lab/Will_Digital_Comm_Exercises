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

mpsk_constellation = [
    [complex(3 + 0*1j), 0],
    [complex(np.sqrt(4.5) + np.sqrt(4.5)*1j), 1],
    [complex(0 + 3*1j), 3],
    [complex(-np.sqrt(4.5) + np.sqrt(4.5)*1j), 2],
    [complex(-3 + 0*1j), 6],
    [complex(-np.sqrt(4.5) + -np.sqrt(4.5)*1j), 7],
    [complex(0 + -3*1j), 5],
    [complex(np.sqrt(4.5) + -np.sqrt(4.5)*1j), 4]
]

bits = [i[1] for i in mpsk_constellation]
bits_str = ['000', '001', '011', '010', '110', '111', '101', '100']
amplitudes = [i[0] for i in mpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "psk8data.mat"
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

# DSP.plot_complex_points(x_kTs, referencePoints=amplitudes) # plotting received constellations

detected_symbols = communications.nearest_neighbor(x_kTs, mpsk_constellation)[data_offset:]
detected_bits = []
for symbol in detected_symbols:
    detected_bits += list(bits_to_bits_str[symbol])
message = communications.bin_to_char(detected_bits)
print(message)