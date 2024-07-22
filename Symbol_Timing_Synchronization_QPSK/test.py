import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications, SCS

def early_late_ted(early_sample, on_time_sample, late_sample, sample_rate):
    timing_error = on_time_sample * (late_sample - early_sample)
    timing_error = timing_error * (1/sample_rate)
    print(timing_error)
    return timing_error


# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25*sample_rate
symbol_clock_offset = 0
qpsk_constellation = [[complex( np.sqrt(4.5)+ np.sqrt(4.5)*1j), 3], 
                      [complex( np.sqrt(4.5)+-np.sqrt(4.5)*1j), 2], 
                      [complex(-np.sqrt(4.5)+-np.sqrt(4.5)*1j), 0], 
                      [complex(-np.sqrt(4.5)+ np.sqrt(4.5)*1j), 1]]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


# TEST SYSTEM ON GIVEN ASCII DATA
test_file = "qpskdata.mat" # (need to figure out data set import)
modulated_data = MatLab.load_matlab_file(test_file)[1][0:200]
data_offset = 12
pulse_shape = communications.srrc(.5, sample_rate, 32)

# DEMODULATE
r_nT_real = np.sqrt(2) * np.real(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate, phase=carrier_frequency))
r_nT_imag = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(modulated_data, carrier_frequency, sample_rate, phase=carrier_frequency))

# MATCH FILTER
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))
x_nT = x_nT_real + 1j * x_nT_imag

# SYMBOL TIMING SYNCHRONIZATION

for i in range(0, len(x_nT), sample_rate):
    if i == 0: # edge case (start)
        early_late_ted(0, x_nT[i], x_nT[i+1], sample_rate)
    elif i == len(x_nT)-1: # edge case (end)
        early_late_ted(x_nT[i-1], x_nT[i], 0, sample_rate)
    else:
        early_late_ted(x_nT[i-1], x_nT[i], x_nT[i+1], sample_rate)

