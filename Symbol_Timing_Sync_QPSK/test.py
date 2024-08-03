import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications, SCS

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
test_file = "qpskscdata.mat" # (need to figure out data set import)
modulated_data = MatLab.load_matlab_file(test_file)[1]
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
loop_bandwidth = 0.2*sample_rate
damping_factor = 1/np.sqrt(2)
scs = SCS.SCS(sample_rate, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor, upsample_rate=10)

for i in range(len(x_nT)):
    if i == 0: # edge case (start)
        scs.insert_new_sample(0, x_nT[i], x_nT[i+1])
    elif i == len(x_nT)-1: # edge case (end)
        scs.insert_new_sample(x_nT[i-1], x_nT[i], 0)
    else:
        scs.insert_new_sample(x_nT[i-1], x_nT[i], x_nT[i+1])

x_kTs = scs.get_scs_output_record()
timing_error_record = scs.get_timing_error_record()
loop_filter_record = scs.get_loop_filter_record()

DSP.plot_complex_points(x_kTs, referencePoints=amplitudes)

plt.figure()
plt.stem(timing_error_record, "ro", label="TED")
plt.stem(loop_filter_record, "bo", label="Loop Filter")
plt.title("Synchronization Error Records")
plt.legend()
plt.show()
