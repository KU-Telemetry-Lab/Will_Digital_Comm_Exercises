
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import examples as ex
from KUSignalLib import MatLab as ml
from KUSignalLib import DSP as dsp

from PLL import PLL
from PLL2 import PLL2

# SYSTEM PARAMETERS
################################################################################################### 
fs = 500 

sig_freq = 10
sig_phase = np.pi / 4
n = np.arange(0,1000)
input_signal = np.exp(1j * ((2 * np.pi * (sig_freq) / fs) * n + (sig_phase)))


# Initialize lists to record simulation results
pll_input = []
pll_output = []
pll_detected_phase_record = []
pll_error_record = []

# PLL INSTANTIATION
################################################################################################### 
loop_bandwidth = 0.02 * fs
damping_factor = 1 / np.sqrt(2)
# pll = PLL(fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)
pll = PLL2(sample_rate=fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)

# Print loop filter gains for reference
print(f"K1: {pll.k1}")
print(f"K2: {pll.k2}")

# PLL SIMULATION
################################################################################################### 
for i in range(len(n)):
    # Insert the new sample into the PLL
    output_signal = pll.insert_new_sample(input_signal[i], i)
    
    # Record detected phase and error
    detected_phase = pll.get_current_phase()
    error = pll.phase_detector(output_signal, input_signal[i])

    # Update records
    pll_input.append(input_signal[i])
    pll_output.append(output_signal)
    pll_detected_phase_record.append(detected_phase)
    pll_error_record.append(error)

# PLOTTING RESULTS
################################################################################################### 
plt.figure(figsize=(12, 8))

# Subplot for Phase Error
plt.subplot(2, 1, 1)
plt.plot(pll_error_record, label='Phase Error', color='r')
plt.title('Phase Error')
plt.xlabel('Sample Index')
plt.ylabel('Phase Error (radians)')
plt.grid()

# Subplot for Input and Output Signals
plt.subplot(2, 1, 2)
plt.plot(np.real(pll_input), label='Input Signal', color='b', alpha=0.7)
plt.plot(np.real(pll_output), label='Output Signal', color='g', alpha=0.7)
plt.title('Input and Output Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()