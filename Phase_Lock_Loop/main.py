import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import PLL, DSP
from KUSignalLib import communications


# system parameters
fs = 500
B = 0.08 * fs
samples = 1000

fc = 10
phase_offset = np.pi/4

pll_input = []
pll_output = []
pll_error = []
lf_output = []


fs = 500
B = 0.08 * fs
# pll parameters
zeta = 1 / np.sqrt(2)  # Damping factor
K0 = 1  # VCO gain
Kp = 1  # Phase detector gain
K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / fs)) / K0)  # Loop filter coefficient 1
K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / fs)) ** 2) / K0)  # Loop filter coefficient 2

print(f"K0: {K0}")
print(f"Kp: {Kp}")
print(f"K1: {K1}")
print(f"K2: {K2}")

pll = PLL.PLL(Kp, K0, K1, K2, fc, fs)

for i in range(1000):
    input_signal = np.exp(1j * (2 * np.pi * fc/fs * i + phase_offset))
    output_signal = pll.insert_new_sample(input_signal, i)
    phase_error = pll.phase_detector(input_signal, output_signal, Kp=1)
    phase_shift = pll.get_current_phase()

    pll_input.append(input_signal)
    pll_output.append(output_signal)
    pll_error.append(phase_error)
    lf_output.append(phase_shift)


plt.plot(pll_error, label='phase error')
plt.plot(np.real(pll_input), label='input signal')
plt.plot(np.real(pll_output), label='output signal')
plt.legend()
plt.show()
    







