import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications


# system parameters
fc = 100
delta_fc = -0.01
fs = 1000
samples = 100
B = 0.06 * fs

# pll parameters
zeta = 1 / np.sqrt(2)  # Damping factor
K0 = 1  # VCO gain
Kp = 1  # Phase detector gain
K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / fs)) / K0)  # Loop filter coefficient 1
K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / fs)) ** 2) / K0)  # Loop filter coefficient 2

pll = DSP.PLL(Kp, K0, K1, K2, fc, fs)

pll_input = []
pll_output = [0]
pll_error = []

for i in range(samples):
    pll_in = np.exp(1j * 2 * np.pi * fc * i / fs)
    fc += delta_fc # varying fc to make tracking visible
    e_nT = pll.phaseDetector(pll_in, pll_output[-1])
    v_nT = pll.loopFilter(e_nT)
    pll_out = pll.DDS(v_nT, pll.getCurentPhase())

    pll_input.append(pll_in)
    pll_output.append(pll_out)
    pll_error.append(e_nT)

plt.plot(np.real(pll_input))
plt.plot(np.real(pll_output[1:]))
plt.show()




