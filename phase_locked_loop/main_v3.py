import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications


fc = 100
delta_fc = -0.01
fs = 1000
samples = 100
loop_bandwidths = np.array([0.02, 0.04, 0.06, 0.08, 0.10]) * fs

for B in loop_bandwidths:
    pll = DSP.PLL(B, carrier_estimate=fc, sample_rate=fs)

    pll_input = []
    pll_output = []
    pll_error = []

    for i in range(samples):
        pll_in = np.exp((1j * 2 * np.pi * fc * i)/ pll.fs) * np.exp(1j*(np.pi/2))
        # fc += delta_fc  # varying fc to make tracking visible
        pll_out = pll.Step(pll_in)

        pll_input.append(pll_in)
        pll_output.append(pll_out)
        pll_error.append(pll.GetPhaseError())

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(np.real(pll_input), label="PLL Input")
    plt.title("PLL Input")

    plt.subplot(2, 1, 1)
    plt.plot(np.real(pll_output), label="PLL Output", color='orange')
    plt.title("PLL Output")

    plt.subplot(2, 1, 2)
    plt.plot(pll_error, label="Phase Error", color='green')
    plt.title("Phase Error")
    plt.tight_layout()
    plt.show()
