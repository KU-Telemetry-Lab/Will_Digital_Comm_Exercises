import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications


class PLL:
    def __init__(self, loop_bandwidth, carrier_estimate, sample_rate=1):
        self.B = loop_bandwidth
        self.fc = carrier_estimate
        self.fs = sample_rate

        self.zeta = 1/np.sqrt(2)
        self.K0 = 1
        self.Kp = 1
        # self.K1 = ((4*self.zeta)/(self.zeta+(1/(4*self.zeta))))*((self.B*(1/self.fs))/self.K0)
        # self.K2 = ((4)/(self.zeta+(1/(4*self.zeta)))**2)*(((self.B*(1/self.fs))**2)/self.K0)
        self.K1 = 0.1247
        self.K2 = 0.0083

        self.loop_filter_mem = 0.0
        self.dds_internal_mem = 0.0
        self.dds_output_mem = np.exp(1j*2*np.pi*self.fc/self.fs)

    def phase_detector(self, x):
        return np.angle(x / self.dds_output_mem, deg=False) * self.Kp

    def loop_filter(self, x):
        term1 = self.K1 * x
        term2 = self.K2 * x + self.loop_filter_mem
        self.loop_filter_mem = term2
        return term1 + term2

    def dds(self, x):
        temp = self.fc/self.fs + x + self.dds_internal_mem
        self.dds_internal_mem = temp
        self.dds_output_mem = np.exp(1j * 2 * np.pi * temp)
        return self.dds_output_mem

#################################################################

fc = 5
delta_fc = +.005
samples = 500

pll = PLL(5, carrier_estimate=fc, sample_rate=100)

pll_input = []
pll_output = []
pll_error = []

for i in range(samples):
    pll_in = np.exp(1j * 2 * np.pi * fc * i / pll.fs)
    fc += delta_fc  # varying fc to make tracking visible
    e_nT = pll.phase_detector(pll_in)
    v_nT = pll.loop_filter(e_nT)
    pll_out = pll.dds(v_nT)

    pll_input.append(pll_in)
    pll_output.append(pll_out)
    pll_error.append(e_nT)

plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
plt.plot(np.real(pll_input), label="PLL Input")
plt.title("PLL Input")

plt.subplot(3, 1, 2)
plt.plot(np.real(pll_output), label="PLL Output", color='orange')
plt.title("PLL Output")

plt.subplot(3, 1, 3)
plt.plot(pll_error, label="Phase Error", color='green')
plt.title("Phase Error")
plt.tight_layout()
plt.show()




