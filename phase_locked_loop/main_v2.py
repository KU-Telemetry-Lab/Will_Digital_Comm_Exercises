import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications

class PLL:
    def __init__(self, loop_bandwidth, K0=1, KP=1):
        self.loop_bandwidth = loop_bandwidth
        self.zeta = 1/np.sqrt(2)
        self.K0 = K0
        self.KP = KP

        self.phase_error = 0.0
        self.phase_output = 0.0
        self.freq_output = 0.0
        self.dds_output = np.exp(1j*self.phase_output)

    def phase_error_detector(self, x):
        self.phase_error = self.KP * np.angle(x*np.conj(self.dds_output))

    def dds(self):
        self.dds_output = self.K0 * np.exp(1j*np.conj(self.phase_output)) 
        
    def loop_filter(self):
        self.freq_output += self.loop_bandwidth * self.phase_error
        self.phase_output += self.zeta * self.phase_error + self.freq_output

    def step(self, x):
        self.phase_error_detector(x)
        self.loop_filter()
        self.dds()

samples = 1000
fc = 10
fs = 1000
delta_fc = 5
B = 0.002*fs

pll_input = []
pll_output = []
pll_phase_error = []

pll = PLL(B)

for i in range(samples):
    complex_input = np.exp(1j*2*np.pi*(fc/fs))
    fc += delta_fc
    pll.step(complex_input)

    pll_input.append(complex_input)
    pll_output.append(pll.dds_output)
    pll_phase_error.append(pll.phase_error)

plt.plot(np.real(pll_input), label="PLL Input")
plt.plot(np.real(pll_output), label="PLL Output")
plt.plot(np.real(pll_phase_error), label="Phase Error")
plt.legend()
plt.title("PLL Results")
plt.show()

