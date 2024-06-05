import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications


class PLL:
    def __init__(self, loop_bandwidth, carrier_estimate, sample_rate=1):
        """
        Initialize the Phase-Locked Loop (PLL) with the given parameters.

        :param loop_bandwidth: Loop bandwidth of the PLL.
        :param carrier_estimate: Initial estimate of the carrier frequency.
        :param sample_rate: Sampling rate of the input signal.
        """
        self.B = loop_bandwidth
        self.fc = carrier_estimate
        self.fs = sample_rate

        self.zeta = 1 / np.sqrt(2)  # Damping factor
        self.K0 = 1  # VCO gain
        self.Kp = 1  # Phase detector gain
        self.K1 = ((4 * self.zeta) / (self.zeta + (1 / (4 * self.zeta)))) * ((self.B * (1 / self.fs)) / self.K0)  # Loop filter coefficient 1
        self.K2 = ((4) / (self.zeta + (1 / (4 * self.zeta))) ** 2) * (((self.B * (1 / self.fs)) ** 2) / self.K0)  # Loop filter coefficient 2

        self.loop_filter_mem = 0.0  # Loop filter memory
        self.dds_internal_mem = 0.0  # DDS internal memory
        self.dds_output_mem = np.exp(1j * 2 * np.pi * self.fc / self.fs)  # DDS output memory

    def phase_detector(self, x):
        """
        Perform phase detection on the input signal.

        :param x: Input signal.
        :return: Detected phase error scaled by the phase detector gain.
        """
        return np.angle(x / self.dds_output_mem, deg=False) * self.K

    def loop_filter(self, x):
        """
        Apply the loop filter to the detected phase error.

        :param x: Detected phase error.
        :return: Filtered phase error.
        """
        term1 = self.K1 * x
        term2 = self.K2 * x + self.loop_filter_mem
        self.loop_filter_mem = term2
        return term1 + term2

    def dds(self, x):
        """
        Perform Direct Digital Synthesis (DDS) to generate the next output.

        :param x: Input to the DDS (filtered phase error).
        :return: DDS output signal.
        """
        temp = self.fc / self.fs + x + self.dds_internal_mem
        self.dds_internal_mem = temp
        self.dds_output_mem = np.exp(1j * 2 * np.pi * temp)
        return self.dds_output_mem

    def step(self, x):
        """
        Step through the PLL modules.

        :param x: Input signal to PLL (complex sinusoid sample).
        """
        e_nT = self.phase_detector(x)
        v_nT = self.loop_filter(e_nT)
        pll_out = self.dds(v_nT)
        return pll_out

#################################################################

fc = 100
delta_fc = -0.01
fs = 1000
samples = 100
loop_bandwidths = np.array([0.001, 0.02, 0.04, 0.06, 0.08, 0.10]) * fs

for B in loop_bandwidths:
    pll = PLL(B, carrier_estimate=fc, sample_rate=fs)

    pll_input = []
    pll_output = []
    pll_error = []

    for i in range(samples):
        pll_in = np.exp(1j * 2 * np.pi * fc * i / pll.fs)
        fc += delta_fc  # varying fc to make tracking visible
        pll_out = pll.step(pll_in)

        pll_input.append(pll_in)
        pll_output.append(pll_out)
        pll_output.append(np.angle(pll_out / pll_in, deg=False))

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




