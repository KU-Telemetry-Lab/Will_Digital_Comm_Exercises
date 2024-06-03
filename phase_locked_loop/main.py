import numpy as np
import math
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications


def complex_sinusoid(length, frequency, phase):
    t = np.arange(length)
    omega = 2 * np.pi * frequency / length
    complex_signal = np.exp(1j * (omega * t + phase))
    return complex_signal

def loop_filter(x, Bn, Ts, zeta=1/math.sqrt(2), K0=1):
    # x = input signal
    # Bn = loop bandwidth
    # Ts = symbol rate
    # zeta = damping factor
    # K0 = ?
    K1 = ((4*zeta)/(zeta+(1/(4*zeta))))*((Bn*Ts)/K0)
    K2 = ((4)/(zeta+(1/(4*zeta)))**2)*(((Bn*Ts)**2)/K0)

    filter_num = [(K1+K2), K2]
    filter_denom = [1]
    y = DSP.DirectForm2(filter_num, filter_denom, x)
    return y

def dds(x, fc, K0=1):
    # x = input signal
    # fc = estimated carrier signal frequency
    # K0 = ?
    y = DDS



length = 1000  # Length of the complex sinusoid
frequency = 10  # Frequency of the sinusoid
phase = 0  # Phase of the sinusoid
complex_signal = complex_sinusoid(length, frequency, phase)

# Plotting Complex Sinusoid
plt.figure(figsize=(10, 5))

plt.subplot(2, 1, 1)
plt.plot(np.real(complex_signal), label='Real part')
plt.plot(np.imag(complex_signal), label='Imaginary part')

plt.subplot(2, 1, 2)
plt.plot(np.abs(complex_signal), label='Magnitude')
plt.legend()

plt.tight_layout()
plt.show()

