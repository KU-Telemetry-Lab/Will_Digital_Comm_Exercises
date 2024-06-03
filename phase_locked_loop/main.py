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


################## PLL SPECIFIC FUNCTIONS #######################

def phaseDetector(sample1, sample2, Kp=1):
    """
    Phase detector implementation.
    """
    return np.angle(sample2 / sample1, deg=False) * Kp

def loopFilter(x, Bn, Ts, zeta=1/np.sqrt(2), K0=1):
    """
    Loop filter implementation (PI controller).
    """
    K1 = ((4*zeta)/(zeta+(1/(4*zeta))))*((Bn*Ts)/K0)
    K2 = ((4)/(zeta+(1/(4*zeta)))**2)*(((Bn*Ts)**2)/K0)
    
    filter_num = np.array([K1 + K2, -K1], dtype=np.float64)
    filter_denom = np.array([1, -1], dtype=np.float64)
    y = np.convolve(x, filter_num, mode='same')
    return y

dds_prev = 0  # dds implementation dependent on previous calculated value

def dds(x, fc, K0=1):
    """
    Direct Digital Synthesis (DDS) implementation.
    """
    global dds_prev
    dds_prev += x + fc * 2 * np.pi
    y = K0 * dds_prev
    return np.cos(y) + 1j * np.sin(y)

def complex_sinusoid(length, frequency, phase=0):
    t = np.arange(length)
    return np.exp(1j * (2 * np.pi * frequency * t + phase))

#################################################################
# Simulation parameters
length = 1000
fc = 10
phase = 0
test_input = complex_sinusoid(length, fc, phase)

# System parameters
Ts = 1/fc
Bn = 0.005
fc_estimated = 11

# Recording parameters
pll_input = []
pll_output = [np.cos(fc_estimated) + 1j*np.sin(fc_estimated)]

# Set up in streaming framework
for i in range(len(test_input)):
    e_nT = phaseDetector(test_input[i], pll_output[-1])
    v_nT = loopFilter([e_nT], Bn, Ts)[-1]
    dds_out = dds(v_nT, fc_estimated)

    pll_input.append(test_input[i])
    pll_output.append(dds_out)

pll_input = np.array(pll_input)
pll_output = np.array(pll_output)

###################### PLOTTING ################################
# Plotting Complex Sinusoid
plt.figure(figsize=(10, 5))
plt.plot(np.real(test_input), label='Input Real part')
plt.plot(np.real(pll_output[:-1]), label='Output Real part')
plt.legend()
plt.title('Input and Output of PLL')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True)

plt.show()