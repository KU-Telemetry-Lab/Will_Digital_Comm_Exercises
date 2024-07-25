
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import examples as ex
from KUSignalLib import MatLab as ml
from KUSignalLib import DSP as dsp
from KUSignalLib import PLL
 
phaseShiftStart = 0.5*np.pi/2
phaseShift = 0
fs = 500
w0 = 10
w0_offset = 5

incomingSignal = []
internalSignal = []
phaseTrack = []
error = []
 
loop_bandwidth = 0.05 * fs
damping_factor = 1/np.sqrt(2)
pll = PLL.PLL(fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)

print(f"Kp: {pll.Kp}")
print(f"K0: {pll.K0}")
print(f"K1: {pll.K1}")
print(f"K2: {pll.K2}")

for n in range(1000):
    incomingSignal.append(np.exp(1j*((np.pi*2*(w0 + w0_offset)/fs)*n + phaseShiftStart)))
    internalSignal.append(pll.insert_new_sample(incomingSignal[n], n))
    
    error.append(pll.phase_detector(internalSignal[n], incomingSignal[n], Kp=1))
    phaseShift = pll.get_current_phase()
    phaseTrack.append(phaseShift)
 
plt.plot(error, label='Phase error')
# plt.plot(phaseTrack, label='Phase Track')
plt.plot(np.real(internalSignal), label='internal Signal')
plt.plot(np.real(incomingSignal), label='incoming Signal')
plt.legend()
plt.show()