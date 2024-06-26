
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import examples as ex
from KUSignalLib import MatLab as ml
from KUSignalLib import DSP as dsp
from KUSignalLib import PLL
 
test = dsp.phase_difference([1+0j], [-1+0j])
phaseShiftStart = 0.5*np.pi/2
phaseShift = 0
fs = 500
w0 = 10
w0_offset = 5*np.pi*2/fs
incomingSignal = []
internalSignal = []
phaseTrack = []
error = []
 
pll = PLL.PLL(kp=0.02, k0=1, k1=1, k2=0.005, wstart=w0, fs=fs, thetaStart = 0)
for n in range(1000):
    incomingSignal.append(np.exp(1j*((np.pi*2*w0/fs+w0_offset)*n + phaseShiftStart)))
   
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