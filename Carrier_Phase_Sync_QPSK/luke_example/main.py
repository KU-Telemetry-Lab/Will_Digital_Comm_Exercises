import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import math
sys.path.insert(0, '../../KUSignalLib/src')
from KUSignalLib import MatLab as ml
from KUSignalLib import DSP as dsp
from KUSignalLib import communications as com
from KUSignalLib import PLL


data = ml.load_matlab_file('qpskcruwdata.mat')
fc = 3.08
fs = 10
pulseShape = com.srrc(0.5, 8, 33)
uniqueWord = [1,0,1,0,1,0,1,0,0,1,0,1,0,1,0,1]
AverageEnergy = 2
K = 1
phaseRot = np.pi*2/2

constellation = [[complex( 1 +1j), 0b11],
                 [complex(-1 +1j), 0b01],
                 [complex(-1 -1j), 0b00],
                 [complex( 1 -1j), 0b10]]
# print(data[1])
frequencyShift = np.array(dsp.modulate_by_exponential(data[1], fc,fs, 0)) * np.sqrt(2)

filtered = dsp.direct_form_2(pulseShape,[1], frequencyShift)
downSample = dsp.downsample(filtered, 8, 8)

pll = PLL.PLL(loop_bandwidth = ((fc/fs)/0.02), damping_factor =(1/math.sqrt(2)), wstart=fc, fs=fs, kp = AverageEnergy*K, sampsPerSym = 2)

downSample = downSample[16:]
downSample = downSample * np.exp(-1j*phaseRot)


for i in range(len(downSample)-1):
    decision = com.nearest_neighbor([downSample[i]], constellation, binary = False)
    e = (np.imag(downSample[i])*np.real(decision[0]) - np.real(downSample[i])*np.imag(decision[0]))
    v = pll.loop_filter(e)
    pll.DDS(i, v)
    phase = pll.get_current_phase()
    downSample[i+1] = downSample[i+1] * np.exp(-1j*phase)


binValue = com.nearest_neighbor(downSample, constellation, binary = True)
binMessage = []
for val in binValue:
    binMessage = binMessage + ([*bin(val)[2:].zfill(2)])
binMessage = list(map(int, binMessage))
message = com.bin_to_char(binMessage[16:2422+16])
print(binMessage[:16])
print(message)
dsp.plot_complex_points(downSample, [sub_array[0] for sub_array in constellation])