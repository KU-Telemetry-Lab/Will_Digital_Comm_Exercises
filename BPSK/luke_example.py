data = ml.loadMatLabFile('bpskdata.mat')
fc = 1
fs = 8
pulseShape = com.srrc2(0.5, 8, 33)
# print(data[1])
frequencyShift = np.array(dsp.dFreqShiftModualation(data[1], fc, fs)) * np.sqrt(2)

filtered = dsp.DirectForm2(pulseShape,[1], frequencyShift)
downSample = dsp.Downsample(filtered, 8, 8)
points = com.nearest_neighbor(downSample)

chars = com.bin_to_char(points[16:])
print(chars)
plt.plot(filtered)
plt.show()