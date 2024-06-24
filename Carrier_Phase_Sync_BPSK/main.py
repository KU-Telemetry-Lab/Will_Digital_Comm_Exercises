import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import MatLab
from KUSignalLib import communications

class PLL():
    '''
    This class is used to simulate a PLL discretely.
    Components can be called individually or as a whole depending on user need.
    Use as an object and initialize variables in init if you want full functionality.
    '''
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, kp=1, k0=1, k1=0, k2=0, wstart=1, thetaStart=0, fs=1):
        '''
        :param kp: Float type. Proportional gain.
        :param k0: Float type. DDS gain.
        :param k1: Float type. Loop filter gain feed-forward.
        :param k2: Float type. Loop filter gain feed-back.
        :param wstart: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency.
        Initialize the PLL object for repeated use, if left blank the object will be initialized with default values.
        '''
        self.Kp = kp
        self.K0 = k0
        self.K1 = k1
        self.K2 = k2
        self.w0 = wstart
        self.phase = thetaStart
        self.sigOut = np.exp(1j * thetaStart)
        self.fs = fs

    def InsertNewSample(self, incomingSignal, n, internalSignal=None):
        """
        :param incomingSignal: Complex number. The current sample of the received signal.
        :param internalSignal: Complex number. The current signal your LO is at. Will use default from constructor if left blank.
        :param n: Int type. The current sample index, used to insert a new sample of the received signal and LO.
        If using as an object, this is the index of the only function you need to call to achieve PLL functionality.
        """
        if internalSignal is None:
            internalSignal = np.exp(1j * (2 * np.pi * (self.w0 / self.fs) * n + self.phase))
        phaseError = self.PhaseDetector(internalSignal, incomingSignal)
        V_t = self.LoopFilter(phaseError)
        pointOut = self.DDS(n, V_t)
        return pointOut

    def PhaseDetector(self, sample1, sample2, Kp=None):
        """
        Phase detector implementation.

        :param sample1: Complex number. First point.
        :param sample2: Complex number. Second point.
        :param Kp: Float type. Proportional gain, should be less than one.
        :return: Float type. Phase difference between the two points.
        """
        if Kp is None:
            Kp = self.Kp
        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle * Kp

    def LoopFilter(self, phaseError, K1=None, K2=None):
        """
        Loop filter implementation.
        :param phaseError: Float type. Phase error.
        :param K1: Float type. Loop filter gain according to Fig C.2.6.
        :param K2: Float type. Loop filter gain according to Fig C.2.6.
        """
        if K1 is None:
            K1 = self.K1
        if K2 is None:
            K2 = self.K2
        LFK2 = K2 * phaseError + self.LFK2prev
        output = K1 * phaseError + LFK2
        self.LFK2prev = LFK2
        return output

    def DDS(self, n, v, k0=None, w0=None, fs=None):
        """
        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :param k0: Float type. DDS gain.
        :param w0: Float type. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type. Sampling frequency.
        DDS implementation.
        """
        if k0 is None:
            k0 = self.K0
        if w0 is None:
            w0 = self.w0
        if fs is None:
            fs = self.fs
        self.phase += v * k0
        self.sigOut = np.exp(1j * (2 * np.pi * (w0 / fs) * n + self.phase))
        return self.sigOut
    
    def GetCurrentPhase(self):
        return self.phase

def modulate_by_exponential(x, f_c, f_s, phase_offset=0):
    y = []
    for i, value in enumerate(x):
        modulation_factor = np.exp(1j * 2 * np.pi * f_c * i / f_s + phase_offset)
        y.append(value * modulation_factor)
    return np.array(y)

def convolve(x, h, mode='full'):
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    h_padded = np.pad(h, (0, N - len(h)), mode='constant')
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    y = np.fft.ifft(X * H)

    if mode == 'same':
        start = (len(h) - 1) // 2
        end = start + len(x)
        y = y[start:end]
    return y

def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    large_len = len(large_array)
    for i in range(large_len - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1

def string_to_ascii_binary(string, num_bits=7):
    ascii_binary_strings = []
    for char in string:
        ascii_binary = bin(ord(char))[2:].zfill(num_bits)
        ascii_binary_strings.append(ascii_binary)
    return ascii_binary_strings


sample_rate = 8
carrier_frequency = sample_rate * 0.2
pulse_shape = "SRRC"
unique_word = [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1]

A = 1
amplitudes = [-1, 1]
bits = [0, 1]
bin_strs = ['0', '1']
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bin_str = dict(zip(bits, bin_strs))

test_input_1 = [1, 0, 0, 1]
test_input_2 = [1, 1, 0, 0, 1, 1, 0, 0]
string_input = "will is cool, this is a test"
test_input_3 = [int(num) for num in ''.join(string_to_ascii_binary(string_input))]

# # 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
# b_k = unique_word + test_input_3
# a_k = [bits_to_amplitude[bit] for bit in b_k]
# a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate=False)

# # 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
# length = 64
# alpha = 0.5
# pulse_shape = communications.srrc(.5, sample_rate, length)
# s_nT = np.array(np.roll(np.real(convolve(a_k_upsampled, pulse_shape, mode="same")), -1))

# # 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR 
# s_nT_modulated = np.array(np.sqrt(2) * modulate_by_exponential(s_nT, carrier_frequency, sample_rate, phase_offset=np.pi/2))

# # 2.1 DEMODULATE THE RECEIVED SIGNAL
# r_nT_real = np.real(np.array(np.sqrt(2) * modulate_by_exponential(s_nT_modulated, sample_rate, carrier_frequency, phase_offset=0)))
# r_nT_imag = np.imag(np.array(np.sqrt(2) * modulate_by_exponential(s_nT_modulated, sample_rate, carrier_frequency, phase_offset=0)))

# # 2.2 MATCH FILTER THE RECEIVED SIGNAL
# x_nT_real = np.array(np.roll(convolve(np.real(r_nT_real), pulse_shape, mode="same"), -1))
# x_nT_imag = np.array(np.roll(convolve(np.imag(r_nT_imag), pulse_shape, mode="same"), -1))

# # 2.3 DOWNSAMPLE EACH PULSE
# x_kTs_real = DSP.downsample(x_nT_real, sample_rate, offset=0)
# x_kTs_imag = DSP.downsample(x_nT_imag, sample_rate, offset=0)

# B = 0.02 * sample_rate
# fs = sample_rate
# fc = carrier_frequency
# zeta = 1 / np.sqrt(2)
# K0 = 1
# Kp = 1
# K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / fs)) / K0)
# K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / fs)) ** 2) / K0)
# pll = PLL(Kp, K0, K1, K2, fc, fs)

# detected_bits = []
# dds_output = 0 + 1j*0
# sample_indexes = np.arange(len(x_kTs_real))

# for sample_index in sample_indexes:
#     # ccw rotation via mixing
#     x_kTs_real_ccwr = modulate_by_exponential([x_kTs_real[sample_index]], np.real(dds_output), sample_rate)
#     x_kTs_imag_ccwr = modulate_by_exponential([x_kTs_imag[sample_index]], np.imag(dds_output), sample_rate)

#     # decision
#     bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]
#     detected_bit_real = communications.nearest_neighbor(x_kTs_real_ccwr, bpsk_constellation)[0]
#     detcted_bit_imag = communications.nearest_neighbor(x_kTs_imag_ccwr, bpsk_constellation)[0]
#     detected_bits.append(detected_bit_real)

#     # input to pll
#     phase_detector_output = pll.PhaseDetector(x_kTs_real_ccwr + 1j*x_kTs_imag_ccwr, detected_bit_real + 1j*detcted_bit_imag)
#     loop_filter_output = pll.LoopFilter(phase_detector_output)
#     dds_output = pll.DDS(sample_index, loop_filter_output[0])

# uw_start_index = find_subarray_index(unique_word, detected_bits)
# detected_bits = detected_bits[uw_start_index + len(unique_word): uw_start_index + len(unique_word) + len(b_k)]
# message = communications.bin_to_char(detected_bits)
# print(message)


# 3 TEST ON REAL DATA
test_file = "bpskcruwdata.mat"
data_offset = 16
input_message_length = 2247
modulated_data = MatLab.load_matlab_file(test_file)[1]
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)

r_nT = np.array(np.sqrt(2) * modulate_by_exponential(modulated_data, sample_rate, carrier_frequency, phase_offset=0))
x_nT = np.array(np.roll(convolve(np.real(r_nT), pulse_shape, mode="same"), -1))

x_kTs_real = DSP.downsample(np.real(x_nT), sample_rate, offset=0)
x_kTs_imag = DSP.downsample(np.imag(x_nT), sample_rate, offset=0)

B = 0.02 * sample_rate
fs = sample_rate
fc = carrier_frequency
zeta = 1 / np.sqrt(2)
K0 = 1
Kp = 1
K1 = ((4 * zeta) / (zeta + (1 / (4 * zeta)))) * ((B * (1 / fs)) / K0)
K2 = ((4) / (zeta + (1 / (4 * zeta))) ** 2) * (((B * (1 / fs)) ** 2) / K0)
pll = PLL(Kp, K0, K1, K2, fc, fs)

detected_bits = []
dds_output = 0 + 1j*0
sample_indexes = np.arange(len(x_kTs_real))

for sample_index in sample_indexes:
    # ccw rotation via mixing
    x_kTs_real_ccwr = modulate_by_exponential([x_kTs_real[sample_index]], np.real(dds_output), sample_rate)
    x_kTs_imag_ccwr = modulate_by_exponential([x_kTs_imag[sample_index]], np.imag(dds_output), sample_rate)

    # decision
    bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]
    detected_bit_real = communications.nearest_neighbor(x_kTs_real_ccwr, bpsk_constellation)[0]
    detcted_bit_imag = communications.nearest_neighbor(x_kTs_imag_ccwr, bpsk_constellation)[0]
    detected_bits.append(detected_bit_real)

    # input to pll
    phase_detector_output = pll.PhaseDetector(x_kTs_real_ccwr + 1j*x_kTs_imag_ccwr, detected_bit_real + 1j*detcted_bit_imag)
    loop_filter_output = pll.LoopFilter(phase_detector_output)
    dds_output = pll.DDS(sample_index, loop_filter_output[0])

uw_start_index = find_subarray_index(unique_word, detected_bits)
print(uw_start_index)
# detected_bits = detected_bits[uw_start_index + len(unique_word): uw_start_index + len(unique_word) + input_message_length]
# message = communications.bin_to_char(detected_bits)
# print(message)