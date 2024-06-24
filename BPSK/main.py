import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import MatLab
from KUSignalLib import communications


def local_oscillator_cos(input, f_c, f_s):
    output = []
    for i in range(len(input)):
        output.append(input[i] * np.sqrt(2) * np.cos(2*np.pi*f_c*i/f_s))
    return output

def string_to_ascii_binary(string, num_bits=7):
    ascii_binary_strings = []
    for char in string:
        ascii_binary = bin(ord(char))[2:].zfill(num_bits)
        ascii_binary_strings.append(ascii_binary)
    return ascii_binary_strings

def error_count(x, y):
    count = 0
    for i in range(len(x)):
        if (x[i] != y[i]):
            count += 1
    return count

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

sample_rate = 8
carrier_frequnecy = .125*8
symbol_clock_offset = 0


A = 1 # incoming signal base amplitude
amplitudes = [-1, 1]
bits = [0, 1]
bin_strs = ['0', '1']
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bin_str = dict(zip(bits, bin_strs))


test_input_1 = [1, 0, 0, 1]
test_input_2 = [1, 1, 0, 0, 1, 1, 0, 0]
string_input = "...will is cool, this is a test..."
test_input_3 = [int(num) for num in ''.join(string_to_ascii_binary(string_input))]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_3
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate=False)

# 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
length = 64
alpha = 0.5
pulse_shape = communications.srrc(.5, sample_rate, length)
s_nT = np.real(np.array(np.roll(np.real(convolve(a_k_upsampled, pulse_shape, mode="same")), -1)))

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR 
s_nT_modulated = np.real(np.array(np.sqrt(2) * np.array(DSP.modulate_by_exponential(s_nT, carrier_frequnecy, sample_rate))))

# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT = np.real(np.array(np.sqrt(2) * np.array(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequnecy, sample_rate))))

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT = np.real(np.array(np.roll(np.real(convolve(r_nT, pulse_shape, mode="same")), -1)))

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs = DSP.downsample(x_nT, sample_rate)

# 2.4 MAKE A DECISON FOR EACH PULSE
bpsk_constellation = [[complex(-1+0j), 0], [complex(1+0j), 1]]
detected_bits = communications.nearest_neighbor(x_kTs, bpsk_constellation)
print(f"Transmission Bit Errors: {error_count(b_k, detected_bits)}")

# 2.3 CONVERT BINARY TO ASCII
message = communications.bin_to_char(detected_bits)
print(message)


# # 3 TEST SYSTEM ON GIVEN ASCII DATA
# test_file = "bpskdata.mat"
# modulated_data = MatLab.load_matlab_file(test_file)[1]
# input_message_length = 805 # symbols (115 ascii characters)
# data_offset = 16
# carrier_frequency = 1
# sample_rate = 8
# pulse_shape = communications.srrc(.5, sample_rate, 32)

# r_nT = np.real(np.array(np.sqrt(2) * freq_shift_modulation(modulated_data, carrier_frequency, sample_rate)))
# x_nT = np.real(np.array(np.roll(np.real(convolve(r_nT, pulse_shape, mode="full")), -1)))
# x_kTs = DSP.downsample(x_nT, sample_rate, offset=sample_rate)
# detected_ints = communications.nearest_neighbor(x_kTs, bpsk_constellation)
# message = communications.bin_to_char(detected_ints[data_offset:])
# print(message)
