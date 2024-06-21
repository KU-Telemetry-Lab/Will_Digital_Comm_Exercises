import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import MatLab
from KUSignalLib import communications


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

def freq_shift_modulation(input, f_c, f_s, phase_offset=0):
    indices = np.arange(len(input))
    complex_exponentials = np.exp(1j * (2 * np.pi * f_c * indices / f_s + phase_offset))
    output = input * complex_exponentials
    return output



sample_rate = 8
pulse_shape = "NRZ" # change to SRRC (50% excess bandwidth)
symbol_clock_offset = 0

bits = [3, 2, 0, 1]
bits_str = ['11', '10', '00', '01']
amplitudes = [complex( 1+ 1j), complex( 1+-1j), complex(-1+-1j), complex(-1+ 1j)]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))


test_input_1 = [0, 0, 0, 2, 0, 0, 2]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "will is cool"
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i+2] for i in range(0, len(string_input_bin), 2)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_1
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.Upsample(a_k, sample_rate, interpolate=False)

# 1.2 PULSE SHAPE THE UPSAMPLED SIGNAL (SRRC)
length = 64
alpha = 0.5
pulse_shape = communications.srrc2(.5, sample_rate, length)
s_nT = np.array(
    np.roll(np.real(convolve(np.real(a_k_upsampled), pulse_shape, mode="same")), -1) + 
    1j * np.roll(np.real(convolve(np.imag(a_k_upsampled), pulse_shape, mode="same")), -1), 
    dtype=complex
)

# 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR 
fc = 1
s_nT_modulated = np.array(
    (np.sqrt(2) * freq_shift_modulation(np.real(s_nT), fc, sample_rate, phase_offset=0)) - 
    (np.sqrt(2) * freq_shift_modulation(np.imag(s_nT), fc, sample_rate, phase_offset=np.pi/2))
)

# # 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT = np.array(
    (np.sqrt(2) * freq_shift_modulation(np.real(s_nT_modulated), fc, sample_rate, phase_offset=0)) - 
    1j * (np.sqrt(2) * freq_shift_modulation(np.imag(s_nT_modulated), fc, sample_rate, phase_offset=np.pi/2))
)

# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT = np.array(
    np.roll(np.real(convolve(np.real(s_nT_modulated), pulse_shape, mode="same")), -1) + 
    1j * np.roll(np.real(convolve(np.imag(r_nT), pulse_shape, mode="same")), -1), 
    dtype=complex
)

# 2.3 DOWNSAMPLE EACH PULSE
x_kTs = DSP.Downsample(x_nT, sample_rate)

# 2.5 MAKE A DECISION FOR EACH PULSE
qpsk = [[complex( 1+ 1j), 3], [complex( 1+-1j), 2], [complex(-1+-1j), 0], [complex(-1+ 1j), 1]]
detected_bits = communications.nearest_neighbor(x_kTs, qpsk)
print(f"Transmission Bit Errors: {error_count(b_k, detected_bits)}")
print(detected_bits)

# Plot original symbols
plt.figure()
plt.stem(np.real(a_k))
plt.title("Original Symbols")

# Plot upsampled symbols
plt.figure()
plt.stem(np.real(a_k_upsampled))
plt.title("Upsampled Symbols")

# Plot modulated signal
plt.figure()
plt.stem(s_nT_modulated)
plt.title("Modulated Signal")

# Plot demodulated signal
plt.figure()
plt.stem(r_nT)
plt.title("Demodulated Signal")

# Plot match filtered signal
plt.figure()
plt.stem(x_nT)
plt.title("Match Filtered Signal")

# Plot downsampled signal
plt.figure()
plt.stem(np.real(x_kTs))
plt.title("Downsampled Signal")
plt.show()


