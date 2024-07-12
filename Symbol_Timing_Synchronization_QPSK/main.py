import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications, SCS

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

# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25 * sample_rate
symbol_clock_offset = .1
qpsk_constellation = [[complex(np.sqrt(1) + np.sqrt(1) * 1j), 3],
                      [complex(np.sqrt(1) + -np.sqrt(1) * 1j), 2],
                      [complex(-np.sqrt(1) + -np.sqrt(1) * 1j), 0],
                      [complex(-np.sqrt(1) + np.sqrt(1) * 1j), 1]]

bits = [i[1] for i in qpsk_constellation]
bits_str = ['11', '10', '00', '01']
amplitudes = [i[0] for i in qpsk_constellation]
amplitude_to_bits = dict(zip(amplitudes, bits))
bits_to_amplitude = dict(zip(bits, amplitudes))
bits_to_bits_str = dict(zip(bits, bits_str))

test_input_1 = [1]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "this is a symbol timing error syncronization test and was developed by William Powers "
string_input_bin = ''.join(string_to_ascii_binary(string_input))
input_bin_blocks = [string_input_bin[i:i + 2] for i in range(0, len(string_input_bin), 2)]
test_input_3 = [int(bin2, 2) for bin2 in input_bin_blocks]

# SYNCHRONIZATION PARAMETERS
header = (3 * np.ones(10, dtype=int)).tolist()

# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = header + test_input_3 + header
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.upsample(a_k, sample_rate, interpolate=False)
a_k_upsampled_real = np.real(a_k_upsampled)
a_k_upsampled_imag = np.imag(a_k_upsampled)

# 1.2 INTRODUCE TIMING OFFSET
timing_offset = 0.3 # fractional offset in symbols
def fractional_delay(signal, delay):
    n = np.arange(len(signal))
    delayed_signal = np.interp(n - delay, n, signal)
    return delayed_signal

a_k_upsampled_real = fractional_delay(a_k_upsampled_real, timing_offset * sample_rate)
a_k_upsampled_imag = fractional_delay(a_k_upsampled_imag, timing_offset * sample_rate)

# 1.3 PULSE SHAPE (TRANSMIT)
length = 64
alpha = 0.5
pulse_shape = communications.srrc(alpha, sample_rate, length)
s_nT_real = np.real(np.roll(DSP.convolve(a_k_upsampled_real, pulse_shape, mode="same"), -1))
s_nT_imag = np.real(np.roll(DSP.convolve(a_k_upsampled_imag, pulse_shape, mode="same"), -1))

# 1.4 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR
s_nT_modulated = (
    np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_real, carrier_frequency, sample_rate))) +
    np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_imag, carrier_frequency, sample_rate)))
)

# 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR
r_nT_real = np.array(np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))
r_nT_imag = np.array(np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_nT_modulated, carrier_frequency, sample_rate)))

# 2.2 MATCH FILTER RECEIVED SIGNAL (+REMOVE HEADER AND TAIL)
x_nT_real = np.real(np.roll(DSP.convolve(r_nT_real, pulse_shape, mode="same"), -1))[len(header)*sample_rate:-len(header)*sample_rate]
x_nT_imag = np.real(np.roll(DSP.convolve(r_nT_imag, pulse_shape, mode="same"), -1))[len(header)*sample_rate:-len(header)*sample_rate]
x_nT = x_nT_real + 1j * x_nT_imag

# 2.3 SYMBOL TIMING ERROR CORRECTION
scs = SCS.SCS(x_nT, sample_rate, interpolation_factor=10)
x_kTs = scs.runner()

# 2.4 MAKE A DECISION FOR EACH PULSE
detected_ints = communications.nearest_neighbor(x_kTs, qpsk_constellation)
error_count = error_count(b_k[len(header):-len(header)], detected_ints)
print(f"Transmission Symbol Errors: {error_count}")
print(f"Bit Error Percentage: {round(error_count * 2 / len(detected_ints), 2)} %")

# 2.5 CONVERT BINARY TO ASCII
detected_bits = []
for symbol in detected_ints:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)




# # DEBUGGING!!!
# plt.figure()
# plt.stem(scs.get_timing_error())
# plt.title("Timing Error")

# plt.show()