import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
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

def interpolate(x, n, mode="linear"):
    """
    Perform interpolation on an upsampled signal.

    :param x: Input signal (already upsampled with zeros).
    :param n: Upsampled factor.
    :param mode: Interpolation type. Modes = "linear", "quadratic".
    :return: Interpolated signal.
    """
    nonzero_indices = np.arange(0, len(x), n)
    nonzero_values = x[nonzero_indices]
    interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind=mode, fill_value='extrapolate')
    new_indices = np.arange(len(x))
    interpolated_signal = interpolation_function(new_indices)
    return interpolated_signal

def gardner_ted(received_signal, samples_per_symbol):
    error = []
    for i in range(samples_per_symbol, len(received_signal) - samples_per_symbol, samples_per_symbol):
        early_sample = received_signal[i - samples_per_symbol // 2]
        on_time_sample = received_signal[i]
        late_sample = received_signal[i + samples_per_symbol // 2]

        timing_error = np.real(early_sample * (on_time_sample.conj() - late_sample.conj()))
        error.append(timing_error)

    return np.array(error)

# SYSTEM PARAMETERS
sample_rate = 8
carrier_frequency = 0.25 * sample_rate
symbol_clock_offset = 0
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

test_input_1 = [1, 0, 1, 0]
test_input_2 = [3, 2, 1, 0, 1, 2, 3]
string_input = "will is cool, this is a test"
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
timing_offset = 0.2  # fractional offset in symbols
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

# 2.1 PULSE SHAPE (RECEIVE) AND REMOVE HEADER AND TAIL
s_nT_real = np.real(np.roll(DSP.convolve(s_nT_real, pulse_shape, mode="same"), -1))[len(header)*sample_rate:-len(header)*sample_rate]
s_nT_imag = np.real(np.roll(DSP.convolve(s_nT_imag, pulse_shape, mode="same"), -1))[len(header)*sample_rate:-len(header)*sample_rate]

# 2.2 SYMBOL TIMING ERROR CORRECTION
timing_sync_output_real = []
timing_sync_output_imag = []

symbol_block_real = None 
symbol_block_real_upsampled = None
symbol_block_real_interpolated = None

counter = sample_rate # start at sample
tau = -timing_offset
upsample_factor = 10

for i in range(len(s_nT_real)):
    if counter == sample_rate:
        if i == 0:
            symbol_block_real = np.concatenate((np.zeros(1), s_nT_real[i:i+2])) # early, on time, late
            symbol_block_imag = np.concatenate((np.zeros(1), s_nT_imag[i:i+2]))
        else:
            symbol_block_real = s_nT_real[i-1:i+2]
            symbol_block_imag = s_nT_imag[i-1:i+2]

        symbol_block_real_upsampled = DSP.upsample(symbol_block_real, upsample_factor, interpolate=False)
        symbol_block_imag_upsampled = DSP.upsample(symbol_block_imag, upsample_factor, interpolate=False)


        symbol_block_real_interpolated = interpolate(symbol_block_real_upsampled, upsample_factor, mode="linear")
        symbol_block_imag_interpolated = interpolate(symbol_block_imag_upsampled, upsample_factor, mode="linear")
    
        output_sample_real = symbol_block_real_interpolated[upsample_factor + int(tau*upsample_factor)]
        output_sample_imag = symbol_block_imag_interpolated[upsample_factor + int(tau*upsample_factor)]

        timing_sync_output_real.append(output_sample_real)
        timing_sync_output_imag.append(output_sample_imag)

        counter = 0 # reset counter
    counter += 1

x_kTs = np.array(timing_sync_output_real) + 1j * np.array(timing_sync_output_imag)

# 2.5 MAKE A DECISION FOR EACH PULSE
detected_ints = communications.nearest_neighbor(x_kTs, qpsk_constellation)
print(f"Transmission Symbol Errors: {error_count(b_k[len(header):-len(header)], detected_ints)}")

# DEBUGGING!!!    

# plt.figure()
# plt.stem(np.imag(a_k[len(header):]))
# plt.title("Original Symbols")

# plt.figure()
# plt.stem(np.imag(a_k_upsampled[len(header)*sample_rate:-len(header)*sample_rate]))
# plt.title("Upsampled Signal")

plt.figure()
plt.stem(s_nT_imag)
plt.title("Match Filtered Signal")

# plt.figure()
# plt.stem(symbol_block_imag)
# plt.title("Symbol Block")

# plt.figure()
# plt.stem(symbol_block_imag_upsampled)
# plt.title("Symbol Block Upsampled")

# plt.figure()
# plt.stem(symbol_block_real_interpolated)
# plt.title("Symbol Block Interpolated")

# plt.figure()
# plt.stem(timing_sync_output_imag)
# plt.title("Timing Error Sync. Output")

# plt.figure()
# plt.stem(s_nT_real_downsampled)
# plt.title("TED Output Symbols")

plt.show()
