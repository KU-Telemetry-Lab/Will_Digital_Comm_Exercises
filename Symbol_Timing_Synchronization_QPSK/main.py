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

###########################################################

class SCS:
    def __init__(self, input_signal_complex, samples_per_symbol, interpolation_factor):
        self.input_signal_real = np.real(input_signal_complex)
        self.input_signal_imag = np.imag(input_signal_complex)
        self.samples_per_symbol = samples_per_symbol
        self.counter = samples_per_symbol # start at sample
        self.timing_error = 0
        self.interpolation_factor = interpolation_factor

        self.loop_bandwidth = 0.2 * samples_per_symbol
        self.damping_factor = 1/np.sqrt(2)
        theta_n = self.loop_bandwidth * (1 / self.samples_per_symbol) / (self.damping_factor + 1/(4 * self.damping_factor))
        self.K1 = self.damping_factor
        self.K2 = theta_n
        self.LFK2Prev = 0

        self.adjusted_symbol_block_real = [0, 0, 0]
        self.adjusted_symbol_block_imag = [0, 0, 0]

        self.timing_error_record = []
        self.scs_output_real = []
        self.scs_output_imag = []

    def loop_filter(self, timing_error):
        LFK2 = self.K2 * timing_error + self.LFK2Prev
        lf_output = self.K1 * timing_error + LFK2
        self.LFK2Prev = LFK2
        return lf_output

    def get_timing_error(self):
        return self.timing_error_record

    def early_late_ted(self):
        timing_error = self.adjusted_symbol_block_real[1] * (self.adjusted_symbol_block_real[2] - self.adjusted_symbol_block_real[0]) + self.adjusted_symbol_block_imag[1] * (self.adjusted_symbol_block_imag[2] - self.adjusted_symbol_block_imag[0])
        self.timing_error_record.append(timing_error)
        if timing_error >= 1:
            timing_error = .99
        return timing_error

    def upsample(self, symbol_block):
        return DSP.upsample(symbol_block, self.interpolation_factor, interpolate=False)

    def interpolate(self, symbol_block):
        return interpolate(symbol_block, self.interpolation_factor, mode="linear")

    def runner(self):
        for i in range(len(self.input_signal_real)):
            if self.counter == sample_rate:
                # splice symbol block
                if i == 0:
                    symbol_block_real = np.concatenate((np.zeros(1), self.input_signal_real[i:i+2])) # early, on time, late
                    symbol_block_imag = np.concatenate((np.zeros(1), self.input_signal_imag[i:i+2]))
                else:
                    symbol_block_real = self.input_signal_real[i-1:i+2]
                    symbol_block_imag = self.input_signal_imag[i-1:i+2]
            
                self.timing_error = self.early_late_ted()
                loop_filter_output = self.loop_filter(self.timing_error)
                print(loop_filter_output)

                symbol_block_real_upsampled = self.upsample(symbol_block_real)
                symbol_block_imag_upsampled = self.upsample(symbol_block_imag)

                symbol_block_real_interpolated = self.interpolate(symbol_block_real_upsampled)
                symbol_block_imag_interpolated = self.interpolate(symbol_block_imag_upsampled)

                on_time_index = self.interpolation_factor + int(self.timing_error * self.interpolation_factor)

                self.scs_output_real.append(symbol_block_real_interpolated[on_time_index])
                self.scs_output_imag.append(symbol_block_imag_interpolated[on_time_index])

                self.adjusted_symbol_block_real = [
                    float(symbol_block_real_interpolated[on_time_index - self.interpolation_factor]),
                    float(symbol_block_real_interpolated[on_time_index]),
                    float(symbol_block_real_interpolated[on_time_index + self.interpolation_factor])
                ]

                self.adjusted_symbol_block_imag = [
                    float(symbol_block_imag_interpolated[on_time_index - self.interpolation_factor]),
                    float(symbol_block_imag_interpolated[on_time_index]),
                    float(symbol_block_imag_interpolated[on_time_index + self.interpolation_factor])
                ]

                self.counter = 0 # reset counter
            self.counter +=1 # increment counter
        return np.array(self.scs_output_real) + 1j * np.array(self.scs_output_imag)


###########################################################

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

test_input_1 = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
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
timing_offset = 0.0 # fractional offset in symbols
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
s_nT = s_nT_real + 1j * s_nT_imag

# 2.2 SYMBOL TIMING ERROR CORRECTION
scs = SCS(s_nT, sample_rate, interpolation_factor=10)
x_kTs = scs.runner()

# 2.3 MAKE A DECISION FOR EACH PULSE
detected_ints = communications.nearest_neighbor(x_kTs, qpsk_constellation)
error_count = error_count(b_k[len(header):-len(header)], detected_ints)
print(f"Transmission Symbol Errors: {error_count}")
print(f"Bit Error Percentage: {round(error_count * 2 / len(detected_ints), 2)} %")














# DEBUGGING!!!    

# plt.figure()
# plt.stem(np.imag(a_k[len(header):]))
# plt.title("Original Symbols")

# plt.figure()
# plt.stem(np.imag(a_k_upsampled[len(header)*sample_rate:-len(header)*sample_rate]))
# plt.title("Upsampled Signal")

# plt.figure()
# plt.stem(s_nT_imag)
# plt.title("Match Filtered Signal")

# plt.figure()
# plt.stem(symbol_block_imag)
# plt.title("Symbol Block")

# plt.figure()
# plt.stem(symbol_block_imag_upsampled)
# plt.title("Symbol Block Upsampled")

# plt.figure()
# plt.stem(symbol_block_imag_interpolated)
# plt.title("Symbol Block Interpolated")

# plt.figure()
# plt.stem(timing_sync_output_imag)
# plt.title("Timing Error Sync. Output")

# plt.figure()
# plt.stem(s_nT_real_downsampled)
# plt.title("TED Output Symbols")

plt.figure()
plt.stem(scs.get_timing_error())
plt.title("Timing Error")

plt.show()
