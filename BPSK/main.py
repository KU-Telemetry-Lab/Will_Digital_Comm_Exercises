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

def srrc2(alpha, N, length):
    """
    Generates a square root raised cosine pulse.

    :param alpha: Roll-off or excess factor.
    :param N: Number of samples per symbol.
    :param length: Length of pulse, should be k*N+1 where k is an integer.
    :return: List. Square root raised cosine pulse.
    """
    pulse = []
    for n in range(length):
        n_shifted = n - np.floor(length / 2)
        if n_shifted == 0:
            num = np.pi * ((1 - alpha) * N)
            den = 4 * alpha
        else:
            num = np.sin(np.pi * n_shifted * (1 - alpha) / N) + 4 * alpha * n_shifted / N * np.cos(np.pi * n_shifted * (1 + alpha) / N)
            den = np.pi * n_shifted * (1 - (4 * alpha * n_shifted / N) ** 2) * np.sqrt(N)
        pulse.append(num / den if den != 0 else 0)
    return pulse


sample_rate = 8
pulse_shape = "NRZ" # change to SRRC (50% excess bandwidth)
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
string_input = "will is cool"
test_input_3 = [int(num) for num in ''.join(string_to_ascii_binary(string_input))]


# 1.1 UPSAMPLE THE BASEBAND DISCRETE SYMBOLS
b_k = test_input_1
a_k = [bits_to_amplitude[bit] for bit in b_k]
a_k_upsampled = DSP.Upsample(a_k, sample_rate, interpolate=False)
plt.plot(a_k)


# 1.2 NRZ FILTER THE UPSAMPLED SIGNAL (PULSE SHAPING)
filter_num = np.real(communications.srrc2(.5, 12, 37))
filter_denom = [1]
s_nT = DSP.DirectForm2(filter_num, filter_denom, a_k_upsampled)


# # 1.3 MODULATE ONTO CARRIER USING LOCAL OSCILLATOR (2)
# carrier_frequency = 2
# s_nT_modulated = local_oscillator_cos(s_nT, carrier_frequency, sample_rate)


# # 2.1 DEMODULATE THE RECEIVED SIGNAL USING LOCAL OSCILLATOR (1khz)
# r_nT = local_oscillator_cos(s_nT_modulated, carrier_frequency, sample_rate)


# 2.2 MATCH FILTER THE RECEIVED SIGNAL
x_nT = DSP.DirectForm2(filter_num, filter_denom, s_nT)


# 2.3 DOWNSAMPLE EACH PULSE
x_kTs = DSP.Downsample(x_nT[sample_rate:], sample_rate)


# 2.4 MAKE A DECISON FOR EACH PULSE
bpsk = [[complex(-1+0j), 0], [complex(1+0j), 1]]
detected_bits = communications.nearest_neighbor(x_kTs, bpsk)
print(f"Transmission Bit Errors: {error_count(b_k, detected_bits)}")
print(detected_bits)
plt.show()

# # 2.3 CONVERT BINARY TO ASCII
# message = communications.bin_to_char(detected_bits)
# print(message)


# # # 3 TEST SYSTEM ON GIVEN ASCII DATA
# # test_file = "bpskdata.mat"
# # input_message_length = 805 # symbols (115 ascii characters)
# # modulated_data = MatLab.loadMatLabFile(test_file)[1]
# # r_nT = [local_oscillator(modulated_data[i], carrier_frequency/sample_rate) for i in range(len(modulated_data))]

# # filter_num = [.25 for i in range(sample_rate)]
# # filter_denom = [1]
# # x_nT = DSP.DirectForm2(np.array(filter_num), np.array(filter_denom), r_nT)
# # x_kTs = DSP.Downsample(x_nT, sample_rate)

# # detected_bits = communications.nearest_neighbor(x_kTs[len(x_kTs)-input_message_length:], bpsk)
# # message = communications.bin_to_char(detected_bits)
# # print(message)
