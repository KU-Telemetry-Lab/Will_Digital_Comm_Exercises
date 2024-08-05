import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP, PLL
from KUSignalLib import communications
from PLL import PLL


def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    for i in range(len(large_array) - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1

def string_to_ascii_binary(string, num_bits=7):
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    return sum(1 for i in range(len(x)) if x[i] != y[i])


# SYSTEM PARAMETERS
###################################################################################################
qpsk_constellation = [[complex( np.sqrt(1) +  np.sqrt(1)*1j), 3], 
                      [complex( np.sqrt(1) + -np.sqrt(1)*1j), 2], 
                      [complex(-np.sqrt(1) + -np.sqrt(1)*1j), 0], 
                      [complex(-np.sqrt(1) +  np.sqrt(1)*1j), 1]]
fs = 8 # sample rate
fc = .25 * fs # carrier frequency
input_message_ascii = "this is a qpsk transceiver test!"

# mapping the ascii characters to binary
input_message_bins = ''.join(string_to_ascii_binary(input_message_ascii))

# grouping the binary into blocks of two bits
input_message_blocks = [input_message_bins[i:i+2] for i in range(0, len(input_message_bins), 2)]

# mapping each block to a symbol in the constellation
input_message_symbols = [int(bin2, 2) for bin2 in input_message_blocks]

# adding unqiue word to symbols
unique_word = [0, 1, 2, 3, 0, 1, 2, 3]
phase_ambiguities = {
    "01230123": 0,
    "20312031": np.pi/2,
    "32103210": np.pi,
    "13021302": 3*np.pi/2
}

input_message_symbols = unique_word + input_message_symbols

bits_to_amplitude = {bit: amplitude for amplitude, bit in qpsk_constellation}

# inphase channel symbol mapping
xk = np.real([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# quadrature channel symbol mapping
yk = np.imag([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# adding header to each channel
header = np.ones(25)
xk = np.concatenate([header, xk])
yk = np.concatenate([header, yk])

# # plot original symbols
# plt.figure()
# plt.stem(yk[len(header):len(header)+5])
# plt.title("Symbols [0:5]")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplitude [V]")

# print(f"\nHeader Length: {len(header)} symbols")
# print(f"Message Length: {len(xk)} symbols")
# print(f"Sample Rate: {fs} samples per symbol")
# print(f"Carrier Frequency: {fc} Hz\n")
# plt.show()


# UPSAMPLING
###################################################################################################
xk_upsampled = DSP.upsample(xk, fs, interpolate_flag=False)
yk_upsampled = DSP.upsample(yk, fs, interpolate_flag=False)

# # plot upsampled symbols
# plt.figure()
# plt.stem(yk_upsampled[len(header)*fs:(len(header)+5)*fs])
# plt.title("Upsampled Symbols")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# PULSE SHAPE
###################################################################################################
length = 64
alpha = 0.10
pulse_shape = communications.srrc(alpha, fs, length)

xk_pulse_shaped = np.real(np.roll(DSP.convolve(xk_upsampled, pulse_shape, mode="same"), -1))
yk_pulse_shaped = np.real(np.roll(DSP.convolve(yk_upsampled, pulse_shape, mode="same"), -1))

# # plot pulse shaped signal
# plt.figure()
# plt.stem(yk_pulse_shaped[len(header)*fs:(len(header)+5)*fs])
# plt.title("Pulse Shaped Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")

# print(f"\nFilter Length: {length} samples")
# print(f"Message Length: {alpha} percent")
# print(f"Sample Rate: {fs} samples per symbol\n")
# plt.show()


# DIGITAL MODULATION
##################################################################################################
# synchronization offsets
fc_offset = 0.0001
phase_offset = np.pi/5

s_RF = (
    np.sqrt(2) * np.real(DSP.modulate_by_exponential(xk_pulse_shaped, fc + fc_offset, fs)) +
    np.sqrt(2) * np.imag(DSP.modulate_by_exponential(yk_pulse_shaped, fc + fc_offset, fs))
) * np.exp(1j * phase_offset)

# # plot modulated RF signal
# plt.figure()
# plt.stem(s_RF[len(header)*fs:(len(header)+5)*fs])
# plt.title("Modulated Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# DIGITAL DEMODULATIOIN
##################################################################################################
xr_nT = np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_RF, fc, fs))
yr_nT = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_RF, fc, fs))

# # plot demodulated signal
# plt.figure()
# plt.stem(yr_nT[len(header)*fs:(len(header)+5)*fs])
# plt.title("Demodulated Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# MATCH FILTER
##################################################################################################
xr_nT_match_filtered = np.real(np.roll(DSP.convolve(xr_nT, pulse_shape, mode="same"), -1))
yr_nT_match_filtered = np.real(np.roll(DSP.convolve(yr_nT, pulse_shape, mode="same"), -1))

# # plot match filtered signal
# plt.figure()
# plt.stem(yr_nT_match_filtered[len(header)*fs:(len(header)+5)*fs])
# plt.title("Match Filtered Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# DOWNSAMPLE EACH PULSE
##################################################################################################
xk = DSP.downsample(xr_nT_match_filtered, fs)[len(header):] # removing header
yk= DSP.downsample(yr_nT_match_filtered, fs)[len(header):] # removing header
rk = xk + 1j * yk

# communications.plot_complex_points(rk, constellation=qpsk_constellation)


# CARRIER PHASE SYNCHRONIZATION
##################################################################################################
loop_bandwidth = 0.02*fs
damping_factor = 1/np.sqrt(2)
pll = PLL(fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)

pll_detected_phase_record = []
pll_error_record = []
pll_loop_filter_record = []

dds_output = np.exp(1j * 0)
rotated_constellations = []
detected_constellations = []

for i in range(len(rk)):
    # perform ccw rotation
    rk_ccwr = rk[i] * dds_output
    rotated_constellations.append(rk_ccwr)

    # find nearest neighbor constellation
    detected_int = communications.nearest_neighbor([rk_ccwr], qpsk_constellation)[0]
    detected_constellation = bits_to_amplitude[detected_int]
    detected_constellations.append(detected_constellation)

    # calculate phase error
    phase_error = pll.phase_detector(rk_ccwr, detected_constellation)
    pll_error_record.append(phase_error)
    
    # feed into loop filter
    loop_filter_output = pll.loop_filter(phase_error)
    pll_error_record.append(loop_filter_output)

    # generate next dds output
    dds_output = np.exp(1j * loop_filter_output)

detected_symbols = communications.nearest_neighbor(detected_constellations, qpsk_constellation)

# UNIQUE WORD RESOLUTION
##################################################################################################
received_unique_word = None

# Find unique word testing phase ambiguities
for unique_word in phase_ambiguities.keys():
    unique_word_index = find_subarray_index([int(i) for i in unique_word], detected_symbols)
    if unique_word_index != -1:
        received_unique_word = unique_word
        break

if received_unique_word is not None:  # Use 'is not None' for comparison
    detected_constellations = np.array(detected_constellations[unique_word_index:]) * np.exp(-1j * phase_ambiguities[received_unique_word])
    print(f"\nReceived unique word: {received_unique_word}")
    print(f"Phase Ambiguity Rotation: {np.degrees(phase_ambiguities[received_unique_word])}\n")
else:
    print("\nUnique word not found.\n")

# MAKE A DECISION FOR EACH PULSE
##################################################################################################
detected_symbols = communications.nearest_neighbor(detected_constellations[len(unique_word):], qpsk_constellation)
symbol_errors = error_count(input_message_symbols[len(unique_word):], detected_symbols)
print(f"Transmission Symbol Errors: {symbol_errors}")
print(f"Bit Error Percentage: {round((symbol_errors * 2) / len(detected_symbols), 2)} %")

# constellation plotting
plt.plot(np.real(rotated_constellations), np.imag(rotated_constellations), 'ro')
plt.plot(np.real(detected_constellations), np.imag(detected_constellations), 'bo')
plt.show()

# converting symbols to binary then binary to ascii
detected_bits = []
for symbol in detected_symbols:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)

