import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications
from SCS import SCS
# from SCS2 import SCS

def string_to_ascii_binary(string, num_bits=7):
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    # Make the lengths of x and y equal by appending zeros to the shorter one
    max_len = max(len(x), len(y))
    x = x + [0] * (max_len - len(x))
    y = y + [0] * (max_len - len(y))
    
    # Count errors
    return sum(1 for i in range(max_len) if x[i] != y[i])


def clock_offset(signal, sample_rate, offset_fraction):
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction
    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset 
    x_shifted = interpolator(t_shifted)
    return x_shifted

def plot_complex_points(complex_array, constellation):
    plt.plot([point.real for point in complex_array], [point.imag for point in complex_array], 'ro', label='Received Points')
    for point, label in constellation:
        plt.plot(point.real, point.imag, 'b+', markersize=10)
        plt.text(point.real, point.imag, f' {label}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Complex Constellation Plot')
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()


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

bits_to_amplitude = {bit: amplitude for amplitude, bit in qpsk_constellation}

# inphase channel symbol mapping
xk = np.real([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# quadrature channel symbol mapping
yk = np.imag([bits_to_amplitude[symbol] for symbol in input_message_symbols])

# adding header to each channel
header = [1,0] * 50
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
# ###################################################################################################
xk_upsampled = DSP.upsample(xk, fs, interpolate_flag=False)
yk_upsampled = DSP.upsample(yk, fs, interpolate_flag=False)

# # plot upsampled symbols
# plt.figure()
# plt.stem(yk_upsampled[len(header)*fs:(len(header)+5)*fs])
# plt.title("Upsampled Symbols")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()

# # plot upsampled constellation
# plot_complex_points((xk_upsampled + 1j * yk_upsampled), constellation=qpsk_constellation)


# INTRODUCE TIMING OFFSET
###################################################################################################
timing_offset = 0.5
sample_shift = 0

xk_upsampled = clock_offset(xk_upsampled, fs, timing_offset)[sample_shift:]
yk_upsampled = clock_offset(yk_upsampled, fs, timing_offset)[sample_shift:]

# # plot timing offset constellation
# plot_complex_points((xk_upsampled + 1j*yk_upsampled), constellation=qpsk_constellation)


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
s_rf = (
    np.sqrt(2) * np.real(DSP.modulate_by_exponential(xk_pulse_shaped, fc, fs)) +
    np.sqrt(2) * np.imag(DSP.modulate_by_exponential(yk_pulse_shaped, fc, fs))
)

# # plot modulated RF signal
# plt.figure()
# plt.stem(s_RF[len(header)*fs:(len(header)+5)*fs])
# plt.title("Modulated Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# DIGITAL DEMODULATIOIN
##################################################################################################
xr_nT = np.sqrt(2) * np.real(DSP.modulate_by_exponential(s_rf, fc, fs))
yr_nT = np.sqrt(2) * np.imag(DSP.modulate_by_exponential(s_rf, fc, fs))

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
r_nT = xr_nT_match_filtered + 1j * yr_nT_match_filtered

# # plot match filtered signal
# plt.figure()
# plt.stem(yr_nT_match_filtered[len(header)*fs:(len(header)+5)*fs])
# plt.title("Match Filtered Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# DOWNSAMPLE BY N/2
##################################################################################################
xr_nT_downsampled = DSP.downsample(xr_nT_match_filtered, int(fs/2))
yr_nT_downsampled = DSP.downsample(yr_nT_match_filtered, int(fs/2))
r_nT = (xr_nT_downsampled + 1j* yr_nT_downsampled)

# # plot downsampled by N/2 signal
# plt.figure()
# plt.stem(yr_nT_downsampled[len(header)*2:(len(header)+5)*2])
# plt.title("Downsampled by N/2 Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()

# plot_complex_points(r_nT, constellation=qpsk_constellation)


# SYMBOL TIMING SYNCHRONIZATION
##################################################################################################
loop_bandwidth = (fc/fs)*0.03
damping_factor = 1/np.sqrt(2)

# # measuring scs system gain
# scs = SCS(samples_per_symbol=2, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor, open_loop=True)

# max_lf_output = 0
# for i in range(len(r_nT)):
#     lf_output = scs.insert_new_sample(r_nT[i])
#     if lf_output > max_lf_output:
#         max_lf_output = lf_output

# print(f"\nSCS Measured System Gain: {1/max_lf_output}\n")

# running scs system
scs = SCS(samples_per_symbol=2, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor, gain=42)
corrected_constellations = []
for i in range(len(r_nT)):
    corrected_constellation = scs.insert_new_sample(r_nT[i])
    if corrected_constellation is not None:
        corrected_constellations.append(corrected_constellation)

# plot_complex_points(corrected_constellations, constellation=qpsk_constellation)

# MAKE A DECISION FOR EACH PULSE
##################################################################################################
detected_symbols = communications.nearest_neighbor(corrected_constellations, qpsk_constellation)

# removing header and adjusting for symbol timing synchronization delay
detected_symbols = detected_symbols[len(header)+2:]

error_count = error_count(input_message_symbols, detected_symbols)

print(f"Transmission Symbol Errors: {error_count}")
print(f"Bit Error Percentage: {round((error_count * 2) / len(detected_symbols), 2)} %")

# converting symbols to binary then binary to ascii
detected_bits = []
for symbol in detected_symbols:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)