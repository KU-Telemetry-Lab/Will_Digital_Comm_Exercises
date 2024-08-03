import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications, SCS

def string_to_ascii_binary(string, num_bits=7):
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    return sum(1 for i in range(len(x)) if x[i] != y[i])

def clock_offset(signal, sample_rate, offset_fraction):
    t = np.arange(0, len(signal) / sample_rate, 1 / sample_rate)
    clock_offset = (1/sample_rate) * offset_fraction
    interpolator = intp.interp1d(t, signal, kind='linear', fill_value='extrapolate')
    t_shifted = t + clock_offset 
    x_shifted = interpolator(t_shifted)
    return x_shifted


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


# INTRODUCE TIMING OFFSET (NEEDS WORK)
###################################################################################################
timing_offset = 0.0
xk_upsampled = clock_offset(xk_upsampled, fs, timing_offset)
yk_upsampled = clock_offset(yk_upsampled, fs, timing_offset)

amplitudes = [i[0] for i in qpsk_constellation]
# DSP.plot_complex_points((xk_upsampled + 1j*yk_upsampled), referencePoints=amplitudes)


# PULSE SHAPE
###################################################################################################
length = 64
alpha = 0.10
pulse_shape = communications.srrc(alpha, fs, length)

xk_pulse_shaped = np.real(DSP.convolve(xk_upsampled, pulse_shape, mode="same"))
yk_pulse_shaped = np.real(DSP.convolve(yk_upsampled, pulse_shape, mode="same"))

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
s_RF = (
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
xr_nT_match_filtered = np.real(DSP.convolve(xr_nT, pulse_shape, mode="same"))
yr_nT_match_filtered = np.real(DSP.convolve(yr_nT, pulse_shape, mode="same"))
r_nT = xr_nT_match_filtered + 1j * yr_nT_match_filtered

# # plot match filtered signal
# plt.figure()
# plt.stem(yr_nT_match_filtered[len(header)*fs:(len(header)+5)*fs])
# plt.title("Match Filtered Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# SYMBOL TIMING SYNCHRONIZATION
##################################################################################################
loop_bandwidth = 0.2*fs
damping_factor = 1/np.sqrt(2)
scs = SCS.SCS(fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)

for i in range(len(r_nT)):
    if i == 0: # edge case (start)
        scs.insert_new_sample(0, r_nT[i], r_nT[i+1])
    elif i == len(r_nT)-1: # edge case (end)
        scs.insert_new_sample(r_nT[i-1], r_nT[i], 0)
    else:
        scs.insert_new_sample(r_nT[i-1], r_nT[i], r_nT[i+1])

# clock synchronized symbol outputs (removing header)
rk = scs.get_scs_output_record()[len(header):]

# scs output records
timing_error_record = scs.get_timing_error_record()
loop_filter_record = scs.get_loop_filter_record()

# DSP.plot_complex_points(rk, referencePoints=amplitudes)

# plt.figure()
# plt.stem(timing_error_record, "ro", label="TED")
# plt.stem(loop_filter_record, "bo", label="Loop Filter")
# plt.title("SCS Output Records")
# plt.legend()
# plt.show()


# MAKE A DECISION FOR EACH PULSE
##################################################################################################
detected_symbols = communications.nearest_neighbor(rk, qpsk_constellation)
error_count = error_count(input_message_symbols, detected_symbols)
print(f"Transmission Symbol Errors: {error_count}")
print(f"Bit Error Percentage: {round((error_count * 2) / len(detected_symbols), 2)} %")

# converting symbols to binary then binary to ascii
detected_bits = []
for symbol in detected_symbols:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)
