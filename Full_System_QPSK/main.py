import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy import interpolate as intp

import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
from KUSignalLib import communications

from PLL import PLL
from SCS import SCS


def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    for i in range(len(large_array) - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1

def string_to_ascii_binary(string, num_bits=7):
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    max_len = max(len(x), len(y))
    x = x + [0] * (max_len - len(x))
    y = y + [0] * (max_len - len(y))
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
###################################################################################################
xk_upsampled = DSP.upsample(xk, fs, interpolate_flag=False)
yk_upsampled = DSP.upsample(yk, fs, interpolate_flag=False)

# # plot upsampled symbols
# plt.figure()
# plt.stem(yk_upsampled[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
# plt.title("Upsampled Symbols")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()


# INTRODUCE TIMING OFFSET
###################################################################################################
timing_offset = 0.0
sample_shift = 0

xk_upsampled = clock_offset(xk_upsampled, fs, timing_offset)[sample_shift:]
yk_upsampled = clock_offset(yk_upsampled, fs, timing_offset)[sample_shift:]

# # plot offset symbols
# plt.figure()
# plt.stem(yk_upsampled[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
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
# plt.stem(yk_pulse_shaped[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
# plt.title("Pulse Shaped Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()

# print(f"\nFilter Length: {length} samples")
# print(f"Message Length: {alpha} percent")
# print(f"Sample Rate: {fs} samples per symbol\n")


# DIGITAL MODULATION
##################################################################################################
# synchronization offsets
fc_offset = 0.0005
phase_offset = 3 * np.pi 

s_rf = (
    np.sqrt(2) * np.real(DSP.modulate_by_exponential(xk_pulse_shaped, fc + fc_offset, fs)) +
    np.sqrt(2) * np.imag(DSP.modulate_by_exponential(yk_pulse_shaped, fc + fc_offset, fs))
) * np.exp(1j * phase_offset)

# # plot modulated RF signal
# plt.figure()
# plt.stem(s_rf[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
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
# plt.stem(yr_nT[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
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
# plt.stem(yr_nT_match_filtered[(len(header)+len(unique_word))*fs:(len(header)+len(unique_word)+5)*fs])
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
# plt.stem(yr_nT_downsampled[(len(header)+len(unique_word))*2:(len(header)+len(unique_word)+5)*2])
# plt.title("Downsampled by N/2 Signal")
# plt.xlabel("Sample Time [n]")
# plt.ylabel("Amplutide [V]")
# plt.show()

# plot_complex_points(r_nT, constellation=qpsk_constellation)


# UNIQUE WORD RESOLUTION
##################################################################################################
def check_unique_word(uw_register):
    uw_register = ''.join(uw_register)
    if uw_register in phase_ambiguities.keys():
        return phase_ambiguities[uw_register]
    else:
        return None

uw_register = ['0', '0', '0', '0', '0', '0', '0', '0']
uw_flag = False
uw_offset = 0


# SPECIFYING PLL AND SCS SYSTEM 
##################################################################################################
pll_loop_bandwidth = (fc/fs) * 0.06
pll_damping_factor = 1/np.sqrt(2)

scs_loop_bandwidth = (fc/fs) * 0.03
scs_damping_factor = 1/np.sqrt(2)

pll = PLL(sample_rate=2, loop_bandwidth=pll_loop_bandwidth, damping_factor=pll_damping_factor, open_loop=True)
scs = SCS(samples_per_symbol=2, loop_bandwidth=scs_loop_bandwidth, damping_factor=scs_damping_factor, open_loop=True)

# MEASURING PLL AND SCS SYSTEM GAIN
##################################################################################################

pll_max_lf_output = 0
scs_max_lf_output = 0
for i in range(len(r_nT)):
    pll_lf_output = pll.insert_new_sample(r_nT[i], i)
    scs_lf_output = scs.insert_new_sample(r_nT[i])

    if pll_lf_output > pll_max_lf_output:
        pll_max_lf_output = pll_lf_output

    if scs_lf_output > scs_max_lf_output:
        scs_max_lf_output = scs_lf_output

pll_gain = pll_max_lf_output
scs_gain = 1/scs_max_lf_output

# print(f"\nPLL Measured System Gain: {pll_gain}\n")
# print(f"\nSCS Measured System Gain: {scs_gain}\n")


# RUNNING PLL AND SCS SYSTEMS
##################################################################################################
pll = PLL(sample_rate=2, loop_bandwidth=pll_loop_bandwidth, damping_factor=pll_damping_factor, gain=pll_gain)
scs = SCS(samples_per_symbol=2, loop_bandwidth=scs_loop_bandwidth, damping_factor=scs_damping_factor, gain=scs_gain, invert=True)

detected_constellations = []
rotated_corrected_constellations = []
pll_error_record = []

dds_output = np.exp(1j * 0) # initial pll rotation

for i in range(len(r_nT)):
    # perform ccw rotation
    r_nT_ccwr = r_nT[i] * dds_output * np.exp(1j * uw_offset)

    # correct clock offset
    corrected_constellation = scs.insert_new_sample(r_nT_ccwr)
    if corrected_constellation is not None:
        rotated_corrected_constellations.append(corrected_constellation)

        # phase error calculation
        detected_symbol = communications.nearest_neighbor([corrected_constellation], qpsk_constellation)[0]
        detected_constellation = bits_to_amplitude[detected_symbol]
        detected_constellations.append(detected_constellation)
        
        # update unquie word register
        uw_register.pop(0)
        uw_register.append(str(detected_symbol))

        if uw_flag == False:
            received_unique_word = check_unique_word(uw_register)
            if received_unique_word is not None:
                uw_offset = received_unique_word
                uw_flag = True
        
        # calculating phase error
        phase_error = pll.phase_detector(corrected_constellation, detected_constellation)
        pll_error_record.append(phase_error)

        # feed into loop filter
        loop_filter_output = pll.loop_filter(phase_error)

        # feed into dds
        pll.dds(i, loop_filter_output)

        # generate next dds output
        dds_output = np.exp(1j * pll.get_current_phase())

print(f"Phase Ambiguity Rotation: {np.degrees(uw_offset)} deg\n")
plt.figure()
plt.plot(pll_error_record, label='Phase Error', color='r')
plt.title('Phase Error')
plt.xlabel('Sample Index')
plt.ylabel('Phase Error (radians)')
plt.grid()
plt.show()

plt.title("PLL Output Constellations")
plt.plot(np.real(rotated_corrected_constellations), np.imag(rotated_corrected_constellations), 'ro', label="Rotated Constellations")
plt.plot(np.real(detected_constellations), np.imag(detected_constellations), 'bo',  label="Esteimated Constellations")
plt.legend()
plt.grid(True)
plt.show()

# MAKE A DECISION FOR EACH PULSE
##################################################################################################
detected_symbols = communications.nearest_neighbor(detected_constellations[len(header)+len(unique_word)+1:], qpsk_constellation)

error_count = error_count(input_message_symbols[len(unique_word):], detected_symbols)
print(f"Transmission Symbol Errors: {error_count}")
print(f"Bit Error Percentage: {round((error_count * 2) / len(detected_symbols), 2)} %")

# converting symbols to binary then binary to ascii
detected_bits = []
for symbol in detected_symbols:
    detected_bits += ([*bin(symbol)[2:].zfill(2)])

message = communications.bin_to_char(detected_bits)
print(message)