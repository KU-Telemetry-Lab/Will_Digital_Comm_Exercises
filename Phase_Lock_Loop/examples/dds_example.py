import pandas as pd
import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

# 1. Define tuning conversions using simulated system parameters
# Returns the frequency tuning word for a desired frequency output 
# Default: 
#    - Sample rate/input clock = 100MHz
#    - Phase Accumulator depth = 32 bits
def freq_to_tuning(freq_out, sample_rate=100000000, acc_depth=32):
    return int((freq_out * (2**acc_depth))/sample_rate)

# Returns the frequency output for a given frequency tuning word
# Default: 
#    - Sample rate/input clock = 100MHz
#    - Phase Accumulator depth = 32 bits
def tuning_to_freq(tuning_word, sample_rate=100000000, acc_depth=32):
    return (tuning_word * sample_rate) / float(2 ** acc_depth)

freq = 2822400
tuning = freq_to_tuning(freq)
freq_actual = tuning_to_freq(tuning)

# error = ((freq_actual - freq)/freq) * 100
# print(f"{freq} Hz signal: {tuning:#010x}")
# print(f"{freq_actual} Hz signal: {tuning:#010x} ({error})")


# 2. Generates a phase accumulator output for a given tuning word
phase = [ 0 ]
while len(phase) < (2**20):
    phase.append(phase[-1] + tuning)

plt.plot(phase)
plt.title("DDS Output")


# 3. Generate a rom table of a sine wave with signed integers 
df_phase = pd.DataFrame(zip(phase, [tuning] * len(phase)), columns=['phase', 'tuning'])
df_phase['phase_trunc'] = np.bitwise_and(df_phase['phase'], 0xFFC00000)
df_phase['phase_trunc'] = np.right_shift(df_phase['phase_trunc'], 20)

def generate_sin_rom(bits, rom_depth):
    # Subtract 1 from bit depth for signed data (msb is sign bit)
    amplitude = 2**(bit_depth-1)
    sin_rom = [int(amplitude * math.sin(a/rom_depth * 2 * math.pi)) for a in range(rom_depth)]
    return(sin_rom)

bit_depth = 12
rom_size = 4096
sin_rom = generate_sin_rom(bit_depth, rom_size)

# 4. Lambda function serves as phase-to-amplitude converter
df_phase['dds_out'] = df_phase['phase_trunc'].apply(lambda x: sin_rom[x])


# 5. Visualize the interrelation of the previously discussed components
data_fft = np.fft.fft(df_phase['dds_out'])
frequencies = np.abs(data_fft)
freq_scale = frequencies * 2 / len(frequencies)
db_freqs = freq_scale / float(2**(bit_depth))
db_freqs = (20 * np.log10(db_freqs))

# Shortening the time domain plot so sine features are visible
plot_len = 200

fig = plt.figure(figsize=(12,12))
plt.subplot(3,1,1)
plt.plot(df_phase['dds_out'][:plot_len])
plt.title("DDS Output")

plt.subplot(3,1,2)
plt.plot(df_phase['phase_trunc'][:plot_len])
plt.title("Phase Accumulator (Truncated)")

plt.subplot(3,1,3)
ax = fig.add_subplot(3,1,3)
ax.set_ylim(-120, 0)
ax.set_yticks([-120, -100, -80, -60, -40, -20, 0])
#ax.set_yscale('symlog')
# fft_ticklabels = ax.get_xticklabels()
# fft_ticklines = ax.get_xticklines()

fft_index = int(len(frequencies)/2) # Half of the FFT output 
plt.plot(db_freqs[:fft_index])
# plt.plot(db_freqs)

plt.title("Frequencies found")
# plt.xlim(0,fft_index)
# plt.show()

len_db_freqs = int(len(db_freqs)/2)
largest_peaks = sorted(db_freqs[:len_db_freqs])[-10:]
print(largest_peaks)
sfdr = largest_peaks[-1] - largest_peaks[-2]
print()
print(f"the SFDR of this signal is {sfdr} dBc")
