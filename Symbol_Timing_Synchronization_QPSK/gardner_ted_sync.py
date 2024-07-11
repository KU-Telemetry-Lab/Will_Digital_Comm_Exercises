import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample

# Parameters
num_symbols = 1000
symbol_rate = 1e3
samples_per_symbol = 8
sample_rate = symbol_rate * samples_per_symbol
time = np.arange(num_symbols * samples_per_symbol) / sample_rate

# Generate random QPSK symbols
symbols = np.random.randint(0, 4, num_symbols)
constellation = {0: 1+1j, 1: 1-1j, 2: -1+1j, 3: -1-1j}
qpsk_signal = np.array([constellation[symbol] for symbol in symbols])

# Upsample the QPSK signal
qpsk_signal_upsampled = np.zeros(len(qpsk_signal) * samples_per_symbol)
qpsk_signal_upsampled[::samples_per_symbol] = qpsk_signal

# Introduce the timing offset
timing_offset = 0.25  # Fractional offset in symbols
delayed_signal = resample(qpsk_signal_upsampled, int(len(qpsk_signal_upsampled) + timing_offset * samples_per_symbol))

# Plot the original and delayed signals
plt.figure(figsize=(10, 6))
plt.plot(np.real(qpsk_signal_upsampled), label='Original Signal')
plt.plot(np.real(delayed_signal[:len(qpsk_signal_upsampled)]), label='Delayed Signal', linestyle='--')
plt.legend()
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('QPSK Signal with Timing Offset')
plt.show()

def gardner_ted(received_signal, samples_per_symbol):
    """
    Gardner Timing Error Detector for timing recovery.
    """
    error = []
    for i in range(samples_per_symbol, len(received_signal) - samples_per_symbol, samples_per_symbol):
        early_sample = received_signal[i - samples_per_symbol // 2]
        on_time_sample = received_signal[i]
        late_sample = received_signal[i + samples_per_symbol // 2]

        timing_error = np.real(early_sample * (on_time_sample.conj() - late_sample.conj()))
        error.append(timing_error)

    return np.array(error)

# Calculate timing error using Gardner TED
timing_error = gardner_ted(delayed_signal, samples_per_symbol)

# Plot timing error
plt.figure(figsize=(10, 6))
plt.plot(timing_error)
plt.xlabel('Symbol Index')
plt.ylabel('Timing Error')
plt.title('Timing Error using Gardner TED')
plt.show()
