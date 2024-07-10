
import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP
import numpy as np
from scipy import interpolate as intp
import matplotlib.pyplot as plt

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

num_samples = 9

phase_shift = np.pi / num_samples
x = np.linspace(0, np.pi, num_samples, endpoint=False)
y = np.sin(x)

# plt.stem(x, y)
# plt.grid(True)
# plt.show()

sample_rate = int((num_samples-1)/2)

counter = 0
tau = 0.5
upsample_factor = 10

output_sample = 0
symbol_array_upsampled = []
symbol_array_interpolated = []

for i in range(len(y)):
    if counter == sample_rate:
        early_index = i
        late_index = i+1
        symbol_array = y[i-sample_rate: i+sample_rate]
        symbol_array_upsampled = DSP.upsample(symbol_array, upsample_factor, interpolate=False)
        symbol_array_interpolated = interpolate(symbol_array_upsampled, upsample_factor, mode="cubic")
        output_sample = symbol_array_interpolated[(i*upsample_factor) + int(tau*upsample_factor)]
    counter += 1

plt.stem(symbol_array_upsampled, "ro")
plt.stem(symbol_array_interpolated, "bo")
plt.show()

print(output_sample)