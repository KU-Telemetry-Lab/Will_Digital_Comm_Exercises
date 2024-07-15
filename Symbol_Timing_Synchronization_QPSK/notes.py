import numpy as np
from scipy.interpolate import interp1d

def apply_clock_offset_to_upsampled(signal, offset=0.5, interpolation_mode="linear"):
    """
    Apply a small clock offset to an already upsampled signal.

    :param signal: Input signal (already upsampled).
    :param offset: Clock offset to apply (in symbol durations).
    :param interpolation_mode: Type of interpolation ("linear", "quadratic").
    :return: Signal with the clock offset applied.
    """
    # Determine the interpolation factor from the upsampled signal
    upsample_factor = int(np.sum(signal != 0) / len(signal))

    # Get the non-zero indices (original symbol locations)
    nonzero_indices = np.where(signal != 0)[0]
    nonzero_values = signal[nonzero_indices]

    # Interpolate the signal
    interpolation_function = interp1d(nonzero_indices, nonzero_values, kind=interpolation_mode, fill_value="extrapolate")
    interpolated_signal = interpolation_function(np.arange(len(signal)))

    # Apply the clock offset
    offset_samples = int(offset * upsample_factor)
    output_signal = np.zeros_like(interpolated_signal)
    output_signal[offset_samples::upsample_factor] = interpolated_signal[:-offset_samples:upsample_factor]
    
    return output_signal[:len(interpolated_signal) - offset_samples]

# Example usage
upsampled_signal = np.array([1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5])
offset = 0.5
result = apply_clock_offset_to_upsampled(upsampled_signal, offset=offset)
print(result)
