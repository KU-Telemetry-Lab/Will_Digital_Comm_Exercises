import numpy as np
from scipy import interpolate as intp
import matplotlib.pyplot as plt

class SCS:
    def __init__(self, samples_per_symbol, loop_bandwidth=None, damping_factor=None, gain=1, upsample_rate=3):
        '''
        Initialize the SCS (Symbol Clock Synchronization) subsystem class.

        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock-on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock-on to the timing error (similar to PLL).
        :param upsample_rate: Int type. Upsample rate of timing error correction interpolation.
        '''
        self.samples_per_symbol = samples_per_symbol
        self.upsample_rate = upsample_rate
        self.gain = gain
        
        self.compute_loop_constants(loop_bandwidth, damping_factor, k0=1, kp=1)
        self.k2_prev = 0
        
        self.adjusted_symbol_block = np.zeros(3, dtype=complex)
        self.timing_error_record = []
        self.loop_filter_record = []
        self.counter = 0

        self.sample_register = np.zeros(samples_per_symbol, dtype=complex)
        self.debugging_flag = False
        
    def compute_loop_constants(self, loop_bandwidth, damping_factor, k0, kp):
        """
        Compute the loop filter gains based on the loop bandwidth and damping factor.

        :param loop_bandwidth: Float type. Loop bandwidth of control loop.
        :param damping_factor: Float type. Damping factor of control loop.
        :param k0: Float type. Loop gain.
        :param kp: Float type. Proportional gain.
        """
        if loop_bandwidth is not None and damping_factor is not None:
            theta_n = (loop_bandwidth * (1 / self.samples_per_symbol) / self.samples_per_symbol) / (damping_factor + 1 / (4 * damping_factor))
            factor = (-4 * theta_n) / (1 + 2 * damping_factor * theta_n + theta_n**2)
            self.k1 = damping_factor * factor / kp
            self.k2 = theta_n * factor / kp
        else:
            self.k1 = 0
            self.k2 = 0
        self.k0 = k0
        self.kp = kp

    def direct_form_2(self, b, a, x):
        n = len(b)
        m = len(a)
        if n > m:
            max_len = n
            a = np.concatenate((a, np.zeros(n - m)))
        else:
            max_len = m
            b = np.concatenate((b, np.zeros(m - n)))
        denominator = a.copy()
        denominator[1:] = -denominator[1:]
        denominator[0] = 0
        x = np.concatenate((x, np.zeros(max_len - 1)))
        y = np.zeros(len(x), dtype=complex)
        delay_line = np.zeros(max_len, dtype=complex)
        for i, value in enumerate(x):
            y[i] = np.dot(b, delay_line)
            tmp = np.dot(denominator, delay_line)
            delay_line[1:] = delay_line[:-1]
            delay_line[0] = value * a[0] + tmp
        return y[1:]

    def get_timing_error_record(self):
        """
        Get the recorded timing errors.
        
        :return: Numpy array type. Recorded timing errors.
        """
        return np.array(self.timing_error_record)

    def get_loop_filter_record(self):
        """
        Get the recorded loop filter outputs.
        
        :return: Numpy array type. Recorded loop filter outputs.
        """
        return np.array(self.loop_filter_record)

    def interpolate(self, symbol_block, mode='cubic'):
        """
        Discrete signal upsample implementation.

        :param symbol_block: List or Numpy array type. Input signal.
        :param mode: String type. Interpolation mode ('linear' or 'cubic').
        :return: Numpy array type. Upsampled signal.
        """
        if mode == "linear":
            symbol_block_upsampled = np.zeros(len(symbol_block) * self.upsample_rate, dtype=complex)
            for i, sample in enumerate(symbol_block):
                symbol_block_upsampled[i * self.upsample_rate] = sample
            nonzero_indices = np.arange(0, len(symbol_block_upsampled), self.upsample_rate)
            nonzero_values = symbol_block_upsampled[nonzero_indices]
            new_indices = np.arange(len(symbol_block_upsampled))
            interpolation_function = intp.interp1d(nonzero_indices, nonzero_values, kind="linear", fill_value='extrapolate')
            symbol_block_interpolated = interpolation_function(new_indices)

        elif mode == "cubic":
            symbol_block = np.append(symbol_block, 0)
            interpolation_function = intp.CubicSpline(np.arange(0, len(symbol_block)), symbol_block)
            symbol_block_interpolated = interpolation_function(np.linspace(0, len(symbol_block)-1, num=(len(symbol_block)-1) * self.upsample_rate))

        else:
            symbol_block_interpolated = symbol_block
        return symbol_block_interpolated

    def loop_filter(self, timing_error):
        """
        Loop filter implementation.
        
        :param timing_error: Float type. The current timing error.
        :return: Float type. The output of the loop filter.
        """
        k2 = self.k2 * timing_error + self.k2_prev
        output = self.k1 * timing_error + k2
        self.k2_prev = k2

        self.loop_filter_record.append(output)
        return output

    def early_late_ted(self, early_sample, current_sample, late_sample):
        """
        Early-late Timing Error Detector (TED) implementation.
        
        :return: Float type. The calculated timing error.
        """
        timing_error = np.real(current_sample) * (np.real(late_sample) - np.real(early_sample))
        self.timing_error_record.append(timing_error)
        return timing_error

    def insert_new_sample(self, current_complex_sample):
        """
        Insert new samples for processing.

        :param current_complex_sample: Complex numpy dtype. Info.
        """

        if self.counter == self.samples_per_symbol - 1:
            # update samples register
            self.sample_register = self.sample_register[1:]
            self.sample_register = np.append(self.sample_register, current_complex_sample)

            corrected_symbol = 0

            self.counter = 0
            return corrected_symbol

        else:
            # update samples register
            self.sample_register = self.sample_register[1:]
            self.sample_register = np.append(self.sample_register, current_complex_sample)

            self.counter += 1
            return None
