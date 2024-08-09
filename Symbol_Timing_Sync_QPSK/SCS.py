import numpy as np

class SCS:
    def __init__(self, samples_per_symbol, loop_bandwidth, damping_factor, gain=1, strobe_start=False, mode='parabolic'):
        '''
        Initialize the SCS (Symbol Clock Synchronization) subsystem class.

        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock-on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock-on to the timing error (similar to PLL).
        :param gain: Float type. Gain added to timing error detector output (symbolizes Kp).
        :param strobe_start: Boolean type. Flag to start with strobe at beginning of simulation.
        :param mode: String type. Interpolation mode, either 'parabolic' or 'cubic'.
        '''
        self.samples_per_symbol = samples_per_symbol
        self.gain = gain
        self.strobe = strobe_start
        self.mode = mode
        self.compute_loop_constants(loop_bandwidth, damping_factor, k0=1, kp=gain)

        self.delay_register_1 = np.zeros(3, dtype=complex)
        self.delay_register_2 = np.zeros(3, dtype=complex)
        self.interpolated_register = np.zeros(3, dtype=complex)

        self.delta_e = 0
        self.delta_e_prev = 0
        self.LFK2_prev = 0

        self.ted_output_record = []
        self.loop_filter_output_record = []

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

    def insert_new_sample(self, input_sample):
        """
        Insert a new sample into the SCS system, performing interpolation and updating the timing error.

        :param input_sample: Numpy array. Input samples as complex numbers.
        :return: Complex. The interpolated output sample.
        """
        interpolated_sample = complex(0, 0)
        if self.mode == 'parabolic':  
            interpolated_sample = self.farrow_interpolator_parabolic(input_sample)
        else:  
            interpolated_sample = self.farrow_interpolator_cubic(input_sample)
        
        error = self.early_late_ted()
        filtered_error = self.loop_filter(error)
        
        self.strobe = not self.strobe
        if self.strobe:
            self.delta_e = self.delta_e_prev
    
        self.delta_e_prev = filtered_error
        self.interpolated_register = np.roll(self.interpolated_register, -1)
        self.interpolated_register[-1] = interpolated_sample

        return interpolated_sample

    def farrow_interpolator_parabolic(self, input_sample, row=0):
        """
        Perform parabolic interpolation on the input signal.

        :param input_sample: Numpy array. The input signal to be interpolated.
        :param row: Int type. The row in the delay buffers to use.
        :return: Complex. The interpolated output sample.
        """
        tmp = self.delta_e

        d1next = -0.5 * input_sample
        d2next = input_sample
    
        v2 = -d1next + self.delay_register_1[2] + self.delay_register_1[1] - self.delay_register_1[0]
        v1 = d1next - self.delay_register_1[2] + self.delay_register_2[1] + self.delay_register_1[1] + self.delay_register_1[0]
        v0 = self.delay_register_2[0]
        output = (((v2 * self.delta_e) + v1) * self.delta_e + v0)

        self.delay_register_1 = np.roll(self.delay_register_1, -1)
        self.delay_register_2 = np.roll(self.delay_register_2, -1)
        self.delay_register_1[-1] = d1next
        self.delay_register_2[-1] = d2next

        self.delta_e = tmp
        
        return output
    
    def farrow_interpolator_cubic(self, input_sample, row=0):
        """
        Perform cubic interpolation on the input signal.

        :param input: Numpy array. The input signal to be interpolated.
        :param row: Int type. The row in the delay buffers to use.
        :return: Complex. The interpolated output sample.
        """
        tmp = self.delta_e

        d1next = input_sample
        d2next = input_sample
        v3 = (1 / 6) * d1next - (1 / 2) * self.delay_register_1[2] + (1 / 2) * self.delay_register_1[1] - (1 / 6) * self.delay_register_1[0]
        v2 = (1 / 2) * self.delay_register_1[2] - self.delay_register_1[1] + (1 / 2) * self.delay_register_1[0]
        v1 = (-1 / 6) * d1next + self.delay_register_1[2] - (1 / 2) * self.delay_register_1[1] - (1 / 3) * self.delay_register_1[0]
        v0 = self.delay_register_2[0]
        output = ((v3 * self.delta_e + v2) * self.delta_e + v1) * self.delta_e + v0

        self.delay_register_1 = np.roll(self.delay_register_1, -1)
        self.delay_register_2 = np.roll(self.delay_register_2, -1)
        self.delay_register_1[-1] = d1next
        self.delay_register_2[-1] = d2next

        self.delta_e = tmp
        
        return output

    def early_late_ted(self):
        """
        Perform early-late timing error detection.

        :return: Float. The calculated timing error based on early and late samples.
        """
        out = 0
        if self.strobe:
            real_est = (self.interpolated_register[2].real - self.interpolated_register[0].real) * (-1 if self.interpolated_register[1].real < 0 else 1)
            imag_est = (self.interpolated_register[2].imag - self.interpolated_register[0].imag) * (-1 if self.interpolated_register[1].imag < 0 else 1)
            out = real_est * self.gain
            self.ted_output_record.append(out)
        return out
    
    def loop_filter(self, phase_error, k1=None, k2=None):
        """
        Apply a loop filter to the phase error to compute the filtered output.

        :param phase_error: Float type. The timing phase error to be filtered.
        :param k1: Float type. (Optional) The proportional gain; defaults to self.k1 if None.
        :param k2: Float type. (Optional) The integral gain; defaults to self.k2 if None.
        :return: Float. The filtered output from the loop filter.
        """
        if k1 is None:
            k1 = self.k1
        if k2 is None:
            k2 = self.k2
        LFK2 = k2 * phase_error + self.LFK2_prev
        output = k1 * phase_error + LFK2
        self.LFK2_prev = LFK2
        self.loop_filter_output_record.append(output)
        return output
