import numpy as np

class SCS:
    def __init__(self, samples_per_symbol, loop_bandwidth, damping_factor, gain=1, open_loop=False):
        '''
        Initialize the SCS (Symbol Clock Synchronization) subsystem class.

        :param samples_per_symbol: Int type. Number of samples per symbol.
        :param loop_bandwidth: Float type. Determines the lock-on speed to the timing error (similar to PLL).
        :param damping_factor: Float type. Determines the oscillation during lock-on to the timing error (similar to PLL).
        :param gain: Float type. Gain added to timing error detector output (symbolizes Kp).
        '''
        self.samples_per_symbol = samples_per_symbol
        self.gain = gain
        self.open_loop = open_loop

        self.compute_loop_constants(loop_bandwidth, damping_factor, samples_per_symbol)

        self.delay_register_1 = np.zeros(3, dtype=complex)
        self.delay_register_2 = np.zeros(3, dtype=complex)
        self.interpolation_register = np.zeros(3, dtype=complex)

        self.strobe = None
        self.LFK2_prev = 0
        self.decrementor_prev = 0
        self.mu = 0

        self.ted_output_record = []
        self.loop_filter_output_record = []

    def compute_loop_constants(self, loop_bandwidth, damping_factor, samples_per_symbol):
        """
        Compute the loop filter gains based on the loop bandwidth and damping factor.

        :param loop_bandwidth: Float type. Loop bandwidth of control loop.
        :param damping_factor: Float type. Damping factor of control loop.
        :param samples_per_symbol: Float type. Number of samples per symbol.
        :param kp: Float type. Proportional loop filter gain.
        """
        theta_n = (loop_bandwidth/samples_per_symbol)/(damping_factor + 1/(4*damping_factor))
        factor = (4*theta_n)/(1+2*damping_factor*theta_n+theta_n**2)
        self.k1 = damping_factor * factor/self.gain
        self.k2 = theta_n * factor/self.gain

    def insert_new_sample(self, input_sample):
        """
        Insert a new sample into the SCS system, performing interpolation and updating the timing error.

        :param input_sample: Numpy array. Input samples as complex numbers.
        :return: Complex. The interpolated output sample.
        """
        interpolated_sample = self.farrow_interpolator_parabolic(input_sample)
        
        # timing error detector
        error = self.early_late_ted()

        # loop filter
        filtered_error = self.loop_filter(error)

        # calculate w(n)
        w_n = filtered_error + (1 / self.samples_per_symbol)

        # update mod 1 decrementor
        decrementor = self.decrementor_prev - w_n

        # check mod 1 decrementor
        if decrementor < 0:
            self.strobe = True
            decrementor = decrementor + 1 # mod 1
        else:
            self.strobe = False
        
        # calculate mu
        if self.strobe:
            self.mu = self.decrementor_prev / w_n
    
        # update interpolation register (shift)
        self.interpolation_register = np.roll(self.interpolation_register, -1)
        self.interpolation_register[-1] = interpolated_sample

        # store decrementor value
        self.decrementor_prev = decrementor

        if self.open_loop == False:
            if self.strobe:
                return interpolated_sample
            else:
                return None
        else:
            return filtered_error

    def farrow_interpolator_parabolic(self, input_sample):
        """
        Perform parabolic interpolation on the input signal.

        :param input_sample: Numpy array. The input signal to be interpolated.
        :return: Complex. The interpolated output sample.
        """
        tmp = self.mu
        d1next = -0.5 * input_sample
        d2next = input_sample
    
        v2 = -d1next + self.delay_register_1[2] + self.delay_register_1[1] - self.delay_register_1[0]
        v1 = d1next - self.delay_register_1[2] + self.delay_register_2[1] + self.delay_register_1[1] + self.delay_register_1[0]
        v0 = self.delay_register_2[0]
        output = (((v2 * self.mu) + v1) * self.mu + v0)

        self.delay_register_1 = np.roll(self.delay_register_1, -1)
        self.delay_register_2 = np.roll(self.delay_register_2, -1)
        self.delay_register_1[-1] = d1next
        self.delay_register_2[-1] = d2next

        self.mu = tmp
        return output

    def early_late_ted(self):
        """
        Perform early-late timing error detection.

        :return: Float. The calculated timing error based on early and late samples.
        """
        out = 0
        if self.strobe:
            real_est = (np.real(self.interpolation_register[2]) - np.real(self.interpolation_register[0])) * (-1 if np.real(self.interpolation_register[1]) < 0 else 1)
            imag_est = (np.imag(self.interpolation_register[2]) - np.imag(self.interpolation_register[0])) * (-1 if np.imag(self.interpolation_register[1]) < 0 else 1)
            out = real_est
            self.ted_output_record.append(out)
        return out
    
    def loop_filter(self, phase_error):
        """
        Apply a loop filter to the phase error to compute the filtered output.

        :param phase_error: Float type. The timing phase error to be filtered.
        :return: Float. The filtered output from the loop filter.
        """
        k1 = self.k1
        k2 = self.k2
        LFK2 = k2 * phase_error + self.LFK2_prev
        output = k1 * phase_error + LFK2
        self.LFK2_prev = LFK2
        self.loop_filter_output_record.append(output)
        return output