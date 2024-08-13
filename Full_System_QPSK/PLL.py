import numpy as np

# GOLD STANDARD

class PLL():
    '''
    This class is used to simulate a Phase-Locked Loop (PLL) discretely.
    Components can be called individually or as a whole depending on user needs.
    Use as an object and initialize variables in __init__ if you want full functionality.
    '''
    lfk2_prev = 0
    phase = 0
    sig_out = 0

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None, gain=1, open_loop=False):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        :param gain: Float type. Gain applied to loop filter output.
        :param open_loop: Boolean type. Allows for open loop testing of required system gain for normalizaiton.
        '''
        self.gain = gain
        self.open_loop = open_loop
        self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)

        self.sample_rate = sample_rate
        self.w0 = 1
        self.phase = 0
        self.sig_out = np.exp(1j * self.phase)

    def compute_loop_constants(self, fs, lb, df):
        """
        Compute the loop filter constants based on the given parameters.

        :param fs: Float type. Sampling frequency.
        :param lb: Float type. Loop bandwidth.
        :param df: Float type. Damping factor.
        """
        denominator = 1 + ((2 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) + ((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2
        self.k1 = (1/self.gain) * ((4 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) / denominator
        self.k2 = (1/self.gain) * (((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2) / denominator


    def insert_new_sample(self, incoming_signal, n, internal_signal=None):
        """
        Process a new sample and return the output signal.

        :param incoming_signal: Complex number. The current sample of the received signal.
        :param internal_signal: Complex number, optional. The current signal your LO (local oscillator) is at. Will use default from constructor if left blank.
        :param n: Int type. The current sample index, used to insert a new sample of the received signal and LO.
        :return: Complex number. The output signal after processing.
        """
        if internal_signal is None:
            internal_signal = np.exp(1j * (2 * np.pi * (self.w0 / self.sample_rate) * n + self.phase))
        phase_error = self.phase_detector(internal_signal, incoming_signal)
        v_t = self.loop_filter(phase_error)
        point_out = self.dds(n, v_t)
        if self.open_loop == True:
            return v_t
        else:
            return point_out

    def phase_detector(self, sample1, sample2):
        """
        Calculate the phase difference between two samples.

        :param sample1: Complex number. The first sample.
        :param sample2: Complex number. The second sample.
        :return: Float type. The phase difference between the two samples, scaled by kp.
        """
        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def loop_filter(self, phase_error):
        """
        Apply the loop filter to the phase error.

        :param phase_error: Float type. The phase error.
        :param k1: Float type, optional. Loop filter gain according to Fig C.2.6.
        :param k2: Float type, optional. Loop filter gain according to Fig C.2.6.
        :return: Float type. The output of the loop filter.
        """
        lfk2 = self.k2 * phase_error + self.lfk2_prev
        output = self.k1 * phase_error + lfk2
        self.lfk2_prev = lfk2
        return output

    def dds(self, n, v):
        """
        Direct Digital Synthesis (DDS) implementation.

        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :return: Complex number. The output signal of the DDS.
        """
        self.phase += v
        self.sig_out = np.exp(1j * (2 * np.pi * (self.w0 / self.sample_rate) * n + self.phase))
        return self.sig_out
    
    def get_current_phase(self):
        """
        Get the current phase of the PLL.

        :return: Float type. The current phase of the PLL.
        """
        return self.phase