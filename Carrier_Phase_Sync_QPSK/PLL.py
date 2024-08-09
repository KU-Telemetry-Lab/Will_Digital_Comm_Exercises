import numpy as np

class PLL():
    '''
    This class is used to simulate a Phase-Locked Loop (PLL) discretely.
    Components can be called individually or as a whole depending on user needs.
    Use as an object and initialize variables in __init__ if you want full functionality.
    '''
    LFK2_prev = 0
    phase = 0
    dds_output = 0

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None, gain=1, w0=1):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        '''
        self.w = 1
        self.phase = 0
        self.dds_output = np.exp(1j * 0)
        self.sample_rate = sample_rate

        self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)
        self.gain = gain

    def compute_loop_constants(self, fs, lb, df):
        """
        Compute the loop filter constants based on the given parameters.

        :param fs: Float type. Sampling frequency.
        :param lb: Float type. Loop bandwidth.
        :param df: Float type. Damping factor.

        Calculates and sets the loop filter gains K0, K1, and K2.
        """
        denominator = 1 + ((2 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) + ((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2
        self.k1 = ((4 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) / denominator
        self.k2 = (((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2) / denominator
        self.k0 = 1

    def insert_new_sample(self, incomingSignal, n, internalSignal=None):
        """
        Process a new sample and return the output signal.

        :param incomingSignal: Complex number. The current sample of the received signal.
        :param internalSignal: Complex number, optional. The current signal your LO (local oscillator) is at. Will use default from constructor if left blank.
        :param n: Int type. The current sample index, used to insert a new sample of the received signal and LO.

        :return: Complex number. The output signal after processing.

        If using as an object, this is the index of the only function you need to call to achieve PLL functionality.
        """
        if internalSignal is None:
            internalSignal = np.exp(1j * (2 * np.pi * (self.w / self.sample_rate) * n + self.phase))
        phaseError = self.phase_detector(internalSignal, incomingSignal)
        V_t = self.loop_filter(phaseError)
        pointOut = self.DDS(n, V_t)
        return pointOut

    def phase_detector(self, sample_1, sample_2):
        """
        Calculate the phase difference between two samples.

        :param sample1: Complex number. The first sample.
        :param sample2: Complex number. The second sample.
        :return: Float type. The phase difference between the two samples, scaled by Kp.
        """

        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi-
        return angle * self.gain

    def loop_filter(self, phase_error):
        """
        Apply the loop filter to the phase error.

        :param phase_error: Float type. The phase error.
        :return: Float type. The output of the loop filter.

        Updates internal state with the new value of LFK2.
        """
        LFK2 = self.k2 * phase_error + self.LFK2_prev
        output = self.k1 * phase_error + LFK2
        self.LFK2_prev = LFK2
        return output

    def DDS(self, n, v):
        """
        Direct Digital Synthesis (DDS) implementation.

        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :return: Complex number. The output signal of the DDS.

        Updates internal phase and returns the synthesized signal.
        """
        self.phase += v * self.k0
        self.dds_output = np.exp(1j * (2 * np.pi * (self.w0 / fs) * n + self.phase))
        return self.dds_output
    
    def get_current_phase(self):
        """
        Get the current phase of the PLL.

        :return: Float type. The current phase of the PLL.
        """
        return self.phase
