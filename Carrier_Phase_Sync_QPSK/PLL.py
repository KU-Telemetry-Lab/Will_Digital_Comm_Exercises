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

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None, gain=1, open_loop=False):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        '''
        self.gain = gain
        self.open_loop = open_loop
        self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)

        self.omega = 1
        self.phase = 0
        self.dds_output = np.exp(1j * 0)
        self.sample_rate = sample_rate

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
            internalSignal = np.exp(1j * (2 * np.pi * (self.omega / self.sample_rate) * n + self.phase))
        phaseError = self.phase_detector(internalSignal, incomingSignal)
        V_t = self.loop_filter(phaseError)
        pointOut = self.DDS(n, V_t)
        if self.open_loop == False:
            return pointOut
        else:
            return V_t

    def phase_detector(self, sample_1, sample_2):
        """
        Calculate the phase difference between two samples.

        :param sample1: Complex number. The first sample.
        :param sample2: Complex number. The second sample.
        :return: Float type. The phase difference between the two samples, scaled by Kp.
        """

        angle = np.angle(sample_2) - np.angle(sample_1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle

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
        self.phase += v
        self.dds_output = np.exp(1j * (2 * np.pi * (self.omega / self.sample_rate) * n + self.phase))
        return self.dds_output
    
    def get_current_phase(self):
        """
        Get the current phase of the PLL.

        :return: Float type. The current phase of the PLL.
        """
        return self.phase
