
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import examples as ex
from KUSignalLib import MatLab as ml
from KUSignalLib import DSP as dsp

class PLL():
    '''
    This class is used to simulate a Phase-Locked Loop (PLL) discretely.
    Components can be called individually or as a whole depending on user needs.
    Use as an object and initialize variables in __init__ if you want full functionality.
    '''
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        '''
        self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)
        self.w0 = 1
        self.phase = 0
        self.sigOut = np.exp(1j * 0)
        self.sample_rate = sample_rate

    def compute_loop_constants(self, fs, lb, df):
        """
        Compute the loop filter constants based on the given parameters.

        :param fs: Float type. Sampling frequency.
        :param lb: Float type. Loop bandwidth.
        :param df: Float type. Damping factor.

        Calculates and sets the loop filter gains K0, K1, and K2.
        """
        denominator = 1 + ((2 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) + ((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2
        self.K1 = ((4 * df) * ((lb * (1 / fs)) / (df + (1 / (4 * df))))) / denominator
        self.K2 = (((lb * (1 / fs)) / (df + (1 / (4 * df)))) ** 2) / denominator
        self.K0 = 1
        self.Kp = 1

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
            internalSignal = np.exp(1j * (2 * np.pi * (self.w0 / self.sample_rate) * n + self.phase))
        phaseError = self.phase_detector(internalSignal, incomingSignal)
        V_t = self.loop_filter(phaseError)
        pointOut = self.DDS(n, V_t)
        return pointOut

    def phase_detector(self, sample1, sample2, Kp=None):
        """
        Calculate the phase difference between two samples.

        :param sample1: Complex number. The first sample.
        :param sample2: Complex number. The second sample.
        :param Kp: Float type, optional. Proportional gain, should be less than one.

        :return: Float type. The phase difference between the two samples, scaled by Kp.

        For BPSK (Binary Phase Shift Keying), take the sign of the real part and multiply by the imaginary part.
        """
        if Kp is None:
            Kp = self.Kp
        angle = np.angle(sample2) - np.angle(sample1)
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi
        return angle * Kp

    def loop_filter(self, phaseError, K1=None, K2=None):
        """
        Apply the loop filter to the phase error.

        :param phaseError: Float type. The phase error.
        :param K1: Float type, optional. Loop filter gain according to Fig C.2.6.
        :param K2: Float type, optional. Loop filter gain according to Fig C.2.6.

        :return: Float type. The output of the loop filter.

        Updates internal state with the new value of LFK2.
        """
        if K1 is None:
            K1 = self.K1
        if K2 is None:
            K2 = self.K2
        LFK2 = K2 * phaseError + self.LFK2prev
        output = K1 * phaseError + LFK2
        self.LFK2prev = LFK2
        return output

    def DDS(self, n, v, k0=None, w0=None, fs=None):
        """
        Direct Digital Synthesis (DDS) implementation.

        :param n: Int type. The current sample index.
        :param v: Float type. The output of the loop filter.
        :param k0: Float type, optional. DDS gain.
        :param w0: Float type, optional. Starting frequency that the received signal is supposed to be at.
        :param fs: Float type, optional. Sampling frequency.

        :return: Complex number. The output signal of the DDS.

        Updates internal phase and returns the synthesized signal.
        """
        if k0 is None:
            k0 = self.K0
        if w0 is None:
            w0 = self.w0
        if fs is None:
            fs = self.sample_rate
        self.phase += v * k0
        self.sigOut = np.exp(1j * (2 * np.pi * (w0 / fs) * n + self.phase))
        return self.sigOut
    
    def get_current_phase(self):
        """
        Get the current phase of the PLL.

        :return: Float type. The current phase of the PLL.
        """
        return self.phase

# SYSTEM PARAMETERS
################################################################################################### 
fs = 500 

sig_freq = 10
sig_phase = np.pi / 4
n = np.arange(0,1000)
input_signal = np.exp(1j * ((2 * np.pi * (sig_freq) / fs) * n + (sig_phase)))


# Initialize lists to record simulation results
pll_input = []
pll_output = []
pll_detected_phase_record = []
pll_error_record = []

# PLL INSTANTIATION
################################################################################################### 
loop_bandwidth = 0.02 * fs
damping_factor = 1 / np.sqrt(2)
pll = PLL(fs, loop_bandwidth=loop_bandwidth, damping_factor=damping_factor)

# Print loop filter gains for reference
print(f"Kp: {pll.Kp}")
print(f"K0: {pll.K0}")
print(f"K1: {pll.K1}")
print(f"K2: {pll.K2}")

# PLL SIMULATION
################################################################################################### 
for i in range(len(n)):
    # Insert the new sample into the PLL
    output_signal = pll.insert_new_sample(input_signal[i], i)
    
    # Record detected phase and error
    detected_phase = pll.get_current_phase()
    error = pll.phase_detector(output_signal, input_signal[i])

    # Update records
    pll_input.append(input_signal[i])
    pll_output.append(output_signal)
    pll_detected_phase_record.append(detected_phase)
    pll_error_record.append(error)

# PLOTTING RESULTS
################################################################################################### 
plt.figure(figsize=(12, 8))

# Subplot for Phase Error
plt.subplot(2, 1, 1)
plt.plot(pll_error_record, label='Phase Error', color='r')
plt.title('Phase Error')
plt.xlabel('Sample Index')
plt.ylabel('Phase Error (radians)')
plt.grid()

# Subplot for Input and Output Signals
plt.subplot(2, 1, 2)
plt.plot(np.real(pll_input), label='Input Signal', color='b', alpha=0.7)
plt.plot(np.real(pll_output), label='Output Signal', color='g', alpha=0.7)
plt.title('Input and Output Signals')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()