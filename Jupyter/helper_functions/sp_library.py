import sys
import math
import numpy as np
import matplotlib.pyplot as plt


# FUNCITON DEFINITIONS
################################################################################################
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

def upsample(x, L, offset=0, interpolate_flag=True):
    """
    Discrete signal upsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param L: Int type. Upsample factor.
    :param offset: Int type. Offset size for input array.
    :param interpolate: Boolean type. Flag indicating whether to perform interpolation.
    :return: Numpy array type. Upsampled signal.
    """
    x_upsampled = [0] * offset  # Initialize with offset zeros
    if interpolate_flag:
        x_upsampled.extend(interpolate(x, L))
    else:
        for i, sample in enumerate(x):
            x_upsampled.append(sample)
            x_upsampled.extend([0] * (L - 1))
    return np.array(x_upsampled)

def plot_complex_points(complex_array, constellation):
    """
    Plot complex points on a 2D plane with constellation points labeled.

    :param complex_array: List or numpy array of complex points to plot.
    :param constellation: List of lists, where each inner list contains a complex point and a label.
    """
    # Extract real and imaginary parts of the complex points
    plt.plot([point.real for point in complex_array], [point.imag for point in complex_array], 'ro', label='Received Points')
    
    # Plot constellation points and add labels
    for point, label in constellation:
        plt.plot(point.real, point.imag, 'b+', markersize=10)
        plt.text(point.real, point.imag, f' {label}', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    
    # Label axes
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('Complex Constellation Plot')
    plt.axhline(0, color='gray', lw=0.5, ls='--')
    plt.axvline(0, color='gray', lw=0.5, ls='--')
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()

def srrc(alpha, m, length):
    """
    Generates a square root raised cosine pulse.

    :param alpha: Roll-off or excess factor.
    :param m: Number of symbols per symbol.
    :param length: Length of pulse. Should be k*m+1 where k is an integer.
    :return: List. Square root raised cosine pulse.
    """
    pulse = []
    for n in range(length):
        n_prime = n - np.floor(length/2)
        if n_prime == 0:
            n_prime = sys.float_info.min  # Handle case when n_prime is zero
        if alpha != 0:
            if np.abs(n_prime) == m/(4*alpha):
                n_prime += 0.1e-12
        num = np.sin(np.pi*((1-alpha)*n_prime/m)) + (4*alpha*n_prime/m)*np.cos(np.pi*((1+alpha)*n_prime/m))
        den = (np.pi*n_prime/m)*(1-(4*alpha*n_prime/m)**2)*np.sqrt(m)
        if den == 0:
            pulse.append(1.0)  # Handle division by zero case
        else:
            pulse.append(num/den)
    return pulse

def convolve(x, h, mode='full'):
    """
    Convolution between two sequences. Can return full or same lengths.

    :param x: List or numpy array. Input sequence one.
    :param h: List or numpy array. Input sequence two.
    :param mode: String. Specifies return sequence length.
    :return: Numpy array. Resulting convolution output.
    """
    N = len(x) + len(h) - 1
    x_padded = np.pad(x, (0, N - len(x)), mode='constant')
    h_padded = np.pad(h, (0, N - len(h)), mode='constant')
    X = np.fft.fft(x_padded)
    H = np.fft.fft(h_padded)
    y = np.fft.ifft(X * H)

    if mode == 'same':
        start = (len(h) - 1) // 2
        end = start + len(x)
        y = y[start:end]
    return y

import numpy as np

def modulate_by_exponential(x, f_c, f_s, phase=0, noise=0):
    """
    Modulates a signal by exponential carrier (cos(x) + jsin(x)) and adds AWGN noise.

    :param x: List or numpy array. Input signal to modulate.
    :param f_c: Float. Carrier frequency of the modulation.
    :param f_s: Float. Sampling frequency of the input signal.
    :param phase: Float. Phase of the modulation in radians. Default is 0.
    :param noise: Float. Standard deviation of the AWGN noise to be added. Default is 0 (no noise).
    :return: Numpy array. Modulated signal with optional noise.
    """
    y = []
    for i, value in enumerate(x):
        modulation_factor = np.exp(-1j * 2 * np.pi * f_c * i / f_s + phase)
        y.append(value * modulation_factor)
    y = np.array(y)
    if noise > 0:
        awgn_noise = np.random.normal(0, noise, y.shape) + 1j * np.random.normal(0, noise, y.shape)
        y += awgn_noise
    return y


def downsample(x, l, offset=0):
    """
    Discrete signal downsample implementation.

    :param x: List or Numpy array type. Input signal.
    :param l: Int type. Downsample factor.
    :param offset: Int type. Offset size for input array.
    :return: Numpy array type. Downsampled signal.
    """
    x_downsampled = [0+0j] * offset  # Initialize with offset zeros
    if l > len(x):
        raise ValueError("Downsample rate larger than signal size.")
    # Loop over the signal, downsampling by skipping every l elements
    for i in range(math.floor(len(x) / l)):
        x_downsampled.append(x[i * l])
    
    return np.array(x_downsampled)

def nearest_neighbor(x, constellation = None, binary = True):
    """
    Find the nearest neighbor in a given constellation.

    :param x: Complex number or array of complex numbers. Point(s) to find the nearest neighbor for.
    :param constellation: 2D numpy array containing point-value pairs. List of complex numbers 
           representing the constellation point and its binary value. defaults to BPAM/BPSK
    :return: List of binary values corresponding to the nearest neighbors in the constellation.
    """
    if constellation is None:
        constellation =  [[complex(1+0j), 0b1], [complex(-1+0j), 0b0]]
    output = []
    for input_value in x:
        smallest_distance = float('inf')
        value = None
        for point in constellation:
            distance = np.abs(input_value - point[0])
            if distance < smallest_distance:
                smallest_distance = distance
                if binary:
                    value = point[1]
                else:
                    value = point[0]
        output.append(value)
    return output

def bin_to_char(x):
    """
    Converts a binary array into 7 bit ascii equivalents.

    :param x: List or numpy array type. Input binary signal.
    :return: String containing concatenated ascii characters.
    """
    segmented_arrays = [x[i:i+7] for i in range(0, len(x), 7)]

    bin_chars = []

    for segment in segmented_arrays:
        binary_string = ''.join(str(bit) for bit in segment)
        decimal_value = int(binary_string, 2)
        ascii_char = chr(decimal_value)
        bin_chars.append(ascii_char)

    return ''.join(bin_chars)

def string_to_ascii_binary(string, num_bits=7):
    return ['{:0{width}b}'.format(ord(char), width=num_bits) for char in string]

def error_count(x, y):
    return sum(1 for i in range(len(x)) if x[i] != y[i])

def find_subarray_index(small_array, large_array):
    small_len = len(small_array)
    for i in range(len(large_array) - small_len + 1):
        if large_array[i:i + small_len] == small_array:
            return i
    return -1


# CLASS DEFINITIONS
################################################################################################

class PLL():
    '''
    This class is used to simulate a Phase-Locked Loop (PLL) discretely.
    Components can be called individually or as a whole depending on user needs.
    Use as an object and initialize variables in __init__ if you want full functionality.
    '''
    LFK2prev = 0
    phase = 0
    sigOut = 0

    def __init__(self, sample_rate, loop_bandwidth=None, damping_factor=None, kp=1, k0=1, k1=0, k2=0, wstart=1, thetaStart=0):
        '''
        Initialize the PLL object with the specified parameters.

        :param sample_rate: Float type. The sampling frequency.
        :param loop_bandwidth: Float type, optional. Loop bandwidth. If specified with damping factor, will compute loop filter gains.
        :param damping_factor: Float type, optional. Damping factor. If specified with loop bandwidth, will compute loop filter gains.
        :param kp: Float type. Proportional gain determined by the system.
        :param k0: Float type. DDS gain, usually 1 or -1.
        :param k1: Float type. Loop filter gain feed-forward.
        :param k2: Float type. Loop filter gain feed-back.
        :param wstart: Float type. Starting frequency that the received signal is supposed to be at.
        :param thetaStart: Float type. Initial phase of the signal.

        Initializes the PLL object for repeated use. If left blank, the object will be initialized with default values.
        '''
        if loop_bandwidth is None and damping_factor is None:
            self.Kp = kp
            self.K0 = k0
            self.K1 = k1
            self.K2 = k2
        else:
            self.compute_loop_constants(sample_rate, loop_bandwidth, damping_factor)
        self.w0 = wstart
        self.phase = thetaStart
        self.sigOut = np.exp(1j * thetaStart)
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


# FUNCTIONS DEFINITIONS (NOT SHOWN IN EXAMPLES)
################################################################################################
window_lut= {"rectangular": {"sidelobe amplitude": 10**(-13/10), 
                             "mainlobe width": 4*np.pi, 
                             "approximation error": 10**(-21/10)},
             "bartlett": {"sidelobe amplitude": 10**(-25/10), 
                          "mainlobe width": 8*np.pi, 
                          "approximation error": 10**(-25/10)},
             "hanning": {"sidelobe amplitude": 10**(-31/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-44/10)},
             "hamming": {"sidelobe amplitude": 10**(-41/10), 
                         "mainlobe width": 8*np.pi, 
                         "approximation error": 10**(-53/10)},
             "blackman": {"sidelobe amplitude": 10**(-57/10), 
                          "mainlobe width": 12*np.pi, 
                          "approximation error": 10**(-74/10)}
            }

def apply_window(n, window_type):
    """
    Windowing function used in aid of FIR design flows.

    :param N: Window length (number of coefficients).
    :param window_type: Window type (see below).
    :return w_n: Numpy array type. Calculated window filter coefficients.
    """
    if window_type == "rectangular":
        w_n = np.array([1 for i in range(n)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (n - 1) / 2)) / (n - 1)) for i in range(n)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (n - 1))) for i in range(n)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (n - 1)) for i in range(n)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (n - 1)) + 0.08 * np.cos((4 * np.pi * i) / (n - 1)) for i in range(n)])
    else: #default to 'rectangular'
        w_n = np.array([1 for i in range(n)])
    return w_n

def fir_low_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR low pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation.
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital lowpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else fc / np.pi for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_high_pass(fc, window=None, fp=None, fs=None, ks=10**(-40/10)):
    """
    FIR high pass filter design.

    :param fc: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp: Passband digital frequency cutoff.
    :param fs: Stopband digital frequency cutoff.
    :param ks: Stopband attenuation level.
    :return: Numpy array. Coefficients (numerator) of digital highpass filter.
    """
    if fp is None or fs is None:
        fp = fc - (.125 * fc)
        fs = fc + (.125 * fc)
    if window is None:
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < ks), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / np.abs(fs - fp)) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([-(np.sin(fc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else 1 - (fc / np.pi) for i in range(n)]) # generate filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn * w_n

def fir_band_pass(fc1, fc2, window=None, fs1=None, fp1=None, fp2=None, fs2=None, ks1=10**(-40/10), ks2=10**(-40/10)):
    """
    FIR band pass filter design.

    :param fc1: Digital cutoff frequency one.
    :param fc2: Digital cutoff frequency two.
    :param window: Window used for filter truncation (see dictionary below).
    :param fp1: Passband digital frequency cutoff one.
    :param fs1: Stopband digital frequency cutoff one.
    :param fp2: Passband digital frequency cutoff two.
    :param fs2: Stopband digital frequency cutoff two.
    :param ks1: Stopband attenuation level one.
    :param ks2: Stopband attenuation level two.
    :return: Numpy array. Coefficients (numerator) of digital bandpass filter.
    """
    if fp1 is None or fs1 is None or fp2 is None or fs2 is None:
        fs1 = fc1 + (.125 * fc1)
        fp1 = fc1 - (.125 * fc1)
        fp2 = fc2 - (.125 * fc2)
        fs2 = fc2 + (.125 * fc2)
        try:
            window_type = min((key for key, value in window_lut.items() if value["sidelobe amplitude"] < min(ks1, ks2)), key=lambda k: window_lut[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else:
        window_type = window
    n = math.ceil(window_lut[window_type]["mainlobe width"] / min(np.abs(fs1 - fp1), np.abs(fs2 - fp2))) # calculating filter order
    alpha = (n - 1) / 2
    h_dn = np.array([((np.sin(fc2 * (i - alpha))) / (np.pi * (i - alpha)) - (np.sin(fc1 * (i - alpha))) / (np.pi * (i - alpha))) if i != alpha else (fc2 / np.pi - fc1 / np.pi)  for i in range(n)]) # determining the filter coefficients
    w_n = apply_window(n, window_type)
    return h_dn*w_n
