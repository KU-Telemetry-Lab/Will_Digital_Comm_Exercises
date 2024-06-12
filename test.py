import numpy as np
import math
from matplotlib import pyplot as plt

# windowing look up table used in aid of FIR design flows
WindowLUT = {"rectangular": {"sidelobe amplitude": 10**(-13/10), "mainlobe width": 4*np.pi, "approximation error": 10**(-21/10)},
             "bartlett": {"sidelobe amplitude": 10**(-25/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-25/10)},
             "hanning": {"sidelobe amplitude": 10**(-31/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-44/10)},
             "hamming": {"sidelobe amplitude": 10**(-41/10), "mainlobe width": 8*np.pi, "approximation error": 10**(-53/10)},
             "blackman": {"sidelobe amplitude": 10**(-57/10), "mainlobe width": 12*np.pi, "approximation error": 10**(-74/10)}}

def Window(N, window_type):
    """
    Windowing function used in aid of FIR design flows.

    :param N: Window length (number of coefficients).
    :param window_type: Window type (see below).
    :return w_n: Numpy array type. Calculated window filter coefficients.
    """
    if window_type == "rectangular":
        w_n = np.array([1 for i in range(N)])
    elif window_type == "bartlett":
        w_n = np.array([(1 - (2 * np.abs(i - (N - 1) / 2)) / (N - 1)) for i in range(N)])
    elif window_type == "hanning":
        w_n = np.array([0.5 * (1 - np.cos((2 * np.pi * i) / (N - 1))) for i in range(N)])
    elif window_type == "hamming":
        w_n = np.array([0.54 - 0.46 * np.cos((2 * np.pi * i) / (N - 1)) for i in range(N)])
    elif window_type == "blackman":
        w_n = np.array([0.42 - 0.5 * np.cos((2 * np.pi * i) / (N - 1)) + 0.08 * np.cos((4 * np.pi * i) / (N - 1)) for i in range(N)])
    else: #default to 'rectangular'
        w_n = np.array([1 for i in range(N)])
    return w_n

def FIRLowPass(cutoff_frequency, window=None, **kwargs):
    """
    FIR low pass filter design.

    Generic design parameters.

    :param cutoff_frequency: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).

    Detailed design parameters (optional).

    :param passband_cutoff: Passband digital frequency cutoff.
    :param stopband_cutoff: Stopband digital frequency cutoff.
    :param passband_attenuation: Passband attenuation level.
    :param stopband_attenuation: Stopband attenuation level.

    :return: Numpy array type. Coefficients (numerator) of digital lowpass filter.
    """
    if 'passband_cutoff' in kwargs and 'stopband_cutoff' in kwargs:
        wp = kwargs['passband_cutoff']
        ws = kwargs['stopband_cutoff']
        if 'passband_attenuation' in kwargs and 'stopband_attenuation' in kwargs:
            kp = kwargs['passband_attenuation']
            ks = kwargs['stopband_attenuation']
        else: # using standard attenuation levels
            kp = 10**(-3/10)
            ks = 10**(-40/10)
        try:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else: # using standard transition width
        wp = cutoff_frequency - (.125 * cutoff_frequency)
        ws = cutoff_frequency + (.125 * cutoff_frequency)
        kp = 10**(-3/10)
        ks = 10**(-40/10)
        if window == None:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        else:
            window_type=window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(WindowLUT[window_type]["mainlobe width"] / np.abs(ws - wp))
    wc = (wp + ws) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([(np.sin(wc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else wc / np.pi for i in range(N)])
    w_n = Window(N, window_type)

    return h_dn*w_n


def FIRHighPass(cutoff_frequency, window=None, **kwargs):
    """
    FIR high pass filter design.

    Generic design parameters.

    :param cutoff_frequency: Digital cutoff frequency.
    :param window: Window used for filter truncation (see dictionary below).

    Detailed design parameters (optional).

    :param stopband_cutoff: Stopband digital frequency cutoff.
    :param passband_cutoff: Passband digital frequency cutoff.
    :param stopband_attenuation: Stopband attenuation level.
    :param passband_attenuation: Passband attenuation level.

    :return: Numpy array type. Coefficients (numerator) of digital highpass filter.
    """
    if 'stopband_cutoff' in kwargs and 'passband_cutoff' in kwargs:
        ws = kwargs['stopband_cutoff']
        wp = kwargs['passband_cutoff']
        if 'stopband_attenuation' in kwargs and 'passband_attenuation' in kwargs:
            ks = kwargs['stopband_attenuation']
            kp = kwargs['passband_attenuation']
        else: # using standard attenuation levels
            ks = 10**(-40/10)
            kp = 10**(-3/10)
        try:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else: # using standard transition width
        ws = cutoff_frequency - (.125 * cutoff_frequency)
        wp = cutoff_frequency + (.125 * cutoff_frequency)
        ks = 10**(-40/10)
        kp = 10**(-3/10)
        if window == None:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        else:
            window_type=window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(WindowLUT[window_type]["mainlobe width"] / np.abs(ws - wp))
    wc = (wp + ws) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([-(np.sin(wc * (i - alpha))) / (np.pi * (i - alpha)) if i != alpha else 1 - (wc / np.pi) for i in range(N)])
    w_n = Window(N, window_type)

    return h_dn*w_n


def FIRBandPass(cutoff_frequency_one, cutoff_frequency_two, window=None, **kwargs):
    """
    FIR band pass filter design.

    Generic design parameters.

    :param cutoff_frequency_one: Digital cutoff frequency one.
    :param cutoff_frequency_two: Digital cutoff frequency two.
    :param window: Window used for filter truncation (see dictionary below).

    Detailed design parameters (optional).

    :param stopband_cutoff_one: Stopband digital frequency cutoff one.
    :param passband_cutoff_one: Passband digital frequency cutoff one.
    :param stopband_attenuation_one: Stopband attenuation level one.
    :param passband_attenuation_one: Passband attenuation level one.
    :param passband_cutoff_two: Passband digital frequency cutoff two.
    :param stopband_cutoff_two: Stopband digital frequency cutoff two.
    :param passband_attenuation_two: Passband attenuation level two.
    :param stopband_attenuation_two: Stopband attenuation level two.


    :return: Numpy array type. Coefficients (numerator) of digital lowpass filter.
    """
    if 'passband_cutoff_one' in kwargs and 'stopband_cutoff_one' in kwargs and 'passband_cutoff_two' in kwargs and 'stopband_cutoff_two' in kwargs:
        ws1 = kwargs['stopband_cutoff_one']
        wp1 = kwargs['passband_cutoff_one']
        wp2 = kwargs['passband_cutoff_two']
        ws2 = kwargs['stopband_cutoff_two']
        if 'stopband_attenuation_one' in kwargs and 'passband_attenuation_one' in kwargs and 'passband_attenuation_two' in kwargs and 'stopband_attenuation_two' in kwargs:
            ks1 = kwargs['stopband_attenuation_one']
            kp1 = kwargs['passband_attenuation_one']
            kp2 = kwargs['passband_attenuation_two']
            ks2 = kwargs['stopband_attenuation_two']
        else: # using standard attenuation levels
            ks1 = ks2 = 10**(-40/10)
            kp1 = kp2 = 10**(-3/10)
        try:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        except ValueError:
            window_type = "blackman"
    else: # using standard transition width
        ws1 = cutoff_frequency_one - (.125 * cutoff_frequency_one)
        wp1 = cutoff_frequency_one + (.125 * cutoff_frequency_one)
        wp2 = cutoff_frequency_two - (.125 * cutoff_frequency_two)
        ws2 = cutoff_frequency_two + (.125 * cutoff_frequency_two)
        ks1 = ks2 = 10**(-40/10)
        kp1 = kp2 = 10**(-3/10)
        if window == None:
            window_type = min((key for key, value in WindowLUT.items() if value["sidelobe amplitude"] < ks), key=lambda k: WindowLUT[k]["sidelobe amplitude"])
        else:
            window_type=window
    # calculating filter shifted and truncated filter parameters
    N = math.ceil(WindowLUT[window_type]["mainlobe width"] / min(np.abs(ws1 - wp1), np.abs(ws2 - wp2)))
    wc1 = (ws1 + wp1) / 2
    wc2 = (wp2 + ws2) / 2
    alpha = (N - 1) / 2

    # determining delayed filter and window coefficients
    h_dn = np.array([((np.sin(wc2 * (i - alpha))) / (np.pi * (i - alpha)) - (np.sin(wc1 * (i - alpha))) / (np.pi * (i - alpha))) if i != alpha else (wc2 / np.pi - wc1 / np.pi)  for i in range(N)])
    w_n = Window(N, window_type)

    return h_dn*w_n


# Define cutoff frequencies
wc1 = 0.3*np.pi
wc2 = 0.4*np.pi

# Generate filter coefficients using Blackman window
coefficients = FIRBandPass(wc1, wc2, window="blackman")
coefficients = FIRLowPass(wc1, window="blackman")
coefficients = FIRHighPass(wc1, window="hanning")

# Plotting the impulse response of the filter
plt.figure(figsize=(10, 5))
plt.stem(coefficients)
plt.title('Impulse Response of FIR Bandpass Filter')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Compute the frequency response using Fourier transform
frequency_response = np.fft.fft(coefficients)

# Plotting the frequency response of the filter
plt.figure(figsize=(10, 5))
plt.plot(np.abs(frequency_response))
plt.title('Frequency Response of FIR Bandpass Filter')
plt.xlabel('Frequency (Normalized)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()