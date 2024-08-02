import numpy as np
import math
from scipy import interpolate as intp, signal as sig
import matplotlib.pyplot as plt
import sys

def Interpolate(x, n, mode="linear"):
    nonzero_indices = np.arange(0, len(x)*n, n) # Generate indices for upsampled signal
    interpolation_function = intp.interp1d(nonzero_indices, np.array(x), kind=mode, fill_value='extrapolate') # create interpolation function
    interpolated_signal = interpolation_function(np.arange(len(x)*n)) # interpolate the signal
    return interpolated_signal

def Upsample(x, L, offset=0, interpolate=True):
    x_upsampled = [0 for i in range(offset)]  # Initialize a list to store the upsampled signal (add offset if needed)
    if interpolate:
        x_upsampled = Interpolate(x, L, mode="linear")
    else:
        for i in range(len(x)):  # Iterate over each element in the input signal
            x_upsampled += [x[i]] + list(np.zeros(L-1, dtype=type(x[0])))  # Add the current element and L zeros after each element
    return x_upsampled

def DirectForm2(b, a, x):
    n = len(b)
    m = len(a)
    if n > m:
        maxLen = n
        a = np.concatenate((a, np.zeros(n - m)))
    else:
        maxLen = m
        b = np.concatenate((b, np.zeros(m - n)))
    denominator = a.copy()
    denominator[1:] = -denominator[1:] #flip sign of denominator coefficients
    denominator[0] = 0 #zero out curent p(0) value for multiply, will add this coeff. back in for new x[n] term
    x = np.concatenate((x, np.zeros(maxLen - 1))) #zero pad x
    y = np.zeros(len(x), dtype=complex)
    delayLine = np.zeros(maxLen, dtype=complex)
    for i in range(len(x)):
        y[i] = np.dot(b, delayLine) #df2 right side
        tmp = np.dot(denominator, delayLine) #df2 left side
        delayLine[1:] = delayLine[:-1] #shift delay line
        delayLine[0] = x[i]*a[0] + tmp #new value is x[n] * a[0] + sum of left side
    return y[1:]

def convolve(x, h, mode='full'):
    if mode not in ['full', 'same']:
        raise ValueError("Mode must be either 'full' or 'same'")
    
    N = len(x) + len(h) - 1 if mode == 'full' else max(len(x), len(h))
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
   
def SRRC(alpha, N, length):
    pulse = []
    for n in range(length):
        n = n - np.floor(length/2)
        if n == 0: # evaluate at limit
            n=sys.float_info.min
        if alpha !=0:# evaluate at limit
            if ((n == N/(4*alpha) or n == -N/(4*alpha))):
                n = n + 0.1e-12
        num = np.sin(np.pi*((1-alpha)*n/N)) + (4*alpha*n/N)*np.cos(np.pi*((1+alpha)*n/N))
        den = (np.pi*n/N)*(1-(4*alpha*n/N)**2)*np.sqrt(N)
        pulse.append(num/den)
    return pulse


test_input_1 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
upsample_factor = 4
test_input_1 = np.array(Upsample(test_input_1, upsample_factor, interpolate=False))

plt.figure()
plt.stem(test_input_1)
plt.title('Upsampled Signal')
plt.xlabel('Symbol Index')
plt.ylabel('Amplitude')
plt.tight_layout()

length_values = [5]
alpha_values = [.25]

figures = []

for length in length_values:
    for alpha in alpha_values:
        pulse_shape = np.roll(np.array(SRRC(alpha, upsample_factor, length)), -1)
        result = convolve(test_input_1, pulse_shape, mode="same")
        # result = convolve(result, np.conjugate(pulse_shape[::-1]))

        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.stem(pulse_shape)
        plt.title(f'Pulse Shape w/ Alpha = {alpha} and Length = {length}')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')
        plt.subplot(2, 1, 2)
        plt.stem(np.real(result))
        plt.title('Pulse Shaped Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Amplitude')

        plt.tight_layout()

        # Append the figure to the list
        figures.append(fig)

# Show all figures at once
plt.show()