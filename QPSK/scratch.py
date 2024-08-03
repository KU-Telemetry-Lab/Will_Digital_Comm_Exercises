import numpy as np
import matplotlib.pyplot as plt

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

# Example usage
qpsk_constellation = [
    [complex(np.sqrt(1), np.sqrt(1)), 3], 
    [complex(np.sqrt(1), -np.sqrt(1)), 2], 
    [complex(-np.sqrt(1), -np.sqrt(1)), 0], 
    [complex(-np.sqrt(1), np.sqrt(1)), 1]
]

# Example complex array of received points
received_points = np.array([complex(0.5, 0.5), complex(-0.5, -0.5), complex(0.5, -0.5)])

# Plot the complex points
plot_complex_points(received_points, qpsk_constellation)
