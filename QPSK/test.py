import numpy as np
import matplotlib.pylab as plt

x = np.sqrt(2) * np.linspace(0, 10, 200)
plt.plot(x, np.sin(x))
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
plt.show()