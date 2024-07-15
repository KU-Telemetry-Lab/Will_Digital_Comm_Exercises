import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

y = [1, 2, 1, 0]

spl = CubicSpline(np.arange(0, len(y)), y)

upsample_factor = 10
x_new = np.linspace(0, len(y)-1, num=(len(y)-1) * upsample_factor)

print(len(x_new))

y_new = spl(x_new)

plt.figure()
plt.stem(np.arange(0, len(y)), y, linefmt='C0-', markerfmt='C0o', basefmt='C0-')
plt.stem(x_new, y_new, 'C1-')
plt.title('Cubic Spline Interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
