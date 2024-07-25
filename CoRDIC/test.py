import math
import sys
sys.path.insert(0, '../KUSignalLib/src')
from KUSignalLib import DSP

x = 90
cos_x, sin_x = DSP.cordic(math.radians(x))
print(cos_x)
print(sin_x)