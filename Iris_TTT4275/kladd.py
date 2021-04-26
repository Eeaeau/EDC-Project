import numpy as np
from numpy.core.defchararray import array
import torch
# from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat


x = np.empty((1, 5))
x[0] = np.arange(1, 6)
y = np.empty((1, 3))
y[0] = np.arange(1, 4)

print(x.T)
print(y)

z = np.matmul(x.T, y)

print(z)
print(z.T)


# xy = np.ones(3, 3)
# xy_transposed = xy

print(np.array([2, 'rer']))