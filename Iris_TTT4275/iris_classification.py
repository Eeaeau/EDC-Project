import numpy as np
from numpy.core.defchararray import array
import torch
# from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat


with open('Iris_TTT4275\class_1.txt', 'r') as f:
    x1all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x1all: ", x1all)

with open('Iris_TTT4275\class_2.txt', 'r') as f:
    x2all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x2all: ", x2all)

with open('Iris_TTT4275\class_3.txt', 'r') as f:
    x3all = np.array([[float(num) for num in line.split(',')] for line in f])
print("x3all: ", x3all)

# x1 = [x1all(:, 4) x1all(:, 1) x1all(:, 2)]
# x2 = [x2all(:, 4) x2all(:, 1) x2all(:, 2)]
# x3 = [x3all(:, 4) x3all(:, 1) x3all(:, 2)]

# x1 = [x1all(:, 3) x1all(:, 4)]
# x2 = [x2all(:, 3) x2all(:, 4)]
# x3 = [x3all(:, 3) x3all(:, 4)]
print("x1 len", len(x1all))

# ----------- constants
num_classes = 3
num_data = 50
num_attributes = 4

training_size = 30
testing_size = 20

alpha = .01  # step size


# ------------- functions -------

def linear_descriminat_classifier(x, W, wo):
    return W*x+wo


def MSE(g, t, N):
    mse = 0
    for k in range(N):
        mse += np.transpose(g[k] - t[k])*(g[k] - t[k])
    return .5*mse


def w_gradiant(x, g, t):
    gradw_mse = 0
    N = len(g)
    for k in range(N):
        gradg_mse = g[k] - t[k]
        gradz_g = np.multiply(g[k], 1-g[k])
        gradw_z = np.transpose(x[k])

        gradw_mse += gradg_mse*gradz_g*gradw_z
        # mse_grad_w += np.gradient(mse, g[k])*np.gradient(g[k],
        #                                                W*x[k]) * np.gradient(W*x[k], W)
    return gradw_mse

# ------------- run -------------


dataset = np.empty((num_classes, num_data, num_attributes))

dataset[0] = x1all
dataset[1] = x2all
dataset[2] = x3all

training_dataset = dataset[:, :training_size, :]
testing_dataset = dataset[:, -testing_size:, :]

print("training dataset: ", (training_dataset))
print(np.shape(testing_dataset))

# x = np.empty((num_attributes, num_data))
x = training_dataset

x[:, :] = np.linspace(0, 10, num_data)

W = np.ones([num_classes, num_attributes])

t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
t = np.array([num_data, num_classes])
t = [[1, 0, 0], [0, 0, 1], [0, 1, 0] """..."""]
training_dataset[1]

g = np.empty(num_classes, num_classes)

for c in range(num_classes):

    g[c] = linear_descriminat_classifier(x, W[c], .5)

    if (0 == c):
        np.append(W, W[c] - alpha*w_gradiant(x, g, np.array([1, 0, 0])))
    elif (1 == c):
        np.append(W, W[c] - alpha*w_gradiant(x, g, np.array([0, 1, 0])))
    else:
        np.append(W, W[c] - alpha*w_gradiant(x, g, np.array([0, 0, 1])))
