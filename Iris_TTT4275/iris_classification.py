import numpy as np
from numpy.core.defchararray import array
import torch
# from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat


with open('Iris_TTT4275\class_1.txt', 'r') as f:
    x1all = np.array([[float(num) for num in line.split(',')] for line in f])
    # x1all = np.append(x1all[:], np.array([1, 0, 0]))
print("x1all: ", x1all)

with open('Iris_TTT4275\class_2.txt', 'r') as f:
    x2all = np.array([[float(num) for num in line.split(',')] for line in f])
    # np.append(x2all[:], np.array([0, 1, 0]))
print("x2all: ", x2all)

with open('Iris_TTT4275\class_3.txt', 'r') as f:
    x3all = np.array([[float(num) for num in line.split(',')] for line in f])
    # x3all = np.append(x3all[:], np.array([0, 0, 1]))
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
    print("x", x)
    print("w:", W)
    return np.matmul(W, x)+wo


# def MSE(g, t, N):
#     mse = 0
#     for k in range(N):
#         mse += np.transpose(g[k] - t[k])*(g[k] - t[k])
#     return .5*mse


def split_dataset(dataset, training_size, testing_size=testing_size):

    training_dataset = dataset[:, :training_size, :]
    testing_dataset = dataset[:, -testing_size:, :]

    return training_dataset, testing_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def combined_mse(x, t):
    # gradw_mse = np.zeros(num_classes)
    gradw_mse = 0
    mse = 0

    N = len(x)

    for k in range(N):

        g = linear_descriminat_classifier(x[k], W[-1], 0)

        g = sigmoid(g)

        print(g)
        print(t)

        gradgk_mse = g - t
        print("gradgk_mse:", gradgk_mse)

        gradzk_g = np.multiply(g, 1-g)
        print("gradzk_g:", gradzk_g)

        gradw_zk = np.transpose(x[k])
        print("gradw_zk: ", gradw_zk)

        # gradw_mse += gradgk_mse*gradzk_g*gradw_zk
        # gradw_mse += np.matmul(np.multiply(gradgk_mse, gradzk_g), gradw_zk)
        # should be be 3x5 matrix 
        gradw_mse += np.matmul(gradgk_mse, gradzk_g)

        mse += np.matmul(gradgk_mse.T, gradgk_mse)

    print("gradw_mse", gradw_mse)

    return 1/2*mse, gradw_mse


def train_LC(training_dataset, W, iterations=500):
    for i in range(iterations):
        for c in range(num_classes):
            # for k in range(training_size):
            # g = linear_descriminat_classifier(training_dataset[c], W[-1], 0)

            # g = sigmoid(g)

            # print("g: ", g)

            if (0 == c):
                mse, gradw_mse = combined_mse(
                    training_dataset[c], np.array([1, 0, 0]))
                np.append(
                    W, W[-1] - alpha * gradw_mse)
            elif (1 == c):
                mse, gradw_mse = combined_mse(
                    training_dataset[c], np.array([0, 1, 0]))
                np.append(
                    W, W[-1] - alpha * gradw_mse)
            else:
                mse, grad_W_MSE = combined_mse(
                    training_dataset[c], np.array([0, 0, 1]))
                np.append(
                    W, W[-1] - alpha * grad_W_MSE)

# ------------- run -------------


dataset = np.empty((num_classes, num_data, num_attributes))

dataset[0] = x1all
dataset[1] = x2all
dataset[2] = x3all


training_dataset, testing_dataset = split_dataset(dataset, training_size)

print("training dataset: ", (training_dataset))
print(np.shape(testing_dataset))

# x = np.empty((num_attributes, num_data))

W = np.ones([1, num_classes, num_attributes])

print("W: ", W)

# t = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# t = np.array([num_data, num_classes])
# t = [[1, 0, 0], [0, 0, 1], [0, 1, 0] """..."""]

# g = np.empty([num_classes])

# setter input data til x


train_LC(training_dataset, W, 1)
