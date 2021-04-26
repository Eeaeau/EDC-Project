import numpy as np
from numpy.core.defchararray import array
import torch
# from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('Iris_TTT4275\class_1.txt', 'r') as f:
    x1all = np.array([[float(num) for num in line.split(',')] for line in f])
    # x1all = np.append(x1all[:], np.array([1, 0, 0]))
# print("x1all: ", x1all)

with open('Iris_TTT4275\class_2.txt', 'r') as f:
    x2all = np.array([[float(num) for num in line.split(',')] for line in f])
    # np.append(x2all[:], np.array([0, 1, 0]))
# print("x2all: ", x2all)

with open('Iris_TTT4275\class_3.txt', 'r') as f:
    x3all = np.array([[float(num) for num in line.split(',')] for line in f])
    # x3all = np.append(x3all[:], np.array([0, 0, 1]))
# print("x3all: ", x3all)

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

alpha = .05  # step size


# ------------- functions -------

def linear_descriminat_classifier(x, W):
    return np.matmul(W, x)


def split_dataset(dataset, training_size, testing_size=testing_size):

    training_dataset = dataset[:, :training_size, :]
    testing_dataset = dataset[:, -testing_size:, :]

    return training_dataset, testing_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def combined_mse(x, t, W):
    # gradw_mse = np.zeros(num_classes)
    gradw_mse = 0
    mse = 0

    N = len(x)

    for k in range(N):

        g_temp = linear_descriminat_classifier(x[k], W)

        g_k = sigmoid(g_temp)

        print('g: ', g_k)
        print(t)

        gradgk_mse = g_k - t
        print("gradgk_mse:", gradgk_mse)

        gradzk_g = np.multiply(g_k, 1-g_k)
        # print("gradzk_g:", gradzk_g)

        gradw_zk = x[k].reshape(num_attributes + 1, 1)  # transposes
        # print("gradw_zk: ", gradw_zk)

        temp = np.multiply(gradgk_mse, gradzk_g).reshape(
            1, num_classes)  # make two dimentional, such that it can represent a matrix

        gradw_mse += gradw_zk @ temp

        mse += np.matmul(gradgk_mse.T, gradgk_mse)

    # print("gradw_mse", gradw_mse)

    return 1/2*mse, gradw_mse


def train_LC(training_dataset, W_track, iterations=500, plot_result=False):
    mse_track = []
    for i in range(iterations):
        for c in range(num_classes):

            if (0 == c):
                mse, gradw_mse = combined_mse(
                    training_dataset[c], np.array([1, 0, 0]), W_track[-1])
            elif(1 == c):
                mse, gradw_mse = combined_mse(
                    training_dataset[c], np.array([0, 1, 0]), W_track[-1])
            else:
                mse, gradw_mse = combined_mse(
                    training_dataset[c], np.array([0, 0, 1]), W_track[-1])

            W_track = np.append(
                W_track, [W_track[-1] - alpha * gradw_mse.T], axis=0)  # tricking gradw to match with W (extra transpose)
            mse_track.append(mse)

            # if (mse > 2):
            #     print("this class kinda sus:", c)
    print("W: ", W_track[-1])

    if plot_result:
        plt.plot(mse_track)
        plt.show()


def get_confusion_matrix(W, training_dataset):
    confusion_matrix = np.zeros((num_classes, num_classes))

    for c in range(num_classes):
        for i in range(testing_size):
            prediction = np.argmax(
                linear_descriminat_classifier(training_dataset[c][i], W))
            confusion_matrix[c, prediction] += 1

    return confusion_matrix

# ------------- run -------------


dataset = np.empty((num_classes, num_data, num_attributes+1))
# dataset[0] = x1all
# dataset[1] = x2all
# dataset[2] = x3all


dataset[0] = np.array([np.append(data, 1) for data in x1all])
dataset[0] = np.array([np.append(data, 1) for data in x1all])
dataset[1] = np.array([np.append(data, 1) for data in x2all])
dataset[2] = np.array([np.append(data, 1) for data in x3all])
# for class in range(3):

# dataset = [np.append(data, 1) for data in dataset]
# dataset = np.append(dataset[:, :], 1)

print("element:", dataset)

training_dataset, testing_dataset = split_dataset(dataset, training_size)

W_track = np.empty([1, num_classes, num_attributes+1])
W_track[0].fill(0)
# print(W_track)


train_LC(training_dataset, W_track, 3)

# print(testing_dataset)


confusion_matrix = get_confusion_matrix(W_track, training_dataset)

print(confusion_matrix)
