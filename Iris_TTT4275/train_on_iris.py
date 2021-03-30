import numpy as np
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
print("x1", x1all[:, 3])
print("x1", len(x1all))

# ----------- constants
num_classes = 3
num_data = 50
num_attributes = 4

dataset = np.empty((num_classes, num_data, num_attributes))
# print(dataset)
dataset[0] = x1all
dataset[1] = x2all
dataset[2] = x3all

# print(dataset)

training_size = 30
testing_size = 20

training_dataset = dataset[:, :training_size, :]
testing_dataset = dataset[:, -testing_size:, :]

print(np.shape(training_dataset))
print(np.shape(testing_dataset))

# np.empty((num_classes, num_data-testing_size, num_attributes))

# x1 = np.array([x1all[:, 3]])
# x2 = np.array([x2all[:, 3]])
# x3 = np.array([x3all[:, 3]])
