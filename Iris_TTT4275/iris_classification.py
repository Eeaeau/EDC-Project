from inspect import Attribute
import numpy as np
from numpy.core.defchararray import array
from seaborn.palettes import color_palette
import torch
from numpy.core.fromnumeric import transpose
import scipy.stats as stats
from scipy.io import loadmat
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


with open('Iris_TTT4275/class_1.txt', 'r') as f:
    x1all = np.array([[float(num) for num in line.split(',')] for line in f])

with open('Iris_TTT4275/class_2.txt', 'r') as f:
    x2all = np.array([[float(num) for num in line.split(',')] for line in f])

with open('Iris_TTT4275/class_3.txt', 'r') as f:
    x3all = np.array([[float(num) for num in line.split(',')] for line in f])

# dataset_pd = pd.read_fwf('Iris_TTT4275\iris.data', widths=[50, 50, 50])
# print(dataset_pd)


# ----------- constants
num_classes = 3
num_data = 50
num_attributes = 4

training_size = 30
testing_size = 20

alpha = .1  # step size

classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

attributes = ["sepal length", "sepal width",
              "petal length", "petal width"]


# ------------- functions -------

def linear_descriminat_classifier(x, W):
    return np.matmul(W, x)


def split_dataset(dataset, training_size, testing_size=testing_size):

    training_dataset = dataset[:, :training_size, :]
    testing_dataset = dataset[:, -testing_size:, :]

    return training_dataset, testing_dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def combined_mse(x, W, true_class):
    gradw_mse = np.zeros((num_classes, num_attributes + 1))
    mse = 0

    N = len(x)
    for k in range(N):
        for c in range(num_classes):

            t = true_class[c]

            g_temp = linear_descriminat_classifier(x[c, k], W)

            g_k = sigmoid(g_temp)
            # print(t)

            gradgk_mse = g_k - t
            # print("gradgk_mse:", gradgk_mse)

            gradzk_g = g_k*(1-g_k)  # elementwise
            # print("gradzk_g:", gradzk_g)

            gradw_zk = x[c, k].reshape(1, num_attributes + 1)  # transposes
            # print("gradw_zk: ", gradw_zk)

            temp = (gradgk_mse*gradzk_g).reshape(
                num_classes, 1)  # make two dimentional, such that it can represent a matrix

            # print("temp: ", temp)
            gradw_mse += temp @ gradw_zk  # matrix multi

            mse += np.matmul(gradgk_mse.T, gradgk_mse)

    # print("gradw_mse", gradw_mse)

    return 1/2*mse, gradw_mse


def train_LC(training_dataset, W_track, total_iterations=500, plot_result=False):

    mse_track = []

    true_classes = np.identity(3, dtype=float)

    for i in range(total_iterations):
        mse, gradw_mse = combined_mse(
            training_dataset, W_track[-1], true_classes)

        W_track = np.append(
            W_track, [W_track[-1] - dynamic_alpha(alpha, i, total_iterations) * gradw_mse], axis=0)  # tricking gradw to match with W (extra transpose)

        mse_track.append(mse)

        # if (mse > 2):
        #     print("this class kinda sus:", c)
    print("W: ", W_track[-1])
    if plot_result:
        plt.plot(mse_track)
        plt.show()

    return W_track[-1]


def get_confusion_matrix(W, training_dataset, plot_result=False):
    confusion_matrix = np.zeros((num_classes, num_classes))

    for c in range(num_classes):
        for i in range(testing_size):
            prediction = np.argmax(
                linear_descriminat_classifier(training_dataset[c][i], W))
            confusion_matrix[c, prediction] += 1

    if plot_result:
        sns.heatmap(confusion_matrix,  annot=True, square=True,
                    xticklabels=classes, yticklabels=classes)

        plt.show()

    return confusion_matrix


def dynamic_alpha(initial_alpha, iteration, total_iterations):
    return initial_alpha*np.exp(-iteration/total_iterations)


def calculate_error_rate(confusion_matrix):
    # TODO: add check for non square matrix
    dim = np.size(confusion_matrix, 0)
    num_classifications_class = np.zeros((dim), dtype=int)
    error_rate = np.zeros(dim)

    for row in range(dim):
        for col in range(dim):
            num_classifications_class[row] += confusion_matrix[row, col]

        error_rate[row] = 1 - confusion_matrix[row, row] / \
            num_classifications_class[row]

    return error_rate

    # for i in range(dimention):


def plot_hist():

    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(num_attributes)

    for a in range(num_attributes):
        for c in range(num_classes):
            axs[a].hist(dataset[c, :, a], bins=10)
            axs[a].set_title(classes[c]+" "+attributes[a])
    fig.tight_layout(pad=.2)

    # for c in range(num_classes):
    #     for a in range(num_attributes):
    #         axs[c, a].hist(dataset[c, :, a], bins=10)
    #         axs[c, a].set_title(classes[c]+" "+attributes[a])
    # fig.tight_layout(pad=.2)

    # fig.show()
    #  col="species", row="sex"
    # print(dataset[0, :, 0])
    # plt.hist(dataset[0, :, 0],)
    # sns.displot(dataset[0, :, 0])
    plt.show()

# ----------------------------------------------------------- #
# -------------------------- run ---------------------------- #
# ----------------------------------------------------------- #


dataset = np.empty((num_classes, num_data, num_attributes+1))
# dataset[0] = x1all
# dataset[1] = x2all
# dataset[2] = x3all

dataset[0] = np.array([np.append(data, 1) for data in x1all])
dataset[1] = np.array([np.append(data, 1) for data in x2all])
dataset[2] = np.array([np.append(data, 1) for data in x3all])
# for class in range(3):

# dataset = [np.append(data, 1) for data in dataset]
# dataset = np.append(dataset[:, :], 1)

# df = pd.DataFrame(dataset[0])

# print("df:", df)

training_dataset, testing_dataset = split_dataset(dataset, training_size)

W_track = np.empty([1, num_classes, num_attributes+1])
W_track[0].fill(0)
# print(W_track)


W_track_final = train_LC(training_dataset, W_track, 2000)

print("final W", W_track_final)

confusion_matrix = get_confusion_matrix(W_track_final, testing_dataset)

print("Confusion matrix", confusion_matrix)

error_rate = calculate_error_rate(confusion_matrix)
print("Error rate", error_rate)

plot_hist()
