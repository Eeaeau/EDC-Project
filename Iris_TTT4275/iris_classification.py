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

alpha = .01  # step size

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


def train_LC(training_dataset, W_track, alpha=alpha, grad_mse_threshold=0.001, max_iterations=500, use_dynamic_alpha=True, plot_result=False):

    mse_track = []

    true_classes = np.identity(num_classes, dtype=float)

    i = 0

    gradw_mse = np.full(
        (num_classes, num_attributes + 1), grad_mse_threshold+1)

    while((i < max_iterations) & ~((np.absolute(gradw_mse) < grad_mse_threshold).all())):
        mse, gradw_mse = combined_mse(
            training_dataset, W_track[-1], true_classes)

        W_track = np.append(
            W_track, [W_track[-1] - (dynamic_alpha(alpha, i, max_iterations) if use_dynamic_alpha else alpha) * gradw_mse], axis=0)

        mse_track.append(mse)

        i += 1

    print("W: ", W_track[-1])
    if plot_result:
        plt.plot(mse_track)
        plt.xlabel("Iterations")
        plt.ylabel("MSE")
        # plt.show()

    print("Completed training with ", i, " iterations")

    return W_track[-1]


def get_confusion_matrix(W, training_dataset, alpha=alpha, plot_result=False):
    confusion_matrix = np.zeros((num_classes, num_classes))

    for c in range(num_classes):
        for i in range(testing_size):
            prediction = np.argmax(
                linear_descriminat_classifier(training_dataset[c][i], W))
            confusion_matrix[c, prediction] += 1

    if plot_result:
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(5, 5))

        sns.heatmap(confusion_matrix,  annot=True, square=True)

        ax.xaxis.tick_bottom()
        plt.xticks(np.arange(3) + .5, rotation=45, labels=classes)
        ax.yaxis.tick_left()
        plt.yticks(np.arange(3) + .5, rotation=45, labels=classes)

        # axis labels
        plt.xlabel('Class')
        plt.ylabel('Class')

        plt.title('alpha='+str(alpha))

        plt.show()

    return confusion_matrix


def dynamic_alpha(initial_alpha, iteration, total_iterations):
    return initial_alpha*np.exp(-iteration/(1*total_iterations))


def calculate_error_rate(confusion_matrix):
    # TODO: add check for non square matrix
    dim = np.size(confusion_matrix, 0)
    num_classifications_class = np.zeros((dim), dtype=float)
    error_rate = np.zeros(dim)

    for row in range(dim):
        for col in range(dim):
            num_classifications_class[row] += confusion_matrix[row, col]

        error_rate[row] = 1 - confusion_matrix[row, row] / \
            num_classifications_class[row]

    return error_rate

    # for i in range(dimention):


def plot_hist(dataset):

    sns.set_theme(style="darkgrid")
    fig, axs = plt.subplots(2, 2)

    row = 0
    for a in range(num_attributes):
        for c in range(num_classes):
            axs[a % 2, row].hist(dataset[c, :, a], bins=12, alpha=0.7)
            axs[a % 2, row].set_title(attributes[a])
            axs[a % 2, row].set_xlabel('cm')
            axs[a % 2, row].set_ylabel('Number of instances')
        axs[a % 2, row].legend(classes, loc='upper right')
        if a == 1:
            row += 1
    fig.set_size_inches(3*3, 2*3)
    fig.tight_layout(pad=.2)
    # for c in range(num_classes):
    #     for a in range(num_attributes):
    #         axs[c, a].hist(dataset[c, :, a], bins=10)
    #         axs[c, a].set_title(classes[c]+" "+attributes[a])
    # fig.tight_layout(pad=.2)

    # fig.show()
    # print(dataset[0, :, 0])
    # plt.hist(dataset[0, :, 0],)
    # sns.displot(dataset[0, :, 0])
    plt.show()


def reduce_dataset_attributes(dataset, removed_attribute_indexs):

    reduced_dataset = np.delete(dataset, removed_attribute_indexs, axis=2)

    current_num_attributes = np.size(dataset, axis=2) - 1

    print("current_num_attributes: ", current_num_attributes)

    num_attributes = current_num_attributes - len(removed_attribute_indexs)

    return reduced_dataset, num_attributes


def format_dataset(datasets):
    formated_dataset = np.empty((num_classes, num_data, num_attributes+1))

    joined_dataset = np.stack(datasets, axis=0)

    for c in range(len(datasets)):
        formated_dataset[c] = np.array([np.append(data, 1)
                                        for data in joined_dataset[c]])

    # dataset[0] = np.array([np.append(data, 1) for data in x1all])

    # dataset[1] = np.array([np.append(data, 1) for data in x2all])
    # dataset[2] = np.array([np.append(data, 1) for data in x3all])

    return formated_dataset

# ----------------------------------------------------------- #
# -------------------------- run ---------------------------- #
# ----------------------------------------------------------- #


dataset = format_dataset([x1all, x2all, x3all])


# W_track = np.full([1, num_classes, num_attributes+1], 0)

# alphas = [0.5, 0.1, 0.05, 0.01]

# fig = plt.figure(figsize=(16/2, 9/2))
# sns.set_theme(style="darkgrid")
# for alpha in alphas:

#     W_track_final = train_LC(training_dataset, W_track,
#  alpha=alpha, total_iterations=1000, plot_result=True)


alphas = [0.5, 0.1, 0.05, 0.01]


# plot histograms
# plot_hist(dataset)

# features_to_remove = [[], [2], [0, 2], [0, 1, 2]]
# features_to_remove = [[], [1], [1, 2], [0, 1, 2]]
features_to_remove = [[], [0], [0, 1], [0, 1, 2]]

for removed in features_to_remove:

    # set alternative dataset with reduced number of attributes
    alternative_dataset, num_attributes = reduce_dataset_attributes(
        dataset, removed)
    print("num_attributes: ", num_attributes)

    W_track = np.full([1, num_classes, num_attributes+1], 0)

    training_dataset, testing_dataset = split_dataset(
        alternative_dataset, training_size)

    W_track_final = train_LC(training_dataset, W_track,
                             alpha=alpha, grad_mse_threshold=0.07, max_iterations=10000, use_dynamic_alpha=True,  plot_result=False)

# plt.legend(["alpha = "+str(alpha) for alpha in alphas], loc="upper right")
# plt.show()

    confusion_matrix = get_confusion_matrix(
        W_track_final, testing_dataset, alpha, True)

    print("Confusion matrix", confusion_matrix)

    error_rate = calculate_error_rate(confusion_matrix)
    print("Error rate", error_rate)
    print("final W", W_track_final)
