import scipy.io as sio
import numpy as np
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
from kmeans_pytorch import kmeans
import math

mat_contents = sio.loadmat("MNist_ttt4275/data_all.mat")


# ----------------------------- functions -------------------------------- #

def knn(reference_images, labels, image, k):
    neighbor_distance_and_indices = []

    for i in range(len(reference_images)):
        # same as norm(2) (reference_images[i] - image)
        dst = distance.euclidean(reference_images[i], image)
        neighbor_distance_and_indices.append((dst, labels[i]))

    sorted_neighbor_distances_and_indices = sorted(
        neighbor_distance_and_indices)

    k_nearest_neighbors = sorted_neighbor_distances_and_indices[:k]

    # TODO steal the magic from the number 10
    count_neighbors = np.zeros(10)
    total_distance = 0

    for neighbor in k_nearest_neighbors:
        count_neighbors[neighbor[1][0]] += 1
        total_distance += neighbor[0]

    avrage_distance = total_distance/k

    return [np.argmax(count_neighbors), avrage_distance]


def partition_data_set(data_set, data_labels, num_clusters=64, use_tensor=False):

    if use_tensor:
        # data

        #print("Label indices", label_indices)
        data_set_partitions = []

        numbers = [[number] for number in range(2)]

        for number in numbers:
            label_indices = np.where(data_labels == number)[0]

            label_data = data_set[label_indices]
            print("dataset_tensor shape: ", np.shape(label_data))
            dataset_tensor = torch.from_numpy(label_data)

            print("dataset_tensor: ", dataset_tensor)

            # kmeans
            cluster_ids_x, cluster_centers = kmeans(
                X=dataset_tensor, num_clusters=num_clusters, distance='euclidean', tol=0.01, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            print("cluster_centers: ", cluster_centers)

            data_set_partitions.append(cluster_ids_x, cluster_centers)

        return data_set_partitions

    else:

        data_set_partitions = []

        numbers = [[number] for number in range(2)]

        for number in numbers:

            label_indices = np.where(data_labels == number)[0]

            #print("Label indices", label_indices)
            label_data = data_set[label_indices]

            cluster = KMeans(n_clusters=num_clusters).fit(
                label_data).cluster_centers_
            data_set_partitions.append([number, cluster])

        return data_set_partitions


def get_confusion_matrix(training_data, training_labels, test_data, test_labels, plot_result=False):

    confusion_matrix = np.zeros([10, 10])

    for i in range(len(test_data)):
        # print(test_labels[i][0])

        prediction = knn(training_data, training_labels, test_data[i], 1)
        confusion_matrix[test_labels[i][0]][prediction[0]] += 1

    if plot_result:
        sns.heatmap(confusion_matrix,  annot=True, square=True)
        plt.show()

    return confusion_matrix


def plot_image(dataset, data_labels, start_index, end_index):

    image_count = end_index-start_index
    nrows = math.floor(image_count/2)
    print(nrows)
    ncols = math.ceil(image_count/2)
    print(ncols)

    # sns.set_theme(style="darkgrid")
    # fig, axs = plt.subplots(nrows, ncols)
    fig, axs = plt.subplots(nrows, ncols)

    row = 0
    cols = 0
    for i in range(image_count):

        axs[row % nrows, cols % ncols].imshow(
            dataset[i].reshape(28, 28), interpolation='none')

        axs[row % nrows, cols % ncols].set_title(
            "Number "+str(data_labels[i][0]))

        cols += 1
        if (i % ncols == 0):
            row += 1

    # axs.reshape(nrows, ncols)
    # for row in range(nrows):
    #     for col in range(ncols):
    #         axs[row, col].imshow(dataset[start_index+row+col].reshape(28,28), interpolation='none')
    #         axs[row, col].set_title("Number "+str(data_labels[start_index+row+col][0]))

    fig.tight_layout(pad=.2)

    # plt.imshow(mat_contents['testv'][0].reshape(28,28), interpolation='bicubic')
    plt.show()

# --------------------------------------------------------- #
# -------------------------- run -------------------------- #
# --------------------------------------------------------- #


#print( len(mat_contents['trainv']) )
# nn = knn(mat_contents['trainv'], mat_contents['trainlab'],
#          mat_contents['testv'][0], 1)
# print(nn)

# print(np.shape(mat_contents['testv']))

partition = partition_data_set(
    mat_contents['trainv'], mat_contents['trainlab'], 64, True)

print("partition: ", partition)

# conf_mat = get_confusion_matrix(mat_contents['trainv'], mat_contents['trainlab'], mat_contents['testv'][0:30] , mat_contents['testlab'][0:30], True)
# print(conf_mat)

# plot_image(mat_contents['trainv'], mat_contents['trainlab'], 0, 7)

print(":)")
