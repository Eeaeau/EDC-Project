import scipy.io as sio
import numpy as np
from scipy.spatial import distance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import torch
import math
from kmeans_pytorch import kmeans

# ----------------------------- constants -------------------------------- #
mat_contents = sio.loadmat("MNist_ttt4275/data_all.mat")

num_clusters = 64


# ----------------------------- functions -------------------------------- #

def knn(reference_images, labels, image, k, use_tensor=False):
    neighbor_distance_and_indices = []

    if(use_tensor):

        for i in range(len(reference_images)):
            # same as norm(2) (reference_images[i] - image)
            print(image)
            dst = torch.cdist(torch.transpose(
                reference_images[i], 0, 1), torch.reshape(image, (28*28, 1)))
            #dst = sum((reference_images - image)**2)
            neighbor_distance_and_indices.append((dst, labels[i]))

        sorted_neighbor_distances_and_indices = sorted(
            neighbor_distance_and_indices)

        k_nearest_neighbors = sorted_neighbor_distances_and_indices[:k]

        # TODO steal the magic from the number 10
        count_neighbors = torch.zeros(10)
        total_distance = 0

        for neighbor in k_nearest_neighbors:
            count_neighbors[neighbor[1][0]] += 1
            total_distance += neighbor[0]

        avrage_distance = total_distance/k

        return [torch.argmax(count_neighbors), avrage_distance]

    else:
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
            print(neighbor)
            count_neighbors[neighbor[1]] += 1
            total_distance += neighbor[0]

        avrage_distance = total_distance/k

        return [np.argmax(count_neighbors), avrage_distance]


def partition_data_set(data_set, data_labels, num_clusters=64, use_tensor=True):

    data_set_partitions = []

    numbers = [[number] for number in range(10)]

    for number in numbers:
        label_indices = np.where(data_labels == number)[0]

        label_data = data_set[label_indices]
        print("dataset_tensor shape: ", np.shape(label_data))

        # torch method using tensor
        if use_tensor:

            dataset_tensor = torch.from_numpy(label_data)
            print("dataset_tensor: ", dataset_tensor)

            # kmeans
            cluster_ids_x, cluster_centers = kmeans(
                X=dataset_tensor, num_clusters=num_clusters, distance='euclidean', tol=0.1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
            print("cluster_centers: ", cluster_centers)

            data_set_partitions.append([cluster_ids_x, cluster_centers])

        # sklearn method using numpy array
        else:

            cluster = KMeans(n_clusters=num_clusters).fit(
                label_data).cluster_centers_
            data_set_partitions.append(cluster)

    return data_set_partitions


def get_confusion_matrix(training_data, training_labels, test_data, test_labels, use_tensor=False, plot_result=False):

    confusion_matrix = np.zeros([10, 10])

    for i in range(len(test_data)):
        # print(test_labels[i][0])

        prediction = knn(training_data, training_labels,
                         test_data[i], 1, use_tensor)
        confusion_matrix[test_labels[i][0]][prediction[0]] += 1
        print(confusion_matrix)

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
            dataset[i + start_index].reshape(28, 28), interpolation='none')

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

    plt.show()


def format_partition_data_cpu(partition, num_clusters):

    partitioned_data_set = []

    for matrix in partition:
        for row in matrix:
            partitioned_data_set.append(row)

    # print(partitioned_data_set)

    partition_labels = []
    for i in range(10):
        for j in range(num_clusters):
            partition_labels.append(i)

    return partitioned_data_set, partition_labels

# --------------------------------------------------------- #
# -------------------------- run -------------------------- #
# --------------------------------------------------------- #


#print( len(mat_contents['trainv']) )
# nn = knn(mat_contents['trainv'], mat_contents['trainlab'],
#          mat_contents['testv'][0], 1)
# print(nn)

# print(np.shape(mat_contents['testv']))
#get_confusion_matrix(mat_contents['trainv'], mat_contents['trainlab'], mat_contents['testv'][0:1000], mat_contents['testlab'][0:1000], True)

# partition = partition_data_set(
#   mat_contents['trainv'], mat_contents['trainlab'], 64, True)
plot_image(mat_contents['trainv'], mat_contents['trainlab'], 0+4*2, 4+4*2)


partition = partition_data_set(
    mat_contents['trainv'], mat_contents['trainlab'], num_clusters, False)
partition_data_set, partition_labels = format_partition_data_cpu(partition, 64)

get_confusion_matrix(partition_data_set, partition_labels,
                     mat_contents['testv'], mat_contents['testlab'], False, True)
#print("partition: ", partition[:][1][1])


partition_cpu = []
partition = partition_data_set(
    mat_contents['trainv'], mat_contents['trainlab'], 64, True)

print(partition[1][1][0])
cuda_partition_data_set = []
for i in range(len(partition)):
    for j in range(len(partition[0][1])):
        cuda_partition_data_set.append(partition[i][1][j])
print("cuda, partitioned data", cuda_partition_data_set)

partition_labels = []
for i in range(10):
    for j in range(64):
        partition_labels.append(i)

get_confusion_matrix(cuda_partition_data_set, partition_labels, torch.tensor(
    (mat_contents['testv'])), mat_contents['testlab'], True, True)

# for p in partition:
# partition_cpu.append([1].detach().to('cpu').numpy())
# partition_cpu.append([element.detach().to('cpu').numpy() for element in partition[:][1][1]])

#partition_cpu = np.array(partition_cpu)

# vil ha datasett på formen [[cluster_0_for tallet 0],[cluster_1_for tallet 0], (...) [cluster_63_for tallet 0], (...) [cluster_0_for tallet 9], [cluster_63_for tallet 0]]

print("partition_cpu shape: ", np.shape(partition_cpu))

print("partition_cpu: ", partition_cpu)
# .detach().to('cpu').numpy()
# conf_mat = get_confusion_matrix(mat_contents['trainv'], mat_contents['trainlab'], mat_contents['testv'][0:30] , mat_contents['testlab'][0:30], True)
# print(conf_mat)

# plot_image(mat_contents['trainv'], mat_contents['trainlab'], 0, 7)
