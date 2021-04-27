import scipy.io as sio
import numpy as np
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

mat_contents = sio.loadmat("Mnist_ttt4275/data_all.mat")
print(mat_contents.keys())
#print(mat_contents['col_size'])
print(mat_contents['testv'])

def knn(reference_images, labels, image, k):
    neighbor_distance_and_indices = []
    
    for i in range( len(reference_images) ):
        dst = distance.euclidean(reference_images[i], image) # same as norm(2) (reference_images[i] - image)
        neighbor_distance_and_indices.append((dst, labels[i]))
    
    sorted_neighbor_distances_and_indices = sorted(neighbor_distance_and_indices)
    
    k_nearest_neighbors = sorted_neighbor_distances_and_indices[:k]

    return k_nearest_neighbors

def partition_data_set(data_set, labels, num_clusters):
    data_set_partitions = []

    for label in [[0],[1],[2],[3],[4],[5],[6],[7],[8],[9]]:
        
        label_indices = np.where(labels == label)[0]
       
        print("Label indices", label_indices)
        label_data = data_set[label_indices]
       
        cluster = KMeans(n_clusters=num_clusters).fit(label_data).cluster_centers_
        data_set_partitions.append([label, cluster])
    
    return data_set_partitions


def get_confusion_matrix(training_data, training_labels,):



def plot_image():
    plt.imshow(mat_contents['testv'][0].reshape(28,28), interpolation='bicubic')
    plt.show()

# --------------------------------------------------------- #
# -------------------------- run -------------------------- #
# --------------------------------------------------------- #

#print( len(mat_contents['trainv']) )
nn = knn(mat_contents['trainv'], mat_contents['trainlab'], mat_contents['testv'][0], 1)
print(nn)

print(np.shape(mat_contents['testv']))

partition = partition_data_set(mat_contents['trainv'], mat_contents['trainlab'], 60)
print(partition)

plot_image()


