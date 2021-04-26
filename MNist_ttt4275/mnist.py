import scipy.io as sio
import numpy as np
from scipy.spatial import distance
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

plot_image()


