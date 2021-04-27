import torch
import numpy as np
from kmeans_pytorch import kmeans

# data
data_size, dims, num_clusters = 1000, 2, 64
x = np.random.randn(data_size, dims) / 6
x = torch.from_numpy(x)
print(x)


cluster_ids_x, cluster_centers = kmeans(
    X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
)

print(cluster_ids_x)
print(cluster_centers)
