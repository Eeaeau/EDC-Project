import torch
import numpy as np
from kmeans_pytorch import kmeans
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.datasets import load_iris

# data_df = load_iris()

sns.set_theme(color_codes=True)

# # data
# data_size, dims, num_clusters = 1000, 2, 64
# x = np.random.randn(data_size, dims) / 6
# x = torch.from_numpy(x)
# print(x)


# # cluster_ids_x, cluster_centers = kmeans(
# #     X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
# # )

# # print("cluster_ids_x: ", cluster_ids_x)
# # print("cluster_centers: ", cluster_centers)


# digits = load_digits()
# data = scale(digits.data)

# print("digits: ", digits)
# print("data: ", data)


def linear_plot():

    iris = sns.load_dataset("iris")

    print(iris)

    # g = sns.lmplot(x="sepal_width", y="sepal_length", hue="species", data=iris,
    #                palette="Set1")
    # plt.legend(["class 1", "class 2"])

    sns.pairplot(iris, hue="species", height=2, palette='muted')
    plt.show()


linear_plot()
