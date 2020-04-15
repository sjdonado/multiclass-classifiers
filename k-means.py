import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import dataset

N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = dataset.get_data(N, D, K)
num_examples = N*K

# hyperparameters
epochs = 1

# generate random centroids
limit = round(num_examples/K)
randp_1 = np.random.randint(limit)
randp_2 = np.random.randint(limit, limit*2)
randp_3 = np.random.randint(limit*2, limit*3)
centroids = X[[randp_1, randp_2, randp_3], :]

for i in range(epochs):
  neigh = NearestNeighbors(round(num_examples / K)).fit(X)
  # calculate nearest neighbors by K
  neigh_search = neigh.kneighbors(centroids, return_distance=False)

  next_points = []
  for c in neigh_search:
    # calculate mean
    c_mean = np.mean(X[c, :], axis=0, keepdims=True)
    neigh = NearestNeighbors(1).fit(X)
    next_c = neigh.kneighbors(c_mean, return_distance=False)
    next_points.append(next_c.reshape(1)[0])

  centroids = X[next_points, :]
  predicted_class = np.zeros(N*K, dtype=int)
  predicted_class[neigh_search[1]] = 1
  predicted_class[neigh_search[2]] = 2

  accuracy = np.mean(predicted_class == y)
  print("iteration: {} accuracy: {}".format(i, accuracy))

fig = plt.figure()
cmap = plt.cm.get_cmap("Spectral")

plt.subplot(3, 1, 1)
plt.title("Accuracy: {}".format(accuracy))
plt.scatter(X[:, 0], X[:, 1], s=20, c=y, cmap=cmap, edgecolors='k')

plt.subplot(3, 1, 2)
plt.scatter(X[:, 0], X[:, 1], s=20, cmap=cmap, edgecolors='k')
plt.scatter(centroids[:, 0], centroids[:, 1], s=20, cmap=cmap, edgecolors='r')

plt.subplot(3, 1, 3)
plt.scatter(X[:, 0], X[:, 1], s=20, c=predicted_class, cmap=cmap, edgecolors='k')

plt.show()
# fig.savefig('k_means.png')
