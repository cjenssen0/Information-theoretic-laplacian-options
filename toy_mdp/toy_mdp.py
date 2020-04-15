import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform, cdist
from sklearn.neighbors import kneighbors_graph
from scipy import linalg
# two Gaussian components
n = 100

x1 = np.random.multivariate_normal([0,0], np.eye(2), n)
x2 = np.random.multivariate_normal([10, 0], np.eye(2), n)

d12 = cdist(x1, x2)

min12 = np.argmin(d12)

c12_1, c12_2 = np.unravel_index(min12, [n, n])

plt.figure()

# Merging the Gaussians to a data set
x = np.concatenate([x1, x2])
plt.scatter(x[:,0], x[:, 1])
plt.scatter(x1[c12_1, 0], x1[c12_1, 1])
plt.scatter(x2[c12_2, 0], x2[c12_2, 1])
plt.show()


# Finding closest indices in all components

# x = x.T

# Pairwise distance matrix
D = squareform(pdist(x))


# Creating a nn - graph

nn = kneighbors_graph(x, 4, mode='connectivity', include_self=True)
nn = nn.toarray()

# plt.figure(figsize=(20,20))
# plt.scatter(x[0, :], x[1, :])

# for i in range(400):
    # indices = np.squeeze(np.nonzero(nn[i,:]))
    # plt.plot([x[0, indices[0]], x[0, indices[1]]],[x[1, indices[0]], x[1, indices[1]]])
