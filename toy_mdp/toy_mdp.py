import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform, cdist
from sklearn.neighbors import kneighbors_graph
# from scipy import linalg

def create_env(n1=100, n2=100, sigma1=1, sigma2=1, savestring='environment_txt'):
# two Gaussian components
    # n1 = 100
    # n2 = 100
    # sigma1 = 1
    # sigma2 = 1

    x1 = np.round(np.random.multivariate_normal([0,0], sigma1 * np.eye(2), n1))
    x2 = np.round(np.random.multivariate_normal([10, 0], sigma2 * np.eye(2), n2))

    # d12 = cdist(x1, x2)

    # min12 = np.argmin(d12)

    # c12_1, c12_2 = np.unravel_index(min12, [n1, n2])

    # plt.figure()

    # Merging the Gaussians to a data set
    x = np.concatenate([x1, x2])
    # plt.scatter(x[:,0], x[:, 1])
    # plt.scatter(x1[c12_1, 0], x1[c12_1, 1])
    # plt.scatter(x2[c12_2, 0], x2[c12_2, 1])
    # plt.show()


    # Finding closest indices in all components

    # x = x.T

    # Pairwise distance matrix
    # D = squareform(pdist(x))


    # Creating a nn - graph

    # nn = kneighbors_graph(x, 4, mode='connectivity', include_self=True)
    # nn = nn.toarray()

    # plt.figure(figsize=(20,20))
    # plt.scatter(x[0, :], x[1, :])

    # for i in range(400):
        # indices = np.squeeze(np.nonzero(nn[i,:]))
        # plt.plot([x[0, indices[0]], x[0, indices[1]]],[x[1, indices[0]], x[1, indices[1]]])

    # HACK!
    X = x
    X[:, 0] = X[:, 0] - min(X[:, 0])
    X[:, 1] = X[:, 1] - min(X[:, 1])

    # plt.figure()
    # plt.scatter(X[:,0], X[:, 1])
    # plt.scatter(X[c12_1, 0], X[c12_1, 1])
    # plt.scatter(X[c12_2 + n2, 0], X[c12_2 + n2, 1])
    # plt.show()

    # Creating environment
    mat = np.zeros([int(np.max(X[:, 1])), int(np.max(X[:, 0]))])
    for i, j in X:
        mat[int(j)-1, int(i)-1] = 1

    plt.figure()
    plt.imshow(mat)
    plt.show()

    # Save environment for importin in RLGLUE
    np.savetxt(fname=savestring, X=mat.astype(int), fmt ='%.0f', delimiter='')


if __name__ == '__main__':
    create_env()
    create_env(10, 100, 1, 1, 'environment_mixture_weight2.txt')
    create_env(100, 100, .5, 5, 'environment_variance2.txt')
