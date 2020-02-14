from scipy import linalg
from sklearn.metrics import pairwise_distances
import numpy as np
import environment

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Option object would learn eigen-options for the enviornment
# env = environment.RoomEnvironment()
env = environment.AsymmetricRoomEnvironment()
# env = environment.LargeDoorsRoomEnvironment()
# env = environment.ClosedRoomEnvironment()

max_row, max_col = env.get_grid_dimension()
room_size = max_row

# Copied from the options code

default_max_actions = env.get_default_max_actions()

# get all possible (r,c) states in env
states_rc = []
for r in range(max_row):
    for c in range(max_col):
        states_rc.append((r, c))

total_states = len(states_rc)

# =======================================================================
# Compute adjacency matrix (take all possible actions from every state)
# =======================================================================
# adjacency = np.zeros((total_states, total_states), dtype = np.int)

# for state in range(total_states):
    # for a in range(default_max_actions):
        # # Take a specified action from a given start state
        # env.set_current_state(state)
        # result = env.step(a)
        # if result['state'] is not None:
            # next_state = result["state"][0]

            # if next_state != state:
                # adjacency[state][next_state] = 1
        # else:
            # break

# # adjacency = adjacency + np.eye(121)
# D = np.zeros((total_states, total_states), dtype = np.int)

# row_sum = np.sum(adjacency, axis=1)
# for state in range(total_states):
   # D[state][state] = row_sum[state]

# =======================================================================
# Compute adjacency matrix using RW
# =======================================================================
adjacency = np.zeros((total_states, total_states), dtype = np.int)

for state in range(total_states):

    if env.states_rc[state] in env.obstacle_vector:
       continue

    for i in range(100):
        env.set_current_state(state)
        for j in range(10):

            result = env.step(np.random.choice(range(4)))
            if result['state'] is not None:
                next_state = result['state'][0]

                adjacency[state][next_state] = 1 # Only add connection
                # adjacency[state][next_state] += 1
            else:
                adjacency[state][next_state] = 1 # Only add connection
                # adjacency[state][next_state] += 1
                break


# ========================
# Laplacian matrix stuff
# ========================

# Calculating the degree matrix
D = np.zeros((total_states, total_states), dtype = np.int)

row_sum = np.sum(adjacency, axis=1)

for state in range(total_states):
   D[state][state] = row_sum[state]

# Normalized laplacian
# diff = D - adjacency
# sq_D = np.diag(1/np.sqrt(np.diag(D)))
# L = np.matmul(sq_D, np.matmul(diff, sq_D))

# Basic laplacian
L = D - adjacency

# Kernelized version
#L = np.exp(-(1 - adjacency)**2 / 1.0)
# L = np.exp(-(L.T*L) / 2.0)

# Heat kernel
# t = 1
# L = np.exp(-np.abs(L))

# Regularized Laplacian kernel
# L = (np.eye(121) + t*L)**(-1)

# ===============
# Kernel matrix
# ===============
D_mat = pairwise_distances(states_rc)
sigma = 2
K = np.exp(-D_mat**2 / sigma)

# Using only the TD connected components
K = K * adjacency

# ======================
# # Entropy components
# ======================
# v_sum = np.dot(v.T, np.ones_like(w))

# scores = (np.sqrt(w)*v_sum)**2
# indexes = np.flip(np.argsort(scores),0)

# w_min_sort = np.argsort(w)

# plt.figure(1)
# for i in range(49):
    # plt.subplot(7, 7, i+1)
    # plt.imshow(v[:, indexes[i]].reshape(11, 11))
    # plt.colorbar()

# plt.suptitle('entropy sorted eigs')

# plt.figure(2)
# for i in range(49):
    # plt.subplot(7, 7, i+1)
    # plt.imshow(v[:, w_min_sort[i]].reshape(11, 11))
    # plt.colorbar()

# plt.suptitle('laplacian sorted eigs')

# plt.figure(3)
# plt.subplot(1, 2, 1)
# plt.stem(scores)
# ======================


# w,v = linalg.eigh(L)
w,v = linalg.eigh(K)

plt.figure(10)
plt.stem(w)
plt.title('Kernel matrix eigvals')
# plt.title('Laplacian matrix eigvals')


plt.figure(3)
for i in range(1, 25):
    plt.subplot(5, 5, i)
    plt.imshow(v[:, -i].reshape(max_row, max_col))
    # plt.imshow(v[:, i+17].reshape(room_size, room_size))
    plt.colorbar()

plt.suptitle('Sorted after eigenvalue (K)')
# plt.show()

# Entropy component analysis
ent = np.sum(v, 0)**2 * w
ent = np.flip(np.argsort(ent), 0)

plt.figure(4)
for i in range(1, 25):
    plt.subplot(5, 5, i)
    plt.imshow(v[:, ent[i-1]].reshape(max_row, max_col))
    plt.colorbar()

plt.suptitle('Entropy components')
# plt.show()

# =========================
# 3D plot of eigenvectors
# =========================
from matplotlib import cm
X = np.arange(max_col)
Y = np.arange(max_row)

X, Y = np.meshgrid(X, Y)
fig = plt.figure(33, figsize=(20,20))

for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i, projection='3d')
    V = v[:, -i].reshape(max_row, max_col)
    ax.plot_surface(X, Y, V, cmap=cm.coolwarm)

# ====================================
# TSNE plot of kernel/laplace matrix
# ====================================

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(K)
# from sklearn.decomposition import PCA
#X_embedded = PCA(n_components=2).fit_transform(K)

plt.figure(1)
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=np.linspace(0, 1, len(X_embedded)))
plt.title('TSNE on the Kernel matrix')
plt.show()
