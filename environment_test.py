import sys
from scipy import linalg
from scipy.spatial.distance import squareform,pdist
import numpy as np
import pickle
import copy

import rlglue
import environment
import agents
import options

import matplotlib
import matplotlib.pyplot as plt

# Option object would learn eigen-options for the enviornment
env = environment.RoomEnvironment()
max_row, max_col = env.get_grid_dimension()

# Plotting the room environment
room = np.zeros([11, 11])
for i in env.obstacle_vector:
    room[i[0], i[1]] = 1


# print(room)

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
adjacency = np.zeros((total_states, total_states), dtype = np.int)

for state in range(total_states):
    for a in range(default_max_actions):
        # Take a specified action from a given start state
        env.set_current_state(state)
        result = env.step(a)
        if result['state'] is not None:
            next_state = result["state"][0]

            if next_state != state:
                adjacency[state][next_state] = 1
        else:
            break

# adjacency = adjacency + np.eye(121)
D = np.zeros((total_states, total_states), dtype = np.int)

row_sum = np.sum(adjacency, axis=1)
for state in range(total_states):
   D[state][state] = row_sum[state]

# =======================================================================
# Compute adjacency matrix using RW
# =======================================================================
adjacency = np.zeros((total_states, total_states), dtype = np.int)

for state in range(total_states):

    if env.states_rc[state] in env.obstacle_vector:
       continue

    for i in range(1000):
        env.set_current_state(state)
        for j in range(10):

            result = env.step(np.random.choice(range(4)))
            if result['state'] is not None:
                next_state = result['state'][0]

                adjacency[state][next_state] += 1
            else:
                adjacency[state][next_state] += 1
                break

plt.figure(121)
for i in range(1, 10):
    plt.subplot(2, 5, i)
    plt.imshow(np.reshape(adjacency[i-1,:], [11, 11]))
    plt.show()

# raise
# adjacency = adjacency/10000
D = np.zeros((total_states, total_states), dtype = np.int)

row_sum = np.sum(adjacency, axis=1)
for state in range(total_states):
   D[state][state] = row_sum[state]

# ========================
# Laplacian matrix stuff
# ========================

# Normalized laplacian
diff = D - adjacency
# sq_D = np.diag(1/np.sqrt(np.diag(D)))
# L = np.matmul(sq_D, np.matmul(diff, sq_D))

# Basic laplacian
L = diff

# Kernelized version
#L = np.exp(-(1 - adjacency)**2 / 1.0)
# L = np.exp(-(L.T*L) / 2.0)

# Heat kernel
# t = 1
# L = np.exp(-np.abs(L))

# Regularized Laplacian kernel
# L = (np.eye(121) + t*L)**(-1)


# w,v = linalg.eigh(L)

# plt.figure()
# plt.plot(w)
# plt.show()

# # Entropy components
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

# plt.subplot(1, 2, 2)
# plt.stem(w)

# plt.show()
# plt.figure(4)
# plt.imshow(L)



''' Kernel matrix testing '''
kernel_size = 1
# K = np.exp(-0.5*(D)**2/kernel_size)
K = np.exp(-0.5*(adjacency/1000*10)**2/kernel_size)
# K = np.exp(-0.5*(adjacency)**2/kernel_size)
# K = adjacency/10000

#new_cool_mat = (adjacency + adjacency.T) / 2

# plt.imshow(adjacency+adjacency.T)
# plt.show()

# new_cool_mat = 50*np.eye(121)+new_cool_mat
# new_cool_mat = new_cool_mat.max(1)-new_cool_mat
# new_cool_mat = np.dot(new_cool_mat, new_cool_mat.t)

# plt.imshow(new_cool_mat[0].reshape(11, 11))
# plt.show()
# raise

#new_cool_mat = new_cool_mat / new_cool_mat.max(1)
# K = np.exp(-0.5*(new_cool_mat)**2/kernel_size)
# L = np.diag(new_cool_mat.sum(1))-new_cool_mat

w,v = linalg.eigh(K)

plt.figure(10)
plt.stem(w)
plt.title('Kernel matrix eigvals')

plt.figure(2)
plt.imshow(K)
plt.colorbar()

plt.figure(3)
for i in range(1, 25):
    plt.subplot(5, 5, i)
    # plt.imshow(v[:, -i].reshape(11, 11))
    plt.imshow(v[:, i-1].reshape(11, 11))
    plt.colorbar()

plt.figure(4)
plt.imshow(v[:, -1].reshape(11, 11))
plt.colorbar()

plt.show()
