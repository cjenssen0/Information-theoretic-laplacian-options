import matplotlib.pyplot as plt
plt.imshow(v[:,53].reshape(10,10))
plt.colorbar()
plt.show()

plt.stem(scores)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
# Make data.
vi = self.eigenvectors[:,0]
#vi = v[:,53]
X = np.arange(10)
Y = np.arange(10)
X, Y = np.meshgrid(Y, X)
Z = vi.reshape(10,10)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)
plt.show()

##
vi = v[:,53]
plt.imshow(vi.reshape(10,10))
plt.show()

#------PLOTS------#
#Plot Ip
plt.stem(scores)
plt.xlabel("i")
plt.ylabel("V")
plt.title("Stem plot of scores for kernel=np.exp(-(L) / 2.0)")
plt.show()

#Plot eigvals
plt.stem(w)
plt.xlabel("i")
plt.ylabel("lambda")
plt.title("Stem plot of sorted eigenvalues for kernel=np.exp(-(L) / 2.0)")
plt.show()

#--Plot eigenvectors
i = 1
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(self.eigenvectors[:,i].reshape(10,10), cmap="viridis")
axs[0, 0].set_title('1st eigenvector')
axs[0, 1].imshow(self.eigenvectors[:,i+1].reshape(10,10))
axs[0, 1].set_title('2nd eigenvector')
axs[1, 0].imshow(self.eigenvectors[:,i+2].reshape(10,10))
axs[1, 0].set_title('3rd eigenvector')
axs[1, 1].imshow(self.eigenvectors[:,i+3].reshape(10,10))
axs[1, 1].set_title('4th eigenvector')
#fig.colorbar(axs, ax=axs.ravel().tolist())

for ax in axs.flat:
    ax.set(xlabel='x', ylabel='y')
    #fig.colorbar(ax, ax=axs.ravel().tolist())
# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
plt.show()
