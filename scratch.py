import matplotlib.pyplot as plt
plt.imshow(v[:,1].reshape(10,10))
plt.show()

plt.stem(scores)
plt.show()

from mpl_toolkits.mplot3d import Axes3D
# Make data.
vi = v[:,4]
X = np.arange(10)
Y = np.arange(10)
X, Y = np.meshgrid(X, Y)
Z = vi.reshape(10,10)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z)

plt.show()