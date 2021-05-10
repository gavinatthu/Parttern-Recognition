import matplotlib.pyplot as plt
from sklearn import manifold
from utils import *

n_points = 500
X, color = make_w_curve(n_points, random_state=0)
n_neighbors = 30
n_components = 2


my_isomap = My_Isomap(n_neighbors, n_components)
data_1 = my_isomap.isomap(X)
my_isomap.scatter_3d(color)
data_2 = manifold.Isomap(n_neighbors, n_components).fit_transform(X)


my_lle = My_LLE(n_neighbors, n_components)
data_3 = my_lle.lle(X)
data_4 = manifold.LocallyLinearEmbedding(n_components=2, n_neighbors = 30).fit_transform(X)


plt.figure(figsize=(10,10))
plt.subplot(221)
plt.title("my_Isomap")
plt.scatter(data_1[:, 0], data_1[:, 1], c = color)
plt.subplot(222)
plt.title("sklearn_Isomap")
plt.scatter(data_2[:, 0], data_2[:, 1], c = color)
#plt.savefig("Isomap.png")
#plt.show()


#plt.figure(figsize=(10,5))
plt.subplot(223)
plt.title("My_LLW")
plt.scatter(data_3[:, 0], data_3[:, 1], c = color)
plt.subplot(224)
plt.title("sklearn_LLE")
plt.scatter(data_4[:, 0], data_4[:, 1], c = color)
plt.savefig("LLE.png")
plt.show()