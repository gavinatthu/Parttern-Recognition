import numpy as np

import matplotlib.pyplot as plt
def make_z_curve(n_samples=100, *, noise=0.0, random_state=None):

    t = 3 * np.pi * (np.random.rand(1, n_samples) - 0.5)
    x = -np.sin(t)
    y = 2.0 * np.random.rand(1, n_samples)
    z = np.sign(t) * (np.cos(t) - 1)

    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    t = np.squeeze(t)
    return X, t

def make_w_curve(n_samples=100, *, noise=0.0, random_state=None):
    x = 4.0 * np.random.rand(1, n_samples) - 2
    z = np.cos(np.pi * x)
    y = 2.0 * np.random.rand(1, n_samples)

    X = np.concatenate((x, y, z))
    X += noise * np.random.randn(3, n_samples)
    X = X.T
    return X, x


class My_Isomap():
    def __init__(self, n_neighbors=30, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def isomap(self, input):
        self.input = input
        dist = self.cal_pairwise_dist()
        dist[dist < 0] = 0
        dist = dist**0.5
        dist_floyd = self.floyd(dist)
        data_n = self.my_mds(dist_floyd, n_dims=self.n_components)
        return data_n

    def cal_pairwise_dist(self):
        x = self.input
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        return dist

    def floyd(self, D):
        Max = np.max(D)*1000
        n1,_ = D.shape
        k = self.n_neighbors
        D1 = np.ones((n1,n1))*Max
        D_arg = np.argsort(D,axis=1)
        for i in range(n1):
            D1[i,D_arg[i,0:k+1]] = D[i,D_arg[i,0:k+1]]
        for k in range(n1):
            for i in range(n1):
                for j in range(n1):
                    if D1[i,k]+D1[k,j]<D1[i,j]:
                        D1[i,j] = D1[i,k]+D1[k,j]
        return D1

    def my_mds(self, dist, n_dims):
        # dist (n_samples, n_samples)
        dist = dist**2
        n = dist.shape[0]
        T1 = np.ones((n,n))*np.sum(dist)/n**2
        T2 = np.sum(dist, axis = 1)/n
        T3 = np.sum(dist, axis = 0)/n
        B = -(T1 - T2 - T3 + dist)/2
        eig_val, eig_vector = np.linalg.eig(B)
        index_ = np.argsort(-eig_val)[:n_dims]
        picked_eig_val = eig_val[index_].real
        picked_eig_vector = eig_vector[:, index_]

        return picked_eig_vector*picked_eig_val**(0.5)

    def scatter_3d(self, y):
        X = self.input
        fig = plt.figure(figsize=(12, 10))
        fig.suptitle("Manifold Learning with %i points, %i neighbors"
             % (1000, self.n_neighbors), fontsize=18)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral)
        ax.view_init(10, -70)
        ax.set_xlabel("$X$", fontsize=18)
        ax.set_ylabel("$Y$", fontsize=18)
        ax.set_zlabel("$Z$", fontsize=18)
        plt.show()
        plt.savefig("scatter_3d.png")

class My_LLE():
    def __init__(self, n_neighbors, n_components):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def cal_pairwise_dist(self):
        x = self.input
        sum_x = np.sum(np.square(x), 1)
        dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
        return dist

    def get_n_neighbors(self):
        dist = self.cal_pairwise_dist()
        dist[dist < 0] = 0
        dist = dist**0.5
        n = dist.shape[0]
        N = np.zeros((n, self.n_neighbors))
        for i in range(n):
            index_ = np.argsort(dist[i])[1:self.n_neighbors+1]
            N[i] = N[i] + index_
        return N.astype(np.int32)

    def lle(self, input):
        self.input = input
        N = self.get_n_neighbors()
        n, D = self.input.shape
        if self.n_neighbors > D:
            tol = 1e-3
        else:
            tol = 0
        W = np.zeros((self.n_neighbors, n))
        I = np.ones((self.n_neighbors, 1))
        for i in range(n):
            Xi = np.tile(self.input[i], (self.n_neighbors, 1)).T
            Ni = self.input[N[i]].T
            Si = np.dot((Xi-Ni).T, (Xi-Ni))
            Si = Si+np.eye(self.n_neighbors)*tol*np.trace(Si)
            Si_inv = np.linalg.pinv(Si)
            wi = (np.dot(Si_inv, I))/(np.dot(np.dot(I.T, Si_inv), I)[0,0])
            W[:, i] = wi[:,0]
        W_y = np.zeros((n, n))
        for i in range(n):
            index = N[i]
            for j in range(self.n_neighbors):
                W_y[index[j],i] = W[j,i]
        I_y = np.eye(n)
        M = np.dot((I_y - W_y), (I_y - W_y).T)
        eig_val, eig_vector = np.linalg.eig(M)
        index_ = np.argsort(np.abs(eig_val))[1:self.n_components+1]
        Y = eig_vector[:, index_]
        return Y


