import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


zkc = nx.karate_club_graph()

G = nx.karate_club_graph()
print("Node Degree")
for v in G:
    print(f"{v:4} {G.degree(v):6}")
nx.draw_circular(G, with_labels=True)
plt.show()


order = sorted(list(zkc.nodes()))


def ReLU(input):
    output = np.maximum(0,input)
    return output


def gcn_layer(A_hat, D_hat, X, W):
    return ReLU(np.linalg.inv(D_hat) * A_hat * X * W)


A = nx.to_numpy_matrix(zkc, nodelist=order)
I = np.eye(zkc.number_of_nodes())
A_hat = A + I

D_hat = np.array(np.sum(A_hat, axis=0))
D_hat = np.diag(D_hat[0])

# initialize the weights randomly
W_1 = np.random.normal(
    loc=0, scale=1, size=(zkc.number_of_nodes(), 4))
W_2 = np.random.normal(
    loc=0, size=(W_1.shape[1], 2))

# propagation
H_1 = gcn_layer(A_hat, D_hat, I, W_1)
output = gcn_layer(A_hat, D_hat, H_1, W_2)



nx.draw(zkc,node_color=order,cmap=plt.cm.Blues, with_labels=True)
plt.show

plt.figure(2)
plt.scatter([output[:,0]],[output[:,1]],c= order, cmap=plt.cm.Blues)
plt.show