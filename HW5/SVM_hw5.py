import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC



x = np.array([[-3.0, -2.9], [0.5, 8.7], [2.9, 2.1], [-0.1, 5.2], [-4.0, 2.2], [-1.3, 3.7], [-3.4, 6.2], [-4.1, 3.4], [-5.1, 1.6], [1.9, 5.1],
    [-2.0, -8.4], [-8.9, 0.2], [-4.2, -7.7], [-8.5, -3.2], [-6.7, -4.0], [-0.5, -9.2], [-5.3, -6.7], [-8.7, -6.4], [-7.1, -9.7], [-8.0, -6.3]])

y = np.array([0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])
print('step = ', 0)
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()

#########################################
clf = SVC(kernel='linear')
x_train = np.empty([0,2])
y_train = np.empty([0])
for i in range (0, 10):
    x_train = np.concatenate((x_train, x[[i, i + 10]]), axis=0)
    y_train = np.concatenate((y_train, y[[i, i + 10]]), axis=0)
    clf.fit(x_train, y_train)
    w = clf.coef_
    b = clf.intercept_
    margin = 2 / np.linalg.norm(w)
    print(margin)
    xx = np.arange(-10,4,0.01)
    yy = (w[0][0] * xx + b) / (-1 * w[0][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('2-Class SVM')
    plt.xlabel('X1')
    plt.ylabel('X2')
    ax.scatter(x[:, 0], x[:, 1], c=y)
    ax.scatter(x_train[:, 0], x_train[:, 1], c = y_train, cmap = 'cool')
    ax.scatter(xx, yy, s=1, marker = 'h')
    print('step = ', i + 1)
    plt.show()
    





