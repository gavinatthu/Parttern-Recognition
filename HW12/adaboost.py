# -*- coding: utf-8 -*-
import numpy as np
from numpy.core.numeric import Inf
import scipy.io as scio
import matplotlib.pyplot as plt
import time

def adaboost(X, y, X_test, y_test, maxIter):
    '''
    adaboost: carry on adaboost on the data for maxIter loops
    Input 
        X       : n * p matirx, training data
        y       : (n, ) vector, training label
        X_test  : m * p matrix, testing data
        y_test  : (m, ) vector, testing label
        maxIter : number of loops
    Output
        e_train : (maxIter, ) vector, errors on training data
        e_test  : (maxIter, ) vector, errors on testing data
    '''

    w = np.ones(y.shape, dtype='float') / y.shape[0]

    k = np.zeros(maxIter, dtype='int')
    a = np.zeros(maxIter)
    d = np.zeros(maxIter)
    alpha = np.zeros(maxIter)

    e_train = np.zeros(maxIter)
    e_test = np.zeros(maxIter)

    for i in range(maxIter):
        k[i], a[i], d[i] = decision_stump(X, y, w)
        print('new decision stump k:%d a:%f, d:%d' % (k[i], a[i], d[i]))
        
        e = decision_stump_error(X, y, k[i], a[i], d[i], w)
        #alpha[i] = np.log((1 - e) / e)
        alpha[i] = 0.5 * np.log((1 - e) / e)
        w = update_weights(X, y, k[i], a[i], d[i], w, alpha[i])
        
        e_train[i] = adaboost_error(X, y, k, a, d, alpha)
        e_test[i] = adaboost_error(X_test, y_test, k, a, d, alpha)
        print('weak learner error rate: %f\nadaboost error rate: %f\ntest error rate: %f\n' % (e, e_train[i], e_test[i]))

    return e_train, e_test

def plot_error(e_train, e_test):
    plt.figure()
    plt.plot(e_train, label = 'train')
    plt.plot(e_test, label = 'test')
    plt.legend(loc = 'upper right')
    plt.title('Error vs iters')
    plt.xlabel('Iters')
    plt.ylabel('Error')
    plt.savefig('adaboost01.png', dpi=300)
    plt.show()


def decision_stump(X, y, w):
    '''
    decision_stump returns a rule ...
    h(x) = d if x(k) <= a, −d otherwise,
    Input
        X : n * p matrix, each row a sample
        y : (n, ) vector, each row a label
        w : (n, ) vector, each row a weight
    Output
        k : the optimal dimension
        a : the optimal threshold
        d : the optimal d, 1 or -1
    '''

    # total time complexity required to be O(p*n*logn) or less
    ### Your Code Here ###

    num_step = 100
    min_error = Inf

    for i in range(X.shape[1]):
        step = (np.max(X[:, i]) - np.min(X[:, i]))/num_step
        temp_stump = np.min(X[:, i])
        for j in range(num_step):
            for temp_d in [1, -1]:
                E = decision_stump_error(X, y, i, temp_stump, temp_d, w)
                if E < min_error:
                    min_error = E
                    #print('update E=', E)
                    k, a, d = i, temp_stump, temp_d
                    #print(k, a, d)

            temp_stump += step
            
    ### Your Code Here ###
    return k, a, d


def decision_stump_error(X, y, k, a, d, w):
    '''
    decision_stump_error returns error of the given stump
    Input
        X : n * p matrix, each row a sample
        y : (n, ) vector, each row a label
        k : selected dimension of features
        a : selected threshold for feature-k
        d : 1 or -1
    Output
        e : number of errors of the given stump 
    '''
    #p = ((X[:, k] <= a).astype('float') - 0.5) * 2 * d # predicted label
    p = np.where(X[:, k] <= a, d, -d)
    e = np.sum((p.astype('int') != y) * w)

    return e


def update_weights(X, y, k, a, d, w, alpha):
    '''
    update_weights update the weights with the recent classifier
    
    Input
        X        : n * p matrix, each row a sample
        y        : (n, ) vector, each row a label
        k        : selected dimension of features
        a        : selected threshold for feature-k
        d        : 1 or -1
        w        : (n, ) vector, old weights
        alpha    : weights of the classifiers
    
    Output
        w_update : (n, ) vector, the updated weights
    '''

    ### Your Code Here ###
    p = np.where(X[:, k] <= a, d, -d)
    #p = ((X[:, k] <= a).astype('float') - 0.5) * 2 * d
    w_update = w * np.exp(-  alpha * y * p )
    w_update = w_update / np.sum(w_update)

    ### Your Code Here ###
    
    return w_update



def adaboost_error(X, y, k, a, d, alpha):
    '''
    adaboost_error: returns the final error rate of a whole adaboost
    
    Input
        X     : n * p matrix, each row a sample
        y     : (n, ) vector, each row a label
        k     : (iter, ) vector,  selected dimension of features
        a     : (iter, ) vector, selected threshold for feature-k
        d     : (iter, ) vector, 1 or -1
        alpha : (iter, ) vector, weights of the classifiers
    Output
        e     : error rate
    '''

    ### Your Code Here ###

    n, _ = X.shape
    pre_label = []
    sum = np.zeros_like(y)
    for i in range(len(k)):
        #p = ((X[:, k[i]] <= a[i]).astype('float') - 0.5) * 2 * d[i]
        p = np.where(X[:, k[i]] <= a[i], d[i], -d[i])
        temp = alpha[i] * p
        sum = sum + temp

    pre_label.append(np.sign(sum))
    e = np.sum(pre_label!=y) / n
    ### Your Code Here ###
    return e

if __name__ == '__main__':

    dataFile = 'ada_data.mat'
    data = scio.loadmat(dataFile)
    X_train = data['X_train']   #(1000, 25), float
    X_test = data['X_test']     #(1000, 25)
    y_train = data['y_train'].ravel()   #(1000, ), +1 or -1
    y_test = data['y_test'].ravel()     #(1000, )
    
    ### Your Code Here ###
    start = time.time()
    e_train, e_test = adaboost(X_train, y_train, X_test, y_test, 300)
    print('time =', time.time() - start)
    plot_error(e_train, e_test)

    ### Your Code Here ###
