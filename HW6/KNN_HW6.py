import numpy as np
import matplotlib.pyplot as plt
import mnist_data_loader
import tensorflow as tf
import time

class KNN():
    def __init__(self, batch_size, k, metrics):
        self.batch_size = batch_size
        self.k = k
        self.metrics = metrics


    def dataload(self):

        # Data Preprocessing
        mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/")
        train_set = mnist_dataset.train
        test_set = mnist_dataset.test

        # train dataset
        train_set = train_set.next_batch(self.batch_size)
        self.input, self.label = train_set
        
        # test dataset
        test_set = test_set.next_batch(1000)
        self.test_input, self.test_label = test_set



    def find_labels(self, num):
        test = self.test_input[num]
        dis_list = []
        labels = []
        for i in range(self.batch_size):
            dis = np.linalg.norm(self.input[i] - test, ord = self.metrics)
            dis_list.append(dis)
        sorted_dis = np.argsort(dis_list)
        for j in range(self.k):
            labels.append(self.label[sorted_dis[j]])
        max_labels = max(labels, key=labels.count)
        return max_labels


    def classifier(self):
        result = []
        acc = 0

        # Training and test
        for i in range(self.test_label.shape[0]):
            knn_labels = self.find_labels(i)
            result.append(knn_labels)
            if knn_labels == self.test_label[i]:
                acc += 1
        #print(acc)
        
        # Accurate rate
        acc_rate = acc/1000
        return result, acc_rate

res = []
acc = []

for num in [100, 300, 1000, 3000, 10000]:
    for k in [1, 2, 5, 10]:
        for m in[ np.inf ,2, 1]:
            start = time.perf_counter()
            print("Training sample =", num, "; k =", k, "; metrics =", m)

            KNN_obj = KNN(num, k, m)
            KNN_obj.dataload()
            result, acc_rate = KNN_obj.classifier()
            res.append(result)
            acc.append(acc_rate)

            end = time.perf_counter()
            Time = end - start
            print("Accuracy =", acc_rate)
            print("CPU time =", Time, '\n')

