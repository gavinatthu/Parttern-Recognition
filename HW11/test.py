import numpy as np
import scipy.io as sio
import math
import time

class MyTree(object):
    def __init__(self, data, datalabel, depth):
        self.leftchild = None
        self.rightchild = None
        self.label = 0			#判断这一类属于什么文档类型
        self.keynumber = -1		#作为该节点继续向下分的关键词编号，叶子结点为-1
        self.delta_entropy = 0.0
        self.entropy = 0.0		#该节点的熵
        self.data = data
        self.datalabel = datalabel
        self.count = dict()
        self.depth = depth
        self.error_num = 0

    def Impurity(self):
        '''不纯度采用Shannon Entropy进行计算'''
        num = len(self.datalabel)
        for i in range(num):
            ind = self.datalabel[i][0]
            if ind not in self.count:
                self.count[ind] = 1
            else:
                self.count[ind] += 1

        maxv = 0
        self.entropy = 0.0
        for i in self.count:
            p = self.count[i]/num
            self.entropy -= p*math.log2(p)
            if self.count[i] > maxv:
                maxv = self.count[i]
                self.label = i

    def SelectFeature(self):
        '''在当前节点选择待分特征:具有最大信息增益的特征'''
        keyamount = len(self.data[0])
        docamount = len(self.data)
        for k in range(keyamount):
            leftdata = []
            leftdatalabel = []
            rightdata = []
            rightdatalabel = []
            for i in range(docamount):
                if self.data[i][k]:
                    leftdata.append(self.data[i])
                    leftdatalabel.append(self.datalabel[i])
                else:
                    rightdata.append(self.data[i])
                    rightdatalabel.append(self.datalabel[i])

            templeftchild = Tree(leftdata, leftdatalabel, self.depth + 1)
            temprightchild = Tree(rightdata, rightdatalabel, self.depth + 1)
            templeftchild.Impurity()
            temprightchild.Impurity()
            tempde = self.entropy - (len(leftdata)*templeftchild.entropy/docamount +
                                    len(rightdata)*temprightchild.entropy/docamount)
            if tempde > self.delta_entropy:
                self.delta_entropy = tempde
                self.leftchild = templeftchild
                self.rightchild = temprightchild
                self.keynumber = k

    def SplitNode(self, de_threshold, depth_threshold):
        if self.delta_entropy > de_threshold and self.depth < depth_threshold:
            self.data = None
            self.datalabel = None
            self.count = dict()
            self.delta_entropy = 0.0
            self.entropy = 0.0
            return True
        else:
            self.leftchild = None
            self.rightchild = None
            self.keynumber = -1
            self.data = None
            self.datalabel = None
            self.count = dict()
            self.delta_entropy = 0.0
            self.entropy = 0.0
            return False

    def GenerateTree(self, de_threshold, depth_threshold):
        self.SelectFeature()
        if self.SplitNode(de_threshold, depth_threshold):
            self.leftchild.GenerateTree(de_threshold, depth_threshold)
            self.rightchild.GenerateTree(de_threshold, depth_threshold)

    def Refresh(self, data, datalabel):
        '''计算当前节点下的错误个数'''
        self.error_num = 0
        leftdata = []
        leftdatalabel = []
        rightdata = []
        rightdatalabel = []
        for i in range(len(data)):
            if datalabel[i][0] != self.label:
                self.error_num += 1
            if self.keynumber >= 0:
                if data[i][self.keynumber]:
                    leftdata.append(data[i])
                    leftdatalabel.append(datalabel[i])
                else:
                    rightdata.append(data[i])
                    rightdatalabel.append(datalabel[i])
        data = None
        datalabel = None
        if self.keynumber >= 0:
            self.leftchild.Refresh(leftdata, leftdatalabel)
            self.rightchild.Refresh(rightdata, rightdatalabel)

    def sum_error_num(self):
        '''递归计算总错误个数'''
        if self.keynumber < 0:
            return self.error_num
        return self.leftchild.sum_error_num() + self.rightchild.sum_error_num()

    def Decision(self, testdata, testlabel):
        '''使用生成的树 GenerateTree，对样本 XToBePredicted 进行预测'''
        amount = len(testlabel)
        self.Refresh(testdata, testlabel)
        error = self.sum_error_num()
        accuracy = (amount - error)/amount
        return accuracy


def Dataloader(path):
    '''数据文件读取，数据集划分'''
    np.random.seed(24)
    data = sio.loadmat(path)
    wordmat = data['wordMat']
    label = data['doclabel']
    num_total = wordmat.shape[0]
    shuffled_indices = np.random.permutation(num_total)
    train_indices = shuffled_indices[:int(num_total*0.6)]
    valid_indices = shuffled_indices[int(num_total*0.6):int(num_total*0.8)]
    test_indices = shuffled_indices[int(num_total*0.8):]
    train_data, train_label = wordmat[train_indices], label[train_indices]
    valid_data, valid_label = wordmat[valid_indices], label[valid_indices]
    test_data, test_label = wordmat[test_indices], label[test_indices]
    return train_data, train_label, valid_data, valid_label, test_data, test_label


def main():

    # 超参数设置
    de_threshold = 0.001
    depth_threshold = 100
    PATH = './Sogou_data/Sogou_webpage.mat'
    traindata, trainlabel, crossdata, crosslabel, testdata, testlabel = Dataloader(PATH)
    time_start = time.time()
    mytree = MyTree(traindata, trainlabel, 0)
    de_threshold = 0.01
    depth_threshold = 100
    mytree.Impurity()
    mytree.GenerateTree(de_threshold, depth_threshold)
    #mytree.Prune(crossdata, crosslabel)
    Test_acc = mytree.Decision(testdata, testlabel)
    Train_acc = mytree.Decision(traindata, trainlabel)
    print("de_threshold={0}, depth_threshold={1}, Train_acc = {2}, Test_acc = {3}, Time = {4}s".format(de_threshold,
                                                                            depth_threshold, Train_acc, Test_acc,
                                                                            time.time() - time_start))

if __name__ == "__main__":
    main()