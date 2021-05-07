import numpy as np
import pandas as pd

def getdata():
    dataset = pd.read_csv('./watermelon_3.csv')
    
    Map = {}
    del dataset['编号']
    #
    
    Map['浅白'],Map['青绿'],Map['乌黑']=0, 0.5, 1
    Map['蜷缩'],Map['稍蜷'],Map['硬挺']=0, 0.5, 1
    Map['沉闷'],Map['浊响'],Map['清脆']=0, 0.5, 1
    Map['模糊'],Map['稍糊'],Map['清晰']=0, 0.5, 1
    Map['凹陷'],Map['稍凹'],Map['平坦']=0, 0.5, 1
    Map['硬滑'],Map['软粘']=0, 1
    Map['否'],Map['是']=0, 1
    '''
    Python 字典类没法实现多对一
    Map = {
    ('浅白','蜷缩','沉闷','模糊','凹陷','硬滑','否'): int(0),
    ('青绿', '稍蜷', '浊响','稍糊','稍凹') : int(0.5),
    ('乌黑', '硬挺', '清脆','清晰','平坦','软粘','是') :int(1),
    }
    '''
    data = dataset.values
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
        #data[i] = list(Map(int, dataset.values[:,i]))
            if data[i, j] in Map:
                data[i, j] = Map[data[i, j]]
    #data = Map[dataset.values[0,1 ]]
    features = dataset.columns.values
    return data, features


def diff(dataSet, i, j, mode=""):
    exDataSet = None
    if mode == 'nh':
        exDataSet = dataSet[dataSet[:, -1] == dataSet[i][-1]]
    if mode == 'nm':
        exDataSet = dataSet[dataSet[:, -1] != dataSet[i][-1]]
    dist = np.inf
    if j < 6:
        dist = 1            # 对于离散型数据，初始dist为1，当遇到相同的j属性值时，置零。
        for k in range(len(exDataSet)):
            if k == i:      # 遇到第i个样本跳过。
                continue
            if exDataSet[k][j] == dataSet[i][j]:
                dist = 0
                break
    else:
        for k in range(len(exDataSet)):
            if k == i:
                continue
            sub = abs(float(exDataSet[k][j]) - float(dataSet[i][j]))
            if sub < dist:
                dist = sub
    return dist


def Relief(input):
    n_samples, n_features = input.shape
    relief = []
    for j in range(n_features - 1):
        rj = 0
        for i in range(n_samples):
            diff_nh = diff(input, i, j, mode='nh')
            diff_nm = diff(input, i, j, mode='nm') 
            rj += diff_nm**2 - diff_nh**2
        relief.append(rj)
    return relief


if __name__ == '__main__':
    data, features = getdata()
    relief = Relief(data)
    #print(relief)
    print("特征排序：",features[np.array(relief).argsort()])
    print(relief)