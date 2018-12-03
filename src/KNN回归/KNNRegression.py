'''
Created on 2018年12月3日

@author: gy
'''
import numpy as np
import pandas as pd
from Tools.scripts.dutree import display

data = pd.read_csv(r"iris.csv")
# 删除不需要的Id与Species列（特征），因为现在进行回归预测，通过花的前三个属性，预测第四个，类别信息就没有用处了。
data.drop(["Id","Species"],axis=1,inplace=True)
print(data)
#删除重复的记录
data.drop_duplicates(inplace=True)

class KNN:
    """使用python实现k近邻算法。（回归预测）
    
    该算法用于回归预测，根据前3个特征属性，寻找最近的k个邻居，然后再根据k个邻居的第4个特征属性
    ，去预测当前样本的第4个特征值。
    
    """
    
    def __init__(self,k):
        """初始化方法
        
        Parameters
        -----
        k:int
邻居的个数。
    
        """
        self.k = k
    
    def fit(self,X,y):
        """训练方法。
        
        Parameters
        -----
        X：类数组类型（特征矩阵）。形状为[样本数量，特征数量]，形状类似二位数组或者矩阵
        待训练的样本特征（属性）
        
        y:类数组类型（目标标签）。形状为[样本数量]
        每个样本的目标值（标签）ps：这个地方将会对应第四个属性值，不需要死板的理解为只是花的类别，还有可能是花额的宽度等
        """
        # 注意，将X与y转化成ndarray数组的形式，方便统一进行操作。
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        
    def predict(self,X):
        """根据参数传递的X，对样本数据进行预测。
        
        Parameters
        -----
        X：类数组类型。形状为[样本数量，特征数量]
        待测试的样本特征（属性）
        
        Returns
        -----
        result:数组类型。
        预测的结果值
        """
        
        # 转换成数组类型
        X = np.asarray(X)
        # 保存预测的结果值。
        result = []
        for x in X:
            # 计算距离。（计算与训练集中每个X的距离）
            dis = np.sqrt(np.sum((x - self.X)**2,axis=1))
            # 返回数组排序后，每个元素在原数组中（排序之前的数组）的索引。
            index = dis.argsort()
            # 取前k个距离最近的索引（在原数组中的索引）。
            index = index[:self.k]
            # 计算均值，然后加入到结果列表当中。
            result.append(np.mean(self.y[index]))
        return np.array(result)
    
t = data.sample(len(data),random_state=0)
train_X = t.iloc[:120,:-1]
train_y = t.iloc[:120,-1]
test_X = t.iloc[120:,:-1]
test_y = t.iloc[120:,-1]
knn = KNN(k=3)
knn.fit(train_X,train_y)
result = knn.predict(test_X)
print(result)
# 这个地方对误差进行平方处理，避免正负误差抵消。
np.mean(np.sum((result - test_y) ** 2))
print(test_y.values)