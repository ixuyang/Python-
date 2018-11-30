'''
Created on 2018年11月29日

@author: UPXY
'''
import numpy as np
import pandas as pd
from Tools.scripts.dutree import display
from numpy.lib.function_base import disp

# 读取鸢尾花数据集，header参数来指定标题的行，默认为0，如果没有标题则使用None。
data = pd.read_csv(r"iris.csv",header=0)
# head（）默认显示前五行，也可以认为的确定显示多少行。
# print(data.head(10))
# 显示末尾的数据使用tail()
# print(data.tail(10))
# 随机抽取样本，默认抽取一条，可以指定样本数量n。
# print(data.sample(10))
# 对数据中的分类标签进行数值化映射转换
data["Species"] = data["Species"].map({"versicolor":0,"setosa":1,"virginica":2})
# 删除不需要的Id列,axis=1表示以列的方式删除,inplace表示对原有的数据进行替换处理
data.drop("Id",axis=1,inplace=True)
# duplicated()表示是否有重复值，True表示有重复值。
print(data.duplicated().any())
# 对重复数据进行删除
print(len(data))
data.drop_duplicates(inplace=True)
print(len(data))
# 查看各个类别鸢尾花具有多少个记录
print(data["Species"].value_counts())
print(data)

class KNN:
    """使用Python语言实现k近邻算法。（实现分类）"""
    
    def __init__(self,k):
        """初始化方法
        
        Parameters
        -----
        k:int 邻居的个数。
        """
        self.k = k
        
    def fit(self,X,y):
        """训练的方法
        
        Parameters
        -----
        X:类数组类型，（通常矩阵类变量要大写），形状为：[样本数量，特征数量]
待训练的样本特征（属性）
        y:类数组类型，形状为：[样本数量]
每个样本的目标值（标签）
        """
        
        # 将X转换成ndarray数组类型。
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        
    def predict(self,X):
        """根据参数传递的样本，对样本数据进行预测
        
        Parameters
        -----
        X:类数组类型，（通常矩阵类变量要大写），形状为：[样本数量，特征数量]
待训练的样本特征（属性）

        Returns
        -----
        result:数组类型
预测的结果。
        """
        
        X = np.asarray(X)
        result = []
        # 对ndarray数组进行遍历，每次取数据中的一行
        for x in X:
            # 改地方的减法会对x进行广播复制，扩展和X行数一样。求差和平方开根号用于计算距离。sum要确定好求和的方式，不是全部，而是一行一行的求和。
            dis = np.sqrt(np.sum((x - self.X)**2,axis=1))
            # 返回数组排序后，每个元素在原数组（排序之前的数组）中的索引。
            index  = dis.argsort()
            # 进行截断，只取前k个元素。【取距离最近的k个元素的索引】
            index = index[:self.k]
            # 返回数组中的每个元素出现的次数，袁术必须是非负的整数。
            count = np.bincount(self.y[index].astype(int))
            # 返回ndarray数组中，值最大的元素对应的索引。该索引就是我们判定的类别。
            # 最大元素索引，就是出现次数最多的元素
            result.append(count.argmax())
            
        return np.asarray(result)
    
# 提取出每个类别的鸢尾花数据
t0 = data[data["Species"] == 0]  
t1 = data[data["Species"] == 1]   
t2 = data[data["Species"] == 2]  
# 对每个类别数据进行洗牌。
t0 = t0.sample(len(t0),random_state=0)
t1 = t1.sample(len(t1),random_state=0)
t2 = t2.sample(len(t2),random_state=0)    
# 构建训练集和测试集。
train_X = pd.concat([t0.iloc[:40,:-1],t1.iloc[:40,:-1],t2.iloc[:40,:-1]],axis=0)
train_y = pd.concat([t0.iloc[:40,-1],t1.iloc[:40,-1],t2.iloc[:40,-1]],axis=0)
test_X = pd.concat([t0.iloc[40:,:-1],t1.iloc[40:,:-1],t2.iloc[40:,:-1]],axis=0)
test_y = pd.concat([t0.iloc[40:,-1],t1.iloc[40:,-1],t2.iloc[40:,-1]],axis=0)
# 创建KNN对象，进行训练与测试。
knn = KNN(k=3)
# 进行训练。
knn.fit(train_X, train_y)
# 进行测试，获得测试的结果。
result = knn.predict(test_X)
# display(result)
# display(test_y)
print(np.sum(result == test_y))
print(np.sum(result == test_y)/len(result))
