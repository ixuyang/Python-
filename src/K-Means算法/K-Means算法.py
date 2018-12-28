'''
Created on 2018年12月28日

@author: gy

k-means是一种最流行的聚类算法，属于无监督学习，没有标签列。
可以在数据集分为相似的组（簇），使得组内数据的相似度较高，组间之间的相似度较低。

算法步骤：
1、从样本中选择k个点作为初始簇中心。
2、计算每个样本到各个簇中心的距离，将样本划分为距离最近的簇中心所对应的簇中。
3、根据每个簇中的所有样本，重新计算簇中心，并更新。
4、重复步骤2与3，直到簇中心的位置变化小于指定的阈值或者达到最大迭代次数为止。

'''
# ①我们导入实验所必须的库。
 
import numpy as np
import pandas as pd
# ②然后，载入相关的数据集。
# 
data = pd.read_csv("order.csv")
t = data.iloc[:, -8:]
print(t)
# ③接下来，我们编写KMeans类的实现。在类中编写初始化方法，训练与测试方法等。
 
class KMeans:
    """使用Python语言实现聚类算法。"""
     
    def __init__(self, k, times):
        """初始化方法
         
        Parameters
        -----
        k : int
            聚类的个数
         
        times : int
            聚类迭代的次数
        """
         
        self.k = k
        self.times = times
         
    def fit(self, X):
        """根据提供的训练数据，对模型进行训练。
         
        Parameters
        -----
        X : 类数组类型，形状为：[样本数量， 特征数量]
            待训练的样本特征属性。
         
        """
        X = np.asarray(X)
        # 设置随机种子，以便于可以产生相同的随机序列。（随机的结果可以重现。）
        np.random.seed(0)
        # 从数组中随机选择k个点作为初始聚类中心。
        self.cluster_centers_ = X[np.random.randint(0, len(X), self.k)]
        self.labels_ = np.zeros(len(X))
         
        for t in range(self.times):
            for index, x in enumerate(X):
                # 计算每个样本与聚类中心的距离
                dis = np.sqrt(np.sum((x - self.cluster_centers_) ** 2, axis=1))
                # 将最小距离的索引赋值给标签数组。索引的值就是当前点所属的簇。范围为[0, k - 1]
                self.labels_[index] = dis.argmin()
            # 循环遍历每一个簇
            for i in range(self.k):
                # 计算每个簇内所有点的均值，更新聚类中心。
                self.cluster_centers_[i] = np.mean(X[self.labels_ == i], axis=0)
                 
#     def predict(self, X):
#         """根据参数传递的样本，对样本数据进行预测。（预测样本属于哪一个簇中）
#         
#         Parameters
#         -----
#         X : 类数组类型。 形状为: [样本数量， 特征数量]
#             待预测的特征属性。
#         
#         Returns
#         -----
#         result : 数组类型
#             预测的结果。每一个X所属的簇。
#         """
#         
#         X = np.asarray(X)
#         result = np.zeros(len(X))
#         for index, x in enumerate(X):
#             # 计算样本到每个聚类中心的距离。
#             dis = np.sqrt(np.sum((x - self.cluster_centers_) ** 2, axis=1))
#             # 找到距离最近的聚类中心，划分类别。
#             result[index] = dis.argmin()
#         return result
# ④对KMeans类进行训练。
# 
# kmeans = KMeans(3, 50)
# kmeans.fit(t)
# ⑤训练之后，查看聚类中心。
# 
# kmeans.cluster_centers_
# 程序运行结果如下：
# 
# array([[46.33977936,  8.93380516, 23.19047005, 13.11741633,  4.8107557 ,
#          1.17283735,  1.35704647,  0.95392773],
#        [19.5308009 , 50.42856608, 14.70652695,  7.89437019,  3.69829234,
#          0.91000428,  1.92515077,  0.82113238],
#        [ 7.93541008,  4.56182052, 30.65583437, 18.57726789,  8.61597195,
#          1.28482514, 26.81950293,  1.30158264]])
# ⑥选择几个测试数据，对结果进行预测。
# 
# kmeans.predict([[30, 30, 40, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 30, 30, 40], [30, 30, 0, 0, 0, 0, 20, 20]])
# 程序运行结果如下：
# 
# array([0., 2., 1.])
# ⑦为了方便的进行数据可视化，我们取数据集中的两列，训练数据。
# 
# t2 = data.loc[:, "Food%":"Fresh%"]
# kmeans = KMeans(3, 50)
# kmeans.fit(t2)
# ⑧导入可视化所必须的库。
# 
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rcParams["font.family"] = "SimHei"
# mpl.rcParams["axes.unicode_minus"] = False
# ⑨进行数据可视化显示。
# 
# plt.figure(figsize=(10, 10))
# # 绘制每个类别的散点图
# plt.scatter(t2[kmeans.labels_ == 0].iloc[:, 0], t2[kmeans.labels_ == 0].iloc[:, 1], label="类别1")
# plt.scatter(t2[kmeans.labels_ == 1].iloc[:, 0], t2[kmeans.labels_ == 1].iloc[:, 1], label="类别2")
# plt.scatter(t2[kmeans.labels_ == 2].iloc[:, 0], t2[kmeans.labels_ == 2].iloc[:, 1], label="类别3")
# # 绘制聚类中心
# plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker="+", s=300)
# plt.title("食物与肉类购买的聚类分析")
# plt.xlabel("食物")
# plt.ylabel("肉类")
# plt.legend()
# plt.show()
# ⑩程序运行结果如下：