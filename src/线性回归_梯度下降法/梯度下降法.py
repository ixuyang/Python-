'''
Created on 2018年12月10日

@author: gy
'''
# ①导入程序需要的库。
# 
import numpy as np
import pandas as pd
# ②导入程序需要用到的数据集（波士顿房价）。并显示前5条记录。
 
data = pd.read_csv(r"boston.csv")
# 波士顿房价前5条记录
print(data.head())

# ③定义线性回归类。并实现初始化，训练与测试方法。训练（fit）方法中，使用梯度下降法逐渐调整权重。
class LinearRegression:
    """使用Python语言实现线性回归算法。（梯度下降）"""
     
    def __init__(self, alpha, times):
        """初始化方法。
         
        Parameters
        -----
        alpha : float
            学习率。用来控制步长。（权重调整的幅度）
             
        times : int
            循环迭代的次数。
        """
        self.alpha = alpha
        self.times = times
         
    def fit(self, X, y):
        """根据提供的训练数据，对模型进行训练。
         
        Parameters
        -----
        X : 类数组类型。形状：[样本数量， 特征数量]
            待训练的样本特征属性。（特征矩阵）
         
        y : 类数组类型。形状：[样本数量]
            目标值（标签信息）。
        """
         
        X = np.asarray(X)
        y = np.asarray(y)
        # 创建权重的向量，初始值为0（或任何其他的值），长度比特征数量多1，X.shape[1]为特征数量。（多出的一个值就是截距。）
        self.w_ = np.zeros(1 + X.shape[1])
        # 创建损失列表，用来保存每次迭代后的损失值。损失值计算： (预测值 - 真实值) 的平方和除以2
        self.loss_ = []
         
        # 进行循环，多次迭代。在每次迭代过程中，不断的去调整权重值，使得损失值不断减小。
        for i in range(self.times):
            # 计算预测值
            y_hat = np.dot(X, self.w_[1:]) + self.w_[0]
            # 计算真实值与预测值之间的差距。
            error = y - y_hat
            # 将损失值加入到损失列表当中。
            self.loss_.append(np.sum(error ** 2) / 2)
            # 根据差距调整权重w_，根据公式： 调整为  权重(j) =  权重(j) + 学习率 * sum((y - y_hat) * x(j))
            self.w_[0] += self.alpha * np.sum(error)
            self.w_[1:] += self.alpha * np.dot(X.T, error) 
             
    def predict(self, X):
        """根据参数传递的样本，对样本数据进行预测。
         
        Parameters
        -----
        X : 类数组类型，形状[样本数量， 特征数量]
            待测试的样本。
             
        Returns
        -----
        result : 数组类型。
            预测的结果。
        """
        X = np.asarray(X)
        result = np.dot(X, self.w_[1:]) + self.w_[0]
        return result
# ④创建线性回归类，对数据集洗牌，然后构建训练集与测试集，并对模型进行训练与测试，并显示结果。
 
lr = LinearRegression(alpha=0.0005, times=20)
t = data.sample(len(data), random_state=0)
train_X = t.iloc[:400, :-1]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, :-1]
test_y = t.iloc[400:, -1]
 
lr.fit(train_X, train_y)
result = lr.predict(test_X)
print(np.mean((result - test_y) ** 2))
print(lr.w_)
print(lr.loss_)
 
# 从运行结果可知，无论是权重与损失值，均非常不理想。这是由于不同特征列的数量级相差较大的缘故。因此，我们需要对数据进行标准化处理，进而取出不同量纲所带来的影响。
 
# ⑤定义标准化类。用于对特征列进行标准化处理。（将每个特征列数据变成标准正态分布的形式）
 
class StandardScaler:
    """该类对数据进行标准化处理。"""
     
    def fit(self, X):
        """根据传递的样本，计算每个特征列的均值与标准差。
         
        Parameters
        -----
        X : 类数组类型
            训练数据，用来计算均值与标准差。
        """
        X = np.asarray(X)
        self.std_ = np.std(X, axis=0)
        self.mean_ = np.mean(X, axis=0)
         
    def transform(self, X):
        """对给定的数据X，进行标准化处理。（将X的每一列都变成标准正态分布的数据）
         
        Parameters
        -----
        X : 类数组类型
            待转换的数据。
             
        Returns
        -----
        result : 类数组类型。
            参数X转换成标准正态分布后的结果。
        """
         
        return (X - self.mean_) / self.std_
     
    def fit_transform(self, X):
        """对数据进行训练，并转换，返回转换之后的结果。
         
        Parameters
        -----
        X : 类数组类型
            待转换的数据
             
        Returns
        -----
        result ： 类数组类型
            参数X转换成标准正态分布后的结果。
         
        """
        self.fit(X)
        return self.transform(X)
         
# ⑥重新定义线性回归类，并对数据进行标准化处理。并进行训练与测试。再次运行程序。
 
# 为了避免每个特征数量级的不同，从而在梯度下降的过程中带来影响，
# 我们现在考虑对每个特征进行标准化处理。
lr = LinearRegression(alpha=0.0005, times=20)
t = data.sample(len(data), random_state=0)
train_X = t.iloc[:400, :-1]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, :-1]
test_y = t.iloc[400:, -1]
 
# 对数据进行标准化处理
s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.transform(test_X)
 
s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)
 
lr.fit(train_X, train_y)
result = lr.predict(test_X)
print(np.mean((result - test_y) ** 2))
print(lr.w_)
print(lr.loss_)
 
# 经过数据标准化后，可以发现，运行结果优化很多。
# ⑦导入可视化库。
 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
# ⑧对预测值与真实值进行可视化显示。
 
plt.figure(figsize=(10, 10))
# 绘制预测值
plt.plot(result, "ro-", label="预测值")
# 绘制真实值
plt.plot(test_y.values, "go--",label="真实值")
plt.title("线性回归预测-梯度下降")
plt.xlabel("样本序号")
plt.ylabel("房价")
plt.legend()
plt.show()
 
# ⑨然后，我们绘制每次迭代过程中的累计误差值。
 
# 绘制累计误差值
plt.plot(range(1, lr.times + 1), lr.loss_, "o-")
plt.show()
 
# ⑩我们挑选两个特征，然后进行线性回归效果的拟合。注意，需要对数据进行标准化处理。
 
# 因为房价分析涉及多个维度，不方便进行可视化显示，为了实现可视化，
# 我们只选取其中的一个维度（RM），并画出直线，实现拟合。
lr = LinearRegression(alpha=0.0005, times=50)
t = data.sample(len(data), random_state=0)
train_X = t.iloc[:400, 5:6]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, 5:6]
test_y = t.iloc[400:, -1]
 
# 对数据进行标准化处理
s = StandardScaler()
train_X = s.fit_transform(train_X)
test_X = s.transform(test_X)
s2 = StandardScaler()
train_y = s2.fit_transform(train_y)
test_y = s2.transform(test_y)
 
lr.fit(train_X, train_y)
result = lr.predict(test_X)
print(np.mean((result - test_y) ** 2))
 
# ⑪绘制散点图与直线，查看拟合效果。
plt.scatter(train_X["RM"], train_y)

# 查看方程系数
print(lr.w_)
# 构建方程 y = -3.03757020e-16 + 6.54984608e-01 * x
x = np.arange(-5, 5, 0.1)
y = -3.03757020e-16 + 6.54984608e-01 * x
# plt.plot(x, y, "r")
# 也可以这样做
plt.plot(x, lr.predict(x.reshape(-1, 1)), "r")
plt.show()