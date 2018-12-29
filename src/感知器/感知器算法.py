'''
Created on 2018年12月29日

@author: gy

感知器是一种人工神经网络，其模拟生物上的神经元结构。

感知器是一个二分类器。
'''
# ①我们导入实验程序所必须用到的包。
# 
import numpy as np
import pandas as pd
# ②然后，我们来对数据集进行读取，并去掉不需要的列，执行去重与映射，过滤等操作。
# 
data = pd.read_csv(r"Iris.csv")
# data.head()
data.drop("Id", axis=1, inplace=True)
data.drop_duplicates(inplace=True)
# 之所以映射为1与-1,而不是之前的0,1,2，是因为感知器的预测结果为1与-1。
# 目的是为了与感知器预测的结果相符。
data["Species"] = data["Species"].map({"Iris-versicolor": 0, "Iris-virginica": 1, "Iris-setosa": -1})
# data["Species"].value_counts()
#删除掉等于0的类别
data = data[data["Species"] != 0]
print(data)
# ③定义感知器类，并在类的内部定义初始化方法，阶跃函数，训练与测试方法。
 
class Perceptron:
    """使用Python语言实现感知器算法，实现二分类。"""
     
    def __init__(self, alpha, times):
        """初始化方法。
         
        Parameters
        -----
        alpha : float
            学习率。
             
        times : int
            最大迭代次数
         
        """
        self.alpha = alpha
        self.times = times
         
    def step(self, z):
        """阶跃函数。
         
         
        Parameters
        -----
        z : 数组类型（或者是标量类型）
            阶跃函数的参数。可以根据z的值，返回1或-1（这样就可以实现二分类）。
             
        Returns
        -----
        value : int
            如果z >= 0,返回1， 否则返回-1。
        """
         
#         return 1 if z >=0 else 0
        return np.where(z >= 0, 1, -1)
 
    def fit(self, X, y):
        """根据提供的训练数据，对模型进行训练。
         
        Parameters
        -----
        X : 类数组类型。形状：[样本数量，特征数量]
            待训练的样本数据。
             
        y : 类数组类型。 形状： [样本数量]
            每个样本的目标值。（分类）
         
         
        """
        X = np.asarray(X)
        y = np.asarray(y)
        # 创建权重的向量，初始值为0.长度比特征多1.（多出的一个就是截距）。
        self.w_ = np.zeros(1 + X.shape[1])
        #创建损失列表，用来保存每次迭代后的损失值。
        self.loss_ = []
        # 循环指定的次数。
        for i in range(self.times):
            #  感知器与逻辑回归的区别：逻辑回归中，使用所有样本计算梯度，然后更新权重。
            # 而感知器中，是使用单个样本，依次进行计算梯度，更新权重。
            loss = 0
            for x, target in zip(X, y):
                # 计算预测值
                y_hat = self.step(np.dot(x, self.w_[1:]) + self.w_[0])
                loss += y_hat != target
                # 更新权重。
                # 更新公式： w(j) = w(j) +  学习率 * （真实值 - 预测值） * x(j)
                self.w_[0] += self.alpha * (target - y_hat)
                self.w_[1:] += self.alpha * (target - y_hat) * x
            # 将循环中累计的误差值增加到误差列表当中。
            self.loss_.append(loss)
             
    def predict(self, X):
        """根据参数传递的样本，对样本数据进行预测。（1或-1）
         
        Parameters
        -----
        X : 类数组类型， 形状为：[样本数量， 特征数量]
            待预测的样本特征。
             
        Returns
        -----
        result : 数组类型
            预测的结果值（分类值1或-1）
         
        """
         
        return self.step(np.dot(X, self.w_[1:]) + self.w_[0])
# ④对两类鸢尾花数据集进行提取与洗牌，然后使用pd.concat合并，构建训练集与测试集。然后创建感知器类，调用训练方法训练，并且调用测试方法测试。显示预测结果与每次迭代的误差损失值。
 
t1 = data[data["Species"] == 1]
t2 = data[data["Species"] == -1]
t1 = t1.sample(len(t1), random_state=0)
t2 = t2.sample(len(t2), random_state=0)
train_X = pd.concat([t1.iloc[:40, :-1], t2.iloc[:40, :-1]], axis=0)
train_y = pd.concat([t1.iloc[:40, -1], t2.iloc[:40, -1]], axis=0)
test_X = pd.concat([t1.iloc[40:, :-1], t2.iloc[40:, :-1]], axis=0)
test_y = pd.concat([t1.iloc[40:, -1], t2.iloc[40:, -1]], axis=0)
p = Perceptron(0.1, 10)
p.fit(train_X, train_y)
result = p.predict(test_X)
print(result)
print(test_y.values)
print(p.w_)
print(p.loss_)
# 程序运行结果如下：
# 
# array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1])
# array([ 1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
#       dtype=int64)
# array([-0.4 , -0.44, -1.44,  1.88,  0.72])
# [1, 2, 3, 2, 0, 0, 0, 0, 0, 0]
# 通过运行结果可知，我们预测正确了所有分类，并且误差值在迭代5次后，就已经为0。
# 
# ⑤导入可视化所必须的库，并进行相关设置，使其能够支持中文与负号。
 
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
# ⑥将之前预测的结果进行可视化的显示。
# 
# # 绘制真实值
plt.plot(test_y.values, "go", ms=15, label="真实值")
# # 绘制预测值
plt.plot(result, "rx", ms=15, label="预测值")
plt.title("感知器二分类")
plt.xlabel("样本序号")
plt.ylabel("类别")
plt.legend()
plt.show()
# ⑦绘制目标函数损失值，在迭代5次后，损失为0。
# 
# # 绘制目标函数的损失值。
# plt.plot(range(1, p.times + 1), p.loss_, "o-")