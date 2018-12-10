'''
Created on 2018年12月6日

@author: UPXY
'''
# ①导入程序所必须的库。
import numpy as np
import pandas as pd

# ②加载波士顿房价数据集。并查看数据集的基本信息。
# 波士顿放假数据集字段说明 
"""CRIM 房屋所在镇的犯罪率
ZN 面积大于25000平方英尺住宅所占的比例
INDUS 房屋所在镇非零售区域所占比例
CHAS 房屋是否位于河边，如果位于河边则值为1，否则值为0
NOX 一氧化碳浓度
RM 平均房间数量
AGE 1940年全建成房屋所占的比例
DIS 房屋距离波士顿五大就业中心的加权距离
RAD 距离房屋最近的公路
TAX 财产税额度
PTRATTO 房屋所在镇师生比例
B 计算公式：1000*（房屋所在镇非美籍人口所在比例 - 0.63）**2
LSTAT 弱势群体所占比例
MEDV 房屋的平局价格
"""
data = pd.read_csv(r"boston.csv")
print(data)
# 查看数据的基本信息.同时，也可以用来查看，各个特征列是否存在缺失值。
data.info()
# 查看是否具有重复值。
Dup = data.duplicated().any()
print(Dup)
# ③定义线性回归类，并在类中实现训练与预测方法。
 
class LinearRegression:
    """使用Python实现的线性回归。（最小二乘法）"""
     
    def fit(self, X, y):
        """根据提供的训练数据X，对模型进行训练。
         
        Parameters
        -----
        X : 类数组类型。形状： [样本数量， 特征数量]
            特征矩阵，用来对模型进行训练。
             
        y : 类数组类型，形状： [样本数量]
         
        """
        
        # 说明：如果X是数组对象的一部分，而不是完整的对象数据（例如，X是由其他对象通过切片传递过来），
        # 则无法完成矩阵的转换。
        # 这里创建X的拷贝对象，拷贝后为完整的数组对象，避免转换矩阵的时候失败。
        X = np.asmatrix(X.copy())
        # y是一维结构（行向量或列向量），一维结构可以不用进行拷贝。
        # 注意：我们现在要进行矩阵的运算，因此需要是二维的结构，我们通过reshape方法进行转换，-1表示根据行数自动处理。
        y = np.asmatrix(y).reshape(-1, 1)
        # 通过最小二乘公式，求解出最佳的权重值，w标识weight，T表示转置，注意是矩阵的乘法。
        self.w_ = (X.T * X).I * X.T * y
         
    def predict(self, X):
        """根据参数传递的样本X，对样本数据进行预测。
         
        Parameters
        -----
        X : 类数组类型。形状： [样本数量， 特征数量]
            待预测的样本特征（属性）。
             
        Returns
        -----
        result : 数组类型
            预测的结果。
         
        """
        # 将X转换成矩阵，注意，需要对X进行拷贝。
        X = np.asmatrix(X.copy())
        result = X * self.w_
        # 将矩阵转换成ndarray数组，进行扁平化处理，然后返回结果。
        # 使用ravel可以将数组进行扁平化处理。
        return np.array(result).ravel()
     
# ④在不考虑截距的情况下，构建训练集与测试集。并创建线性回归类对象，进行训练与预测。并显示均方误差与权重值。
# 不考虑截距的情况
t = data.sample(len(data), random_state=0)
train_X = t.iloc[:400, :-1]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, :-1]
test_y = t.iloc[400:, -1]
 
lr = LinearRegression()
lr.fit(train_X, train_y)
result = lr.predict(test_X)
# result
print(np.mean((result - test_y) ** 2))
# 查看模型的权重值
print(lr.w_)
 
# ⑤在考虑截距的情况下，重新构建训练集与测试集。
# 考虑截距，增加一列，该列的所有值都是1，使得W0乘以X0的值为原W0。
t = data.sample(len(data), random_state=0)
# 可以这样增加一列。
# t["Intercept"] = 1
# 按照习惯，截距作为w0，我们为之而配上一个x0，x0列放在最前面。
new_columns = t.columns.insert(0, "Intercept")
# 重新安排列的顺序，如果值为空，则使用fill_value参数指定的值进行填充。
t = t.reindex(columns=new_columns, fill_value=1)
# t["Intercept"] = 1
print(t)
train_X = t.iloc[:400, :-1]
train_y = t.iloc[:400, -1]
test_X = t.iloc[400:, :-1]
test_y = t.iloc[400:, -1]
 
lr = LinearRegression()
lr.fit(train_X, train_y)
result = lr.predict(test_X)
# result
print(np.mean((result - test_y) ** 2))
print(lr.w_)
 
# ⑥导入可视化库，进行数据可视化。
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False
 
# ⑦绘制预测值与真实值，对比二值之间的差距。
plt.figure(figsize=(10, 10))
# 绘制预测值
plt.plot(result, "ro-", label="预测值")
# 绘制真实值
plt.plot(test_y.values, "go--", label="真实值")
plt.title("线性回归预测-最小二乘法")
plt.xlabel("样本序号")
plt.ylabel("房价")
# 用于生成图例
plt.legend()
plt.show()
