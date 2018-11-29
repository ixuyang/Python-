'''
Created on 2018年11月29日

@author: UPXY
'''
import numpy as up
import pandas as pd

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


