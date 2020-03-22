'''
    范围缩放
'''

# 将样本矩阵中的每一列的最小值和最大值设定为相同的区间，统一各列特征值的范围。一般情况下会把特征值缩放至[0, 1]区间

import numpy

ary = numpy.array([[13, 3, 26],
                  [12, 4, 27],
                  [11, 5, 28]])
# 如何使一组特征值的最小值为0呢？
# 每个元素减去特征值数组所有元素的最小值即可
min = ary.min(axis=0)
print(ary - min)
# 如何使一组特征值的最大值为1呢？
# 把特征值数组的每个元素除以最大值即可
max = ary.max(axis=0)
print(ary / max)

# 范围缩放API

import sklearn.preprocessing as sp
# 创建MinMax 缩放器
mms = sp.MinMaxScaler(feature_range=(0, 1))

# 调用mms  对象的方法执行缩放操作，返回缩放过后的结果
result = mms.fit_transform(ary)
print(result)