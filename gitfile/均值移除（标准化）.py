import numpy as np
import sklearn.preprocessing as sp

data = np.array([[3, -1.5, 2, -5.4],
                 [0, 4, -0.3, 2.1],
                 [1, 3.3, -1.9, -4.3]])  # 原始数据矩阵 shape=(3,4)
data_standardized = sp.scale(data)  # 0均值处理
print(data_standardized.shape)
print('Mean={}'.format(data_standardized.mean(axis=0)))  # 对列求均值（浮点计算有误差，接近0）
print('Mean2={}'.format(np.mean(data_standardized, axis=0)))  # 对列求均值（浮点计算有误差，接近0）
# axis：int类型，初始值为0，axis用来计算均值 means 和标准方差 standard deviations.
# 如果是0，则单独的标准化每个特征（列），如果是1，则标准化每个观测样本（行）。
print('标准化后: ')
print(data_standardized)
print('标准差={}'.format(np.std(data_standardized, axis=0)))

ary = np.array([[12, 25, 77],
               [3, 11, 44],
               [4, 33, 22]
               ])

std_samples=  sp.scale(ary)

print(std_samples.mean(axis=0))
print(std_samples.std(axis=0))
