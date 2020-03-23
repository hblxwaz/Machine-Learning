'''
    根据字符串形式的特征值在特征序列中的位置，为其指定一个数字标签，用于提供给基于数值算法的学习模型。
'''

import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    'audi', 'ford', 'audi', 'toyota',
    'ford', 'bmw', 'toyota', 'ford',
    'audi'])

print(raw_samples)
# 创建一个标签编码器
lbe = sp.LabelEncoder()
# 调用标签编码器的fit_transform方法训练并且为原始样本矩阵进行标签编码
lbe_samples= lbe.fit_transform(raw_samples)

print(lbe_samples)
# 根据标签编码的结果矩阵反查字典 得到原始数据矩阵
inv_samples = lbe.inverse_transform(lbe_samples)
print(inv_samples)