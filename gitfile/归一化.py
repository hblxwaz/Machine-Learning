
'''

有些情况每个样本的每个特征值具体的值并不重要，但是每个样本特征值的占比更加重要
所以归一化即是用每个样本的每个特征值除以该样本各个特征值绝对值的总和。变换后的样本矩阵，每个样本的特征值绝对值之和为1。
'''
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])
print(raw_samples)

nor_samples = raw_samples.copy() # 复制一份样本

for row in nor_samples:
    print(row)
    abs_sum = abs(row).sum() # 取行元素 绝对值 和
    print(abs_sum)
    row /= abs_sum # 行元素除以 行元素约对值和

    print(row)
print(nor_samples)
# 归一化处理
# 归一化相关API
nor_samples = sp.normalize(raw_samples,norm='l2')
print(nor_samples)
print(abs(nor_samples).sum(axis=1))