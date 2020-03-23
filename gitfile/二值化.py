'''
    有些业务并不需要分析矩阵的详细完整数据（比如图像边缘识别只需要分析出图像边缘即可），可以根据一个事先给定的阈值，用0和1表示特征值不高于或高于阈值。二值化后的数组中每个元素非0即1，达到简化数学模型的目的。
'''

import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])

print(raw_samples)

bin_samples = raw_samples.copy()
bin_samples[bin_samples<=80]=0 #  小于80 的 赋值为 0
bin_samples[bin_samples>80] =1 # 大于80 的 赋值为1

print(bin_samples)

# 根据给定的阈值创建一个二值化器
bin = sp.Binarizer(threshold=80)  # 80  是一个阈值
bin_samples = bin.transform(raw_samples)
print(bin_samples)