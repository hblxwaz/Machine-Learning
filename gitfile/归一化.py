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

nor_samples = sp.normalize(raw_samples,norm='l2')
print(nor_samples)
print(abs(nor_samples).sum(axis=1))