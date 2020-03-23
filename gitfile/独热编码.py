'''
    独热编码

    为样本特征的每个值建立一个由一个1和若干个0组成的序列，用该序列对所有的特征值进行编码

'''
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array([
    [17., 100., 4000],
    [20., 80., 5000],
    [23., 75., 5500]])


# sparse: 是否使用紧缩格式（稀疏矩阵）
# dtype: 数据类型


# 创建一个独热编码器
ohe  = sp.OneHotEncoder(sparse=False,dtype=int)
print(ohe)
# 使用独特编码器对原始样本矩阵做独热编码
# 对原始样本矩阵进行训练，得到编码字典
ohe_dict = ohe.fit(raw_samples)
print(ohe_dict)
# 调用encode_dict字典的transform方法 对数据样本矩阵进行独热编码
ohe_samples = ohe_dict.transform(raw_samples)
print(ohe_samples)

ohe_samples = ohe.fit_transform(raw_samples)
print(ohe_samples)
