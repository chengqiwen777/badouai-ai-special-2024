'''
去除数据的单位限制，将其转化为无量纲的纯数值
将数据统一映射到 [0,1] 区间上
'''

import numpy as np
import matplotlib.pyplot as plt


def Normalization1(X):
    x = [(val - np.min(X)) / (np.max(X) - np.min(X)) for val in X]
    return x


def Normalization2(X):
    x = [(i - np.mean(X)) / (np.max(X) - np.min(X)) for i in X]
    return x


def Normalization3(X):
    x_mean = np.mean(X)
    x_std = np.sqrt(np.sum([(i - x_mean) * (i - x_mean) for i in X]) / len(X))
    x = [(i - x_mean) / x_std for i in X]
    return x


X = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]
cs = []
for i in X:
    cs.append(X.count(i))
print(cs)

nor1 = Normalization1(X)
nor2 = Normalization2(X)
nor3 = Normalization3(X)
print(nor3)

plt.figure()
plt.plot(X, cs, 'r')
plt.plot(nor1, cs, 'g')
plt.plot(nor2, cs, 'b')
plt.plot(nor3, cs)
plt.show()
