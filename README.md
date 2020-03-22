# 机器学习

## 概述

### 什么是机器学习

机器学习是一门能够让编程计算机从数据中学习的计算机科学。
一个计算机程序在完成任务T之后，获得经验E，其表现效果为P，如果任务T的性能表现，也就是用以衡量的P，随着E增加而增加，那么这样计算机程序就被称为机器学习系统。
自我完善，自我增进，自我适应。

### 为什么需要机器学习

* 自动化的升级和维护
* 解决那些算法过于复杂甚至跟本就没有已知算法的问题

### 机器学习的问题

1. 建模问题
   所谓机器学习，在形式上可这样理解：在数据对象中通过统计或推理的方法，寻找一个接受特定输入X，并给出预期输出Y的功能函数f，即Y=f(X)。
2. 评估问题
   针对已知的输入，函数给出的输出(预测值)与实际输出(目标值)之间存在一定的误差，因此需要构建一个评估体系，根据误差的大小判定函数的优劣。
3. 优化问题
   学习的核心在于改善性能，通过数据对算法的反复锤炼，不断提升函数预测的准确性，直至获得能够满足实际需求的最优解，这个过程就是机器学习。

### 机器学习的种类

**监督学习、无监督学习、半监督学习、强化学习**

1. 有监督学习：用已知输出评估模型的性能。
2. 无监督学习：在没有已知输出的情况下，仅仅根据输入信息的相关性，进行类别的划分。
3. 半监督学习：先通过无监督学习划分类别，再根据人工标记通过有监督学习预测输出。
4. 强化学习：通过对不同决策结果的奖励和惩罚，使机器学习系统在经过足够长时间的训练以后，越来越倾向于给出接近期望结果的输出。

**批量学习和增量学习**

1. 批量学习：将学习的过程和应用的过程截然分开，用全部的训练数据训练模型，然后再在应用场景中实现预测，当预测结果不够理想时，重新回到学习过程，如此循环。
2. 增量学习：将学习的过程和应用的过程统一起来，在应用的同时以增量的方式，不断学习新的内容，边训练边预测。