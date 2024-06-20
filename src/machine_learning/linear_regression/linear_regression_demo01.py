# Linear Regression（线性回归）
# 线性回归API
# sklearn.linear_model.LinearRegression()
#   LinearRegression.coef_:回归系数
#   fit_intercept: 是否计算偏置
"""
1. 获取数据集
2. 数据基本处理
3. 特征工程
4. 机器学习
5. 模型评估
"""
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# 1. 获取数据
x = [[80, 86],
     [82, 80],
     [85, 78],
     [90, 90],
     [86, 82],
     [82, 90],
     [78, 80],
     [92, 94]]  # x通常是二维数组
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]  # y通常是一维数组

# 2. 模型训练
# 2.1 实例化一个估计器
estimator = LinearRegression()
# 2.2 调用fit方法，进行模型训练
estimator.fit(x, y)
# 查看下系数值
result = estimator.coef_
print("系数是：\n", result)

# 预测(平时考80，期末考100)
print("预测值是：\n", estimator.predict([[80, 100]]))

# 1 线性回归举例
x = np.linspace(0, 10, 50)

y = 0.8 * x - 5
plt.plot(x, y, color='red')

# np.linalg.solve(X, y)
