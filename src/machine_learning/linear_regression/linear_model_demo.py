"""
1. 获取数据
2. 数据基本处理
2.1 数据集划分
3. 特征工程 --标准化
4. 机器学习（线性回归）
5. 模型评估
"""
# from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


def linear_model1():
    """
    正规方程
    :return: None
    """
    # 1.获取数据
    boston = fetch_california_housing()

    # 2.数据基本处理
    # 2.1数据集划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    # 3.特征工程 --标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习（线性回归）
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 5.1 预测值和准确率
    y_pred = estimator.predict(x_test)
    print('预测值是：\n', y_pred)

    score = estimator.score(x_test, y_test)
    print('准确率是：\n', score)

    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pred)
    print('均方误差是：\n', ret)


def linear_model2():
    """
    梯度下降法
    :return: None
    """
    # 1.获取数据
    boston = fetch_california_housing()

    # 2.数据基本处理
    # 2.1数据集划分
    x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2)

    # 3.特征工程 --标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.fit_transform(x_test)

    # 4.机器学习（线性回归）
    estimator = SGDRegressor()
    estimator.fit(x_train, y_train)

    # 5.模型评估
    # 5.1 预测值和准确率
    y_pred = estimator.predict(x_test)
    print('预测值是：\n', y_pred)

    score = estimator.score(x_test, y_test)
    print('准确率是：\n', score)

    # 5.2 均方误差
    ret = mean_squared_error(y_test, y_pred)
    print('均方误差是：\n', ret)


if __name__ == '__main__':
    linear_model1()
    linear_model2()
