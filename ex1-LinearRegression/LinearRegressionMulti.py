# -*- encoding:utf-8 -*-
# created by IvanChan

import numpy as np
import matplotlib.pyplot as plt


def features_norm(X):
    """
    Parameter
    ----------
    X : Numpy.ndarray, 输入特征

    Return
    ----------
    X_norm: numpy.ndarray, 标准化后的特征
    """
    X_norm = (X - X.mean(0)) / X.std(0)

    return X_norm
    


def compute_cost(X, y, theta):
    """
    Parameters
    ----------
    X: numpy.ndarray, 输入特征
    y: numpy.ndarry, 输入因变量
    theta: numpy.ndarry, 参数向量

    Returns
    ----------
    J : Float, 损失函数值
    """
    m = len(y)  # 计算样本个数 y.size/
    J = 1 / (2 * m) * ((X.dot(theta) - y) ** 2).sum()   # 和一元的计算方法一样
    # J = 1/(2*m) * ((X.dot(theta) - y).T).dot(X.dot(theta) - y)  # 形如matlab中使用向量化方法J = 1/(2*m)*(((X*theta-y).')*(X*theta-y))
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Parameters
    ----------
    X: numpy.ndarray, 标准化后的特征矩阵
    y: numpy.ndarray, 输入因变量
    theta: numpy.ndarray, 初始参数向量
    alpha: float, 学习速率
    num_iters： int, 迭代次数
    
    Returns
    --------
    theta: numpy.ndarray, 迭代计算后的估计参数
    J_history: numpy.ndarray, 历次迭代计算出来的损失函数值
    """
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):


        # matlab 代码：theta = theta - alpha / m * X' * (X * theta - y); 
        theta = theta - alpha / m * (X.T.dot(X.dot(theta) - y))
        J_history[iter] = compute_cost(X, y, theta)
    
    return theta, J_history


if __name__ == "__main__":