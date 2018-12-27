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
    compute cost function

    Parameters
    ----------
    X: numpy.ndarray, 输入特征
    y: numpy.ndarry, 输入因变量
    theta: numpy.ndarry, 参数向量

    Returns
    ----------
    j : Float, 损失函数值
    """
    m = len(y)  # 计算样本个数 y.size/
    j = 1 / (2 * m) * ((X.dot(theta) - y) ** 2).sum()   # 和一元的计算方法一样
    # J = 1/(2*m) * ((X.dot(theta) - y).T).dot(X.dot(theta) - y)  # 形如matlab中使用向量化方法J = 1/(2*m)*(((X*theta-y).')*(X*theta-y))
    return j


def gradient_descent(X, y, theta, alpha, iters):
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
    J_history = np.zeros(iters)
    for i in range(iters):


        # matlab 代码：theta = theta - alpha / m * X' * (X * theta - y); 
        theta = theta - alpha / m * (X.T.dot(X.dot(theta) - y))
        J_history[i] = compute_cost(X, y, theta)
    
    return theta, J_history


def norm_equation(X, y):
    """
    Normal Equation

    Parameters
    ----------

    :param X: np.array
    :param y: np.array

    Return
    ----------
    :return: theta numpy.array
    """

    theta = (np.linalg.pinv(X.T.dot(X)).dot(X.T)).dot(y)  # matlab代码 theta = pinv(X.'*X)*X.'*y
    return theta


if __name__ == "__main__":
    """
    运行多元线性回归
    """

    data = np.loadtxt('ex1data2.txt', delimiter=',')
    x_mat = data[:, 0:2]
    y_ob = data[:, 2]
    m = len(y_ob)  # x_mat.shape[0]

    """------------------使用梯度下降gradient descent算法求参数"""

    x_mat_norm = features_norm(x_mat)  # 调用特征归一化函数
    x_mat_i = np.vstack((np.ones(m), x_mat_norm.T))
    x_mat_p = x_mat_i.T

    alpha = 0.01
    num_iters = 400

    theta_z = np.zeros(3)

    theta, j_h = gradient_descent(x_mat_p, y_ob, theta_z, alpha, num_iters)
    print(theta)

    fig, ax =plt.subplots(1,1)
    ax.plot(np.array(range(num_iters)), j_h, 'b-')
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Cost Function J Value")
    plt.show()

    """使用正规方程Normal Equation 求解参数"""

    x_matn = data[:, 0:2]
    x_matn_i = np.vstack((np.ones(m), x_matn.T))
    x_matn_p = x_matn_i.T
    theta = norm_equation(x_matn_p, y_ob)
    print(theta)