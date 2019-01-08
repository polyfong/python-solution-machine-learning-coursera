# -*- coding:utf-8 -*-
# created by IvanChan

import scipy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import pandas as pd


def plot_data(X, y):
    """
    Parameters:
    -----------
    :param X: numpy array, 特征矩阵
    :param y: numpy array, 因变量

    Return:
    ----------
    :return: none
    """

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    pos_feature1 = X[y == 1][:, 0]  # 布尔索引
    pos_feature2 = X[y == 1][:, 1]
    neg_feature1 = X[y == 0][:, 0]
    neg_feature2 = X[y == 0][:, 1]
    ax.plot(pos_feature1, pos_feature2, 'k+', label = 'Admitted')
    ax.plot(neg_feature1, neg_feature2, 'yo', label = 'Not Admitted')
    # ax.plot(X[y == 1][:, 0], X[y == 1][:, 1], 'k+', X[y == 0][:, 0], X[y == 0][:, 1], 'ko')
    ax.set_xlabel('Exam 1 score')
    ax.set_ylabel('Exam 2 score')
    ax.legend()
    return fig, ax



def sigmoid(z):
    """
    Parameters:
    ----------
    :param z: int、float or numpy array, 输入变量（可为标量、向量、矩阵）

    Return:
    ----------
    :return g: float, S函数值
    """
    g = 1/(1 + np.exp(-z))
    return g


def cost_function(theta, X, y):
    """
    Parameters:
    ----------
    :param X:
    :param y:
    :param theta:

    Returns:
    ----------
    :return j:
    :return grad:
    """
    m_f, n_f = X.shape

    j = (-1/m_f) * (y.dot(np.log(sigmoid(X.dot(theta)))) + (1 - y).dot(np.log(1 - sigmoid(X.dot(theta)))))
    # MatLab code: J = (-1/m)*sum(y.*log(sigmoid(X*theta)) +(1-y).*log(1-sigmoid(X*theta))); 使用向量化计算方式更方便

    grad = (1/m_f) * (X.T.dot((sigmoid(X.dot(theta)) - y)))
    # MatLab code: grad = (1/m) .* (X' * ((sigmoid(X*theta)) - y));

    return j, grad


def map_feature(x1, x2, degree=6):
    """
    多项式
    :param x1:
    :param x2:
    :param degree:
    :return:
    """

    quads = pd.Series([x1**(i-j) * x2**j for i in range(1,degree+1) for j in range(i+1)])
    return pd.Series([1]).append([pd.Series(x1), pd.Series(x2), quads])


def plot_decision_boundary(theta, X, y):
    """

    Parameters:
    ----------
    :param theta:
    :param X:
    :param y:

    Return:
    ----------
    :return: none
    """

    """显示散点"""
    fig, ax = plot_data(X[:, 1:], y)

    """画出决策边界"""
    if X.shape[1] <= 3:
        plot_x = np.array([min(X[:, 2])-2, max(X[:, 2])+2])
        plot_y = (-1/theta[2]) * (theta[1] * plot_x + theta[0])
        ax.plot(plot_x, plot_y)
    else:

        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = [
            np.array([map_feature(u[i], v[j]).dot(theta) for i in range(len(u))]) for j in range(len(v))
            ]
        plt.contour(u,v,z, levels=[0.0])


def predict(theta, X):
    """

    :param theta:
    :param X:
    :return:
    """
    m_e = X.shape[0]

    p = sigmoid(X.dot(theta))

    for i in range(m_e):
        if p[i] >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p


def map_feature_reg(X, degree=6):
    """

    :param X:
    :param degree:
    :return:
    """
    quads = pd.Series([X[0]**(i-j) * X.iloc[1]**j for i in range(1,degree+1) for j in range(i+1)])
    return pd.Series([1]).append([X,quads])


def cost_function_reg(theta, X, y, lamb):
    """

    :param theta:
    :param X:
    :param y:
    :param lamb:
    :return:
    """


    m_f = X.shape[0]
    theta_t = np.array([0, theta[1], theta[2]])  # theta[0]不需正则化，将theta[0]设为0，使用向量化计算

    j = (-1/m_f) * (y.dot(np.log(sigmoid(X.dot(theta)))) + (1 - y).dot(np.log(1 - sigmoid(X.dot(theta))))) + (lamb/(2*m_f)) * theta_t.dot(theta_t)

    grad = (1/m_f) * (X.T.dot((sigmoid(X.dot(theta)) - y))) + (lamb/m_f) * theta_t

    return j, grad


if __name__ == "__main__":
    """
    使用逻辑回归算法
    """

    """加载数据"""
    data = np.loadtxt('ex2data1.txt', delimiter=',')
    x_mat = data[:, 0:2]
    y_ob = data[:, 2]

    """显示测试数据"""
    plot_data(x_mat, y_ob)
    plt.show()

    """计算损失函数和梯度"""
    m, n = x_mat.shape
    x_mat_t = np.hstack((np.ones((m, 1)), x_mat))

    initial_theta = np.zeros(n+1)

    cost_zero, gradient_zero = cost_function(initial_theta, x_mat_t, y_ob)

    print("初始theta 为0时，损失函数J值(应接近0.639）：", cost_zero)
    print("梯度值为（接近[-0.1, -12.0092, -11.2628]：", gradient_zero)

    test_theta = np.array([-24, 0.2, 0.2])
    cost_test, gradient_test = cost_function(test_theta, x_mat_t, y_ob)

    print("测试theta 为[-24, 0.2, 0.2], 损失函数J值（接近0.218）:", cost_test)
    print("梯度值为（接近[0.043, 2.566, 2.647]）：", gradient_test)

    """计算最优解"""
    # 使用scipy.optimize模块代替matlab中fminunc函数

    optimize_result = opt.minimize(fun=cost_function, x0=initial_theta, args=(x_mat_t, y_ob), method='TNC', jac=True)
    print(optimize_result)
    print('theta 初始为0时，最优化解:', optimize_result.x)

    """画出决策边界"""
    plot_decision_boundary(optimize_result.x, x_mat_t, y_ob)
    plt.show()

    """分类并计算模型准确性"""
    theta_opt = optimize_result.x
    prob = sigmoid(np.array([1, 45, 85]).dot(theta_opt))
    print()

    """正则化"""

    """加载并可视化数据"""
    data_reg = np.loadtxt("ex2data2.txt", delimiter=',')
    x_reg = data_reg[:, 0:2]
    y_reg = data_reg[:, 2]

    plot_data(x_reg, y_reg)
    plt.show()

    """正则化逻辑回归"""

    x_reg_map = map_feature_reg(x_reg)
    l_lambda = 1
    cost_reg, grad_reg = cost_function_reg(initial_theta, x_reg_map, y_reg, lamb=l_lambda)

    print("正则化逻辑回归：\n")
    print("theta为0时，损失函数值（接近0.693）：", cost_reg)
    print("梯度值（接近0.0085，0.0188，0.0001，0.0503，0.0115）：", grad_reg[0:5])




