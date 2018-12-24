# created by IvanChan

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_data(x_var, y_var):
    """
        显示变量散点图

    Parameters:
    --------
    x_var: numpy.ndarray
        x变量
    y_var: numpy.ndarray
        y因变量
    :return:fig, ax

    """
    fig, ax = plt.subplots(1,1)
    ax.scatter(x_var, y_var, s=10, label='Training data')
    ax.set_xlabel("Population of the city in 10,000s")
    ax.set_ylabel("Profit in 10,000s")
    return fig, ax


def compute_cost_func(x_f, y_v, theta):
    """
        损失函数计算
    :param x_f: np.ndarray
        特征矩阵
    :param y_v: np.ndarray
        y
    :param theta: np.ndarray
        theta向量
    :return: j: dtype.float
        成本函数值
    """
    m_e = len(y_v)  # 样本个数
    j = 1/(2*m_e)*((x_f.dot(theta) - y_v) ** 2).sum()
    # 尽量和 matlab 的语句相似 J = 1/(2*m)*sum((X*theta-y).^2);
    return j


def gradient_descent(x_f, y_v, theta, alpha, num_iters):
    """

    :param x_f: ndarray, 特征矩阵
    :param y_v: ndarray, 因变量
    :param theta: ndarray，theta初始化参数
    :param alpha: float， 学习速率learning rate
    :param num_iters: int， 迭代次数 iterations
    :return: theta：array，更新后的theta值
            j_history: array, 每次迭代的计算的成本函数值
    """

    m_e = len(y_v)
    j_history = np.zeros(num_iters)

    for i in range(num_iters):

        theta_t = theta # 设置一个临时变量，保存该次迭代之前的theta值

        # matlab/octave语句 ：
        # theta(1) = theta(1) - alpha / m * sum((X * theta_s - y). * X(:, 1));
        # theta(2) = theta(2) - alpha / m * sum((X * theta_s - y). * X(:, 2));

        theta[0] = theta[0] - alpha/m_e * ((x_f.dot(theta_t) - y_v) * x_f[:,0]).sum()
        theta[1] = theta[1] - alpha/m_e * ((x_f.dot(theta_t) - y_v) * x_f[:,1]).sum()

        j_history[i] = compute_cost_func(x_f, y_v, theta)  # 计算并保存历次迭代的损失函数值

    return theta, j_history


if __name__ == '__main__':
    # 类似octave中ex1.m

    # 载入数据 ex1data1.txt是课程给的练习数据 画出变量散点图
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    m = len(y)
    print("Plotting data\n")
    fig, ax = plot_data(x, y)
    fig.savefig('plot_data.png')

    # 计算损失函数 costfunction值 J
    x_mat = np.vstack((np.ones(m), x))
    x_mat = x_mat.T
    theta_z = np.zeros(2)

    print("测试计算costFunction：\n当theta = [0,0]时，损失函数值：\n", compute_cost_func(x_mat, y, theta_z))
    print("当theta = [-1, 2]时，损失函数值：\n", compute_cost_func(x_mat, y, [-1, 2]))

    # 利用梯度下降算法 gradient descent 计算theta值
    iterations = 1500
    alpha = 0.01
    theta, j_h = gradient_descent(x_mat, y, theta_z, alpha, iterations)
    print("当costFunction最小时，theta = ", theta)

    # 可视化Linear Regression模型
    ax.plot(x, x_mat.dot(theta), 'b-', label='Linear Regression')
    ax.legend(numpoints=1, loc=0)
    fig.savefig('Linear-Regression.png')

    # 测试模型
    predict1 = np.array([1, 3.5]).dot(theta)
    predict2 = np.array([1, 7]).dot(theta)
    print("当人口为35，000时，预测盈利为：", predict1*10000)
    print("当人口为70，000时，预测盈利为：", predict2*10000)

    # 可视化损失函数costFunction计算结果值J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    j_vals = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            j_vals[i][j] = compute_cost_func(x_mat, y, t)
    j_vals = j_vals.T
    theta0_mesh, theta1_mesh = np.meshgrid(theta0_vals, theta1_vals)

    fig_costFunction = plt.figure()
    ax_costFunction = fig_costFunction.add_subplot(121, projection='3d')
    ax_costFunction.plot_surface(theta0_mesh, theta1_mesh, j_vals, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0, antialiased=False)
    ax_costFunction.set_xlabel(r'$\theta_0$', labelpad=5)
    ax_costFunction.set_ylabel(r'$\theta_1$', labelpad=5)
    ax_costFunction.set_zlabel('Cost, ' + r'$J(\theta)$', labelpad=5)
    ax_costFunction.view_init(elev=15., azim=235)

    ax_theta = fig_costFunction.add_subplot(122)
    ax_theta.contour(theta0_mesh, theta1_mesh, np.log10(j_vals), np.linspace(-2, 3, 20))
    ax_theta.scatter(theta[0], theta[1], c='r', marker='x', s=10)
    ax_theta.set_xlabel(r'$\theta_0$')
    ax_theta.set_ylabel(r'$\theta_1$')
    plt.show()
    fig_costFunction.savefig('fig_costFunction.png')

