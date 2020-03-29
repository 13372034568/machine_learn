import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x[0] * x[0] + 50 * x[1] * x[1]


def g(x):
    return np.array([2 * x[0], 100 * x[1]])


def build_2D_func():
    xi = np.linspace(-200, 200, 1000)
    yi = np.linspace(-100, 100, 1000)
    X, Y = np.meshgrid(xi, yi)
    Z = X * X + 50 * Y * Y
    return X, Y, Z


# 绘制等高线图
def contour(X, Y, Z, arr=None):
    plt.figure(figsize=(12, 8))
    # plt.contourf(X, Y, Z, 100, cmap=plt.cm.hot)
    plt.contour(X, Y, Z, colors='black')
    plt.plot(0, 0, marker='*', markersize=12)
    if arr is not None:
        arr = np.array(arr)
        # plt.plot(arr[:, 0], arr[:, 1], marker="o", markersize=3)
        for i in range(len(arr) - 1):
            plt.plot(arr[i:i + 2, 0], arr[i:i + 2, 1], marker="o", markersize=3)

    plt.show()


def gd(x_start, step, g):  # gd代表了Gradient Descent
    x = np.array(x_start, dtype='float64')
    passing_dot = [x.copy()]
    for i in range(50):
        grad = g(x)
        x -= grad * step

        passing_dot.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break
    return x, passing_dot


if __name__ == '__main__':
    X, Y, Z = build_2D_func()
    contour(X, Y, Z)
    res, x_arr = gd([150, 75], 0.016, g)
    contour(X, Y, Z, x_arr)
