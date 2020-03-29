from random import randint
import numpy as np
import matplotlib.pyplot as plt


def f(x, thetas):
    y = 0
    for pow_time, theta in enumerate(thetas):
        y += theta * pow(x, pow_time)
    return y


def gradient(x, thetas):
    y = 0
    for pow_time, theta in enumerate(thetas):
        if pow_time == 0:
            y += 0
        else:
            y += pow_time * theta * pow(x, pow_time - 1)
    return y


def gradient_descent(x_start, step, g, thetas,
                     超参数_最大迭代数=1000,
                     超参数_记录间隔=500,
                     超参数_梯度停止阈值=1e-6):  # gd代表了Gradient Descent
    x_records = []
    x = x_start
    for i in range(超参数_最大迭代数):
        grad = g(x, thetas)
        if i == 0 or i % 超参数_记录间隔 == 0:
            x_records.append(x)
        x -= grad * step
        if i == 0 or i % 超参数_记录间隔 == 0:
            print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(grad) < 超参数_梯度停止阈值:
            break
    return x, x_records


def build_多项式函数():
    x_area = [-50, 50]
    theta_area = [-10, 10]
    pow_args = 2

    x = np.arange(x_area[0], x_area[1])
    thetas = []
    y = 0
    for pow_arg in range(0, pow_args + 1):
        if pow_arg == pow_args:
            theta = randint(1, theta_area[1])
        else:
            theta = randint(theta_area[0], theta_area[1])
        y += theta * np.power(x, pow_arg)
        thetas.append(theta)

    return thetas, x, y


def draw_多项式函数(line_x, line_y,
               predict_x, predict_y,
               progress_x, progress_y,
               out_png_fp):
    plt.grid(True)
    plt.plot(line_x, line_y, linewidth=1)
    plt.axvline(x=predict_x, ls="--", c="green")  # 添加垂直直线
    plt.axhline(y=predict_y, ls="--", c="red")  # 添加水平直线
    plt.plot(progress_x, progress_y, marker="o", linewidth=1)
    plt.savefig(out_png_fp)
    plt.show()
    plt.close()


if __name__ == '__main__':
    out_png_fp = "梯度下降_函数实验1.png"
    thetas, line_x, line_y = build_多项式函数()
    超参数_学习率 = 4e-5
    超参数_起始点 = 45
    超参数_最大迭代数 = int(1e6)
    超参数_记录间隔 = 500
    超参数_梯度停止阈值 = 1e-6
    predict_x, progress_x = gradient_descent(超参数_起始点, 超参数_学习率,
                                             gradient, thetas,
                                             超参数_最大迭代数, 超参数_记录间隔,
                                             超参数_梯度停止阈值)
    predict_y = f(predict_x, thetas)
    progress_y = [f(t_x, thetas) for t_x in progress_x]
    print("thetas=", thetas)
    print("predict_x=", predict_x)
    print("predict_y=", predict_y)
    print("len(progress_x)=", len(progress_x))
    print("len(progress_y)=", len(progress_y))
    draw_多项式函数(line_x, line_y,
               predict_x, predict_y,
               progress_x, progress_y,
               out_png_fp)
