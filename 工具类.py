from random import sample, randint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def 创建多元多次函数(元数, 最大次数, 系数范围, 自变量取值范围):
    # 创建系数
    元组 = []
    for _ in range(元数):
        元项 = []
        次数列 = list(range(0, 最大次数 + 1))
        采样数 = randint(1, 最大次数 + 1)
        次数采样 = sample(次数列, 采样数)
        次数采样.sort(reverse=True)
        for e, 次数 in enumerate(次数采样):
            if e == 0:
                系数 = randint(1, 系数范围[1])
            else:
                系数 = randint(系数范围[0], 系数范围[1])
            元项.append((系数, 次数))
        元组.append(元项)
    # 实例化曲线
    f_应变量 = 0
    自变量列表 = []
    for 元项序号, 元项 in enumerate(元组):
        f_元项 = 0
        x_area = 自变量取值范围[元项序号]
        x = np.arange(x_area[0], x_area[1])
        for 系数, 次数 in 元项:
            f_元项 += 系数 * np.power(x, 次数)
        f_应变量 += f_元项
        自变量列表.append(x)

    return 元组, 自变量列表, f_应变量


def 绘制多元多次函数曲线(自变量列表, f_应变量, out_png_fp=None):
    plt.grid(True)
    fig = plt.figure()

    if len(自变量列表) == 2:
        x = 自变量列表[0]
        y = 自变量列表[1]
        ax = fig.gca(projection='3d')
        ax.plot(x, y, f_应变量, color='r')
        ax.legend()
    elif len(自变量列表) == 1:
        x = 自变量列表[0]
        plt.plot(x, f_应变量, linewidth=1)
    else:
        plt.close()
        return

    plt.show()
    plt.close()


if __name__ == '__main__':
    元数 = 2
    最大次数 = 2
    系数范围 = [-10, 10]
    自变量取值范围 = [
        [-200, 200],
        [-200, 200]
    ]
    元组, 自变量列表, f_应变量 = 创建多元多次函数(元数, 最大次数, 系数范围, 自变量取值范围)
    print(np.shape(自变量列表))
    绘制多元多次函数曲线(自变量列表, f_应变量)
