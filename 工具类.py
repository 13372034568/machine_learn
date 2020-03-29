from random import sample, randint

import numpy as np


def 创建多元多次函数(元数, 最大次数, 系数范围):
    # 创建系数
    元组 = []
    for _ in range(元数):
        元项 = []
        次数列 = list(range(0, 最大次数 + 1))
        采样数 = randint(1, 最大次数 + 1)
        次数采样 = sample(次数列, 采样数)
        次数采样.sort(reverse=True)
        for 次数 in 次数采样:
            系数 = randint(系数范围[0], 系数范围[1])
            元项.append((系数, 次数))
    元组.append(元项)
    # 实例化曲线
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
