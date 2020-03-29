from random import randint

import numpy as np
import matplotlib.pyplot as plt


def 多项式函数():
    x = np.arange(-10, 10)
    pow_args = 2
    theta_area = [-10, 10]
    thetas = []
    y = 0
    for pow_arg in range(0, pow_args + 1):
        theta = randint(theta_area[0], theta_area[1])
        y += theta * np.power(x, pow_arg)
        thetas.append(theta)
    plt.grid(True)
    plt.plot(x, y, linewidth=1)
    plt.show()
    plt.close()
    return thetas


if __name__ == '__main__':
    thetas = 多项式函数()
    print(thetas)
