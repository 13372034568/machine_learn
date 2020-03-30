import os
import random
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


def set_x_y():
    # Size of the points dataset.
    m = 100

    # Points x-coordinate and dummy value (x0, x1).
    x0 = np.ones((m, 1))
    x1 = np.arange(1, m + 1).reshape(m, 1)
    X = np.hstack((x0, x1))

    # Points y-coordinate
    # y = np.array([
    #     3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12,
    #     11, 13, 13, 16, 17, 18, 17, 19, 21
    # ]).reshape(m, 1)

    x = np.arange(1, m + 1)
    y = []
    for xi in x:
        yi = 0.7 * xi + 2.4
        yi += (random.randint(1, 100) - 50) / 10
        y.append(yi)
    y = np.array(y).reshape(m, 1)

    return X, y


def loss_function(theta, X, y):
    m = np.shape(X)[0]
    diff = np.dot(X, theta) - y
    return (1 / (2 * m)) * np.dot(np.transpose(diff), diff)


def gradient_function(theta, X, y):
    m = np.shape(X)[0]
    diff = np.dot(X, theta) - y
    return (1 / m) * np.dot(np.transpose(X), diff)


def gradient_descent(X, y, alpha):
    theta = np.array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, y)
    try_index = 1
    while not np.all(np.absolute(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        if try_index == 1 or try_index % 20 == 0:
            draw_pic(X, theta, y, try_index)
        try_index += 1
    draw_pic(X, theta, y, try_index)
    return theta, try_index


def draw_pic(X, theta, y_origin, try_index):
    t_y = np.dot(X, theta)
    t_X = [int(i[1]) for i in X]
    t_y = np.transpose(t_y)[0]
    t_y_origin = np.transpose(y_origin)[0]
    plt.grid(True)
    plt.plot(t_X, t_y, alpha=min(try_index / 4000, 1), linewidth=0.5)
    if try_index == 1:
        plt.scatter(t_X, t_y_origin)


if __name__ == '__main__':
    X, y = set_x_y()
    m = np.shape(X)[0]

    # The Learning Rate alpha.
    alpha = 2e-4

    optimal, try_index = gradient_descent(X, y, alpha)
    print(try_index)
    print("optimal:", optimal)
    print("loss:", loss_function(optimal, X, y))

    plt.savefig("gradient_descent.png")
    plt.close()
