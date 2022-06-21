import numpy as np
import matplotlib.pyplot as plt

import random

D = 0.25
dt = 0.01

x_min = -2
x_max = 1
y_min = 0
y_max = 5

n_x = int(x_max - x_min) * 10 + 1
n_y = int(y_max - y_min) * 10 + 1

# n_x = 20
# n_y = 40

x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)

x_thr = 1.0
n_trajectories = 1000

n_rotations = 9

alpha = 0.0125


def f(x, y):
    return np.cos(x) * np.sin(y) + alpha * np.sin(2*x)


def g(x, y):
    return -np.sin(x)*np.cos(y) + alpha*np.sin(2*y)


def _update(x, y):
    noise_1 = random.gauss(0, 1)
    noise_2 = random.gauss(0, 1)

    x_new = x + f(x, y) + np.sqrt(2 * D * dt) * noise_1
    y_new = y + g(x, y) + np.sqrt(2 * D * dt) * noise_2

    return x_new, y_new


def integrate_forwards(x, y):
    for i in range(n_rotations):
        while True:
            x, y = _update(x, y)
            phi = np.arctan2(y, x)
    

def update_until_line_cross(x, y):
    x_, y_ = integrate_forwards(x, y)

    n_timesteps = len(x_)
    return n_timesteps


def get_mean_rt(x, y):
    rt = []
    for i in range(n_trajectories):
        n_timesteps = update_until_line_cross(x, y)
        rt.append(n_timesteps * dt)

    return np.mean(rt)


def get_mean_T():
    T = np.zeros((len(x), len(y)))

    for i, _x in enumerate(x):
        print(i)
        for j, _y in enumerate(y):
            mean_rt = get_mean_rt(_x, _y)
            T[i, j] = mean_rt

    return T


if __name__=='__main__':
    pass

    T = get_mean_T()

    V, A = np.meshgrid(x, y)

    plt.contourf(V, A, T.trynspose(), lexels=30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
