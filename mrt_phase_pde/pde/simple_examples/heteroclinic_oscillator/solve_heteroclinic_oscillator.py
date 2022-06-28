import numpy as np
# import scipy.linalg
# from scipy.linalg import lu
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D = 0.01125
alpha = 0.1
T_mean = 16.225796508539315

x_min = -np.pi / 2
x_max = np.pi / 2
y_min = -np.pi/2
y_max = np.pi/2

x_cut_out = 0.10471975511965992
y_cut_out = 0.10471975511965992

n_x = int(x_max - x_min) * 20 + 1
n_y = int(y_max - y_min) * 20 + 1

x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)

delta_x = np.diff(x)[0]
delta_y = np.diff(y)[0]

v_thr = x_max

J = len(x) - 1
L = len(y) - 1


def flatten(j, l):
    """
    2d index to 1d array index
    """
    return j * (L + 1) + l


j_mid = (len(x) - 1) / 2
l_mid = (len(y) - 1) / 2


def is_in_cut_out(x, y):
    if x < x_cut_out and x > -x_cut_out and y < y_cut_out and y > -y_cut_out:
        return True
    else:
        return False


def is_close_to_cut_out(x, y):
    if np.isclose(x, x_cut_out, atol=0.1) and y_cut_out > y > - y_cut_out:
        return True
    elif np.isclose(x, -x_cut_out, atol=0.1) and y_cut_out > y > - y_cut_out:
        return True
    elif np.isclose(y, y_cut_out, atol=0.1) and x_cut_out > x > - x_cut_out:
        return True
    elif np.isclose(y, -y_cut_out, atol=0.1) and x_cut_out > x > - x_cut_out:
        return True
    else:
        return False


def construct_A_b():
    A = []
    b = []

    for j, _x in enumerate(x):
        for l, _y in enumerate(y):
            row = np.zeros(n_x * n_y)

            if j == 0 and l == 0:
                row[flatten(j, l)] += 1

                b.append(0)
            elif is_in_cut_out(_x, _y):
                continue
            elif _x == 0 and _y <= 0:
                row[flatten(j, l)] += 1
                row[flatten(j + 1, l)] -= 1

                b.append(0)
            elif _x == x_min:
                row[flatten(j, l + 1)] += 1
                row[flatten(j, l)] -= 1

                b.append(0)
            elif _x == x_max:
                row[flatten(j, l)] += 1
                row[flatten(j, l - 1)] -= 1

                b.append(0)
            elif _y == y_min:
                row[flatten(j + 1, l)] += 1
                row[flatten(j, l)] -= 1

                b.append(0)
            elif _y == y_max:
                row[flatten(j, l)] += 1
                row[flatten(j - 1, l)] -= 1

                b.append(0)
            else:
                f_x = np.cos(_x) * np.sin(_y) + alpha * np.sin(2 * _x) * 1/delta_x
                g_y = -np.sin(_x) * np.cos(_y) + alpha * np.sin(2 * _y) * 1/delta_y

                B_xx = D / (delta_x ** 2)
                B_yy = D / (delta_y ** 2)

                row[flatten(j + 1, l)] += 1 / 2 * f_x
                row[flatten(j - 1, l)] -= 1 / 2 * f_x

                row[flatten(j, l + 1)] += 1 / 2 * g_y
                row[flatten(j, l - 1)] -= 1 / 2 * g_y

                row[flatten(j + 1, l)] += 1 * B_xx
                row[flatten(j, l)] -= 2 * B_xx
                row[flatten(j - 1, l)] += 1 * B_xx

                row[flatten(j, l + 1)] += 1 * B_yy
                row[flatten(j, l)] -= 2 * B_yy
                row[flatten(j, l - 1)] += 1 * B_yy

                b.append(-1)

            A.append(row)

    return A, b


def T_to_T_matrix(T):
    T_matrix = np.zeros((n_x, n_y))

    for j, _v in enumerate(x):
        for l, _a in enumerate(y):
            i = flatten(j, l)
            T_matrix[j, l] = T[i]

    return T_matrix


def T_matrix_to_T(T_matrix):
    T = np.zeros(n_x * n_y)

    for j, _v in enumerate(x):
        for l, _a in enumerate(y):
            i = flatten(j, l)
            T[i] = T_matrix[j, l]

    return T


if __name__ == '__main__':
    A, b = construct_A_b()

    T = np.linalg.lstsq(A, b)
    T = T[0]

    print(np.allclose(np.dot(A, T), b))
    T_matrix = T_to_T_matrix(T)

    # with open(f'T_pde_D_{D}_v_thr_{v_thr}.pkl', 'wb') as f:
    #     pickle.dump(T_matrix, f)

    x_, y_ = np.meshgrid(x, y)

    plt.figure()
    # plt.plot(np.dot(A, T_numeric))
    plt.plot(np.dot(A, T))
    plt.plot(b)
    plt.legend(['b_numeric', 'b_calc', 'b'])
    plt.ylim([-2, 1])

    plt.figure()
    plt.contourf(x_, y_, T_matrix.transpose(), levels=20)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
