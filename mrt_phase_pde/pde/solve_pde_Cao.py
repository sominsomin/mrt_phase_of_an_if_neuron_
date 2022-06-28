import numpy as np
# import scipy.linalg
# from scipy.linalg import lu
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from config import equation_config

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

# from scipy.optimize import lsq_linear

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = 0.5

T_mean = equation_config['mean_T_dict'][D]

v_min = -1.0
v_max = 1.0
a_min = 0
a_max = 4

n_v = int(v_max - v_min) * 10 + 1
n_a = int(a_max - a_min) * 10 + 1

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

delta_v = np.diff(v)[0]
delta_a = np.diff(a)[0]

v_thr = v_max

J = len(v) - 1
L = len(a) - 1


def flatten(j, l):
    """
    2d index to 1d array index
    """
    return j * (L + 1) + l


def get_boundary_conditions():
    boundary_conditions = []
    b_boundary = []

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
            i = flatten(j, l)

            row = np.zeros(n_v * n_a)
            if _v == v_min:
                row[flatten(j + 1, l)] += 1
                row[flatten(j, l)] -= 1

                boundary_conditions.append(row)
                b_boundary.append(0)

            # row = np.zeros(n_v * n_a)
            # if _v == 0:
            #     l_delta_a = np.where(a == _a - Delta_a)
            #     if any(l_delta_a):
            #         l_delta_a = l_delta_a[0][0]
            #         j_max = len(v) - 1
            #
            #         row[flatten(j, l)] += 1
            #         row[flatten(j_max, l_delta_a)] -= 1
            #         b_boundary.append(T_mean)
            #     else:
            #         row[flatten(j + 1, l)] += 1
            #         row[flatten(j, l)] -= 1
            #         b_boundary.append(0)
            #     boundary_conditions.append(row)

            row = np.zeros(n_v * n_a)
            if _v == v_thr:
                # l_delta_a = np.where(a == _a + Delta_a)
                # if any(l_delta_a):
                #     # l_delta_a = l_delta_a[0][0]
                #     # j_0 = np.where(v == 0)[0][0]
                #     #
                #     # row[flatten(j, l)] += 1
                #     # row[flatten(j_0, l_delta_a)] -= 1
                #     # b_boundary.append(T_mean)
                #     pass
                # else:
                # row[flatten(j - 1, l)] -= 1
                row[flatten(j, l)] += 1
                b_boundary.append(0)
                boundary_conditions.append(row)


            # if j == len(v) - 1:  # and _a < mu - _v:
            #     row[i] += 1
            #
            #     boundary_conditions.append(row)
            #     b_boundary.append(0)

            # row = np.zeros(n_v * n_a)
            # if j == len(v) - 1:  # and _a < mu - _v:
            #     # row[i] += 1
            #     boundary_conditions.append(row)
            #     b_boundary.append(0)

            row = np.zeros(n_v * n_a)
            if l == 0:
                row[flatten(j, l + 1)] += 1
                row[flatten(j, l)] -= 1
                boundary_conditions.append(row)
                b_boundary.append(0)

            row = np.zeros(n_v * n_a)
            if l == len(a) - 1:
                row[flatten(j, l)] += 1
                row[flatten(j, l - 1)] -= 1
                boundary_conditions.append(row)
                b_boundary.append(0)

    boundary_conditions = np.array(boundary_conditions)

    return boundary_conditions, b_boundary


def construct_A_b():
    A = []
    b = []

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
            i = flatten(j, l)

            d = (mu - _v - _a) / delta_v
            f = - _a / delta_a
            e = D / (delta_v ** 2)

            row = np.zeros(n_v * n_a)
            if j != 0 and j != len(v) - 1 and l != 0 and l != len(a) - 1:
                row[flatten(j + 1, l)] += 1 / 2 * d
                row[flatten(j - 1, l)] -= 1 / 2 * d

                row[flatten(j + 1, l)] += 1 * e
                row[flatten(j, l)] -= 2 * e
                row[flatten(j - 1, l)] += 1 * e

                row[flatten(j, l + 1)] += 1 / 2 * f
                row[flatten(j, l - 1)] -= 1 / 2 * f

                A.append(row)
                b.append(-1)

    boundary_conditions, b_boundary = get_boundary_conditions()

    A = np.append(A, boundary_conditions, axis=0)
    b = np.append(b, b_boundary, axis=0)

    return A, b


def T_to_T_matrix(T):
    T_matrix = np.zeros((n_v, n_a))

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
            i = flatten(j, l)
            T_matrix[j, l] = T[i]

    return T_matrix


def T_matrix_to_T(T_matrix):
    T = np.zeros(n_v * n_a)

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
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

    v_, a_ = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(v_, a_, T_matrix.transpose(), levels=20)
    plt.xlabel('v')
    plt.ylabel('a')
    plt.title(f'D = {D}, v_thr = {v_thr}, delta_a = {delta_a}, mu = {mu}')
    plt.colorbar()

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1])

    # with open('..\\numeric\\T_D_0.1_v_thr_1.0_n_v_20_n_a_40.pkl', 'rb') as f:
    #     T_numeric = pickle.load(f)
    #
    # T_numeric = T_matrix_to_T(T_numeric)
    #
    # print(np.allclose(np.dot(A, T_numeric), b))
    #

    plt.figure()
    # plt.plot(np.dot(A, T_numeric))
    plt.plot(np.dot(A, T))
    plt.plot(b)
    plt.legend(['b_numeric', 'b_calc', 'b'])
    plt.ylim([-2, 1])

