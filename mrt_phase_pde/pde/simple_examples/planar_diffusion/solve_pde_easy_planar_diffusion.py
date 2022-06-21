import numpy as np
# import scipy.linalg
# from scipy.linalg import lu
import matplotlib.pyplot as plt
import pickle

from mrt_phase_numeric.src.config import equation_config

# from scipy.optimize import lsq_linear

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

D = 0.1

v_min = -1
v_max = 1
a_min = 0
a_max = 4

n_v = int(v_max - v_min) * 10
n_a = int(a_max - a_min) * 10

# n_v = 50
# n_a = 50

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

delta_v = np.diff(v)[0]
delta_a = np.diff(a)[0]

v_thr = 1.0

J = len(v) - 1
L = len(a) - 1


def flatten(j, l):
    """
    2d index to 1d array
    """
    return j * (L + 1) + l


def get_boundary_conditions():
    boundary_conditions = []
    b_boundary = []

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
            i = flatten(j, l)

            row = np.zeros(n_v * n_a)
            if j == 0:
                # row[flatten(j, l)] += 1
                row[flatten(j+1, l)] += 1
                row[flatten(j, l)] += 1

                boundary_conditions.append(row)
                b_boundary.append(0)

            row = np.zeros(n_v * n_a)
            if j == len(v) - 1:  # and _a < mu - _v:
                row[flatten(j, l)] += 1
                boundary_conditions.append(row)
                b_boundary.append(0)

            row = np.zeros(n_v * n_a)
            if l == 0:  # and _a < mu - _v:
                # row[flatten(j, l)] += 1
                row[flatten(j, l + 1)] += 1
                row[flatten(j, l)] += 1

                boundary_conditions.append(row)
                b_boundary.append(0)

            row = np.zeros(n_v * n_a)
            if l == len(a) - 1:  # and _a < mu - _v:
                # row[flatten(j, l)] += 1
                row[flatten(j, l)] += 1
                row[flatten(j, l - 1)] += 1

                boundary_conditions.append(row)
                b_boundary.append(0)

    boundary_conditions = np.array(boundary_conditions)

    return boundary_conditions, b_boundary


def construct_A_b():
    # A = np.zeros((n_v * n_a, n_v * n_a))
    # b = - np.ones(n_v * n_a)

    A = []
    b = []

    for j, _v in enumerate(v):
        for l, _a in enumerate(a):
            row = np.zeros(n_v * n_a)
            i = flatten(j, l)

            e = D / (delta_v ** 2)

            if j != 0 and j != len(v) - 1 and l != 0 and l != len(a) - 1:

                row[flatten(j + 1, l)] += 1 * e
                row[flatten(j, l)] -= 2 * e
                row[flatten(j - 1, l)] += 1 * e

                row[flatten(j, l + 1)] += 1 * e
                row[flatten(j, l)] -= 2 * e
                row[flatten(j, l - 1)] += 1 * e

                A.append(row)
                b.append(-1)

    boundary_conditions, b_boundary = get_boundary_conditions()

    A = np.append(A, boundary_conditions, axis=0)
    b = np.append(b, b_boundary, axis=0)

    return A, b


def solve_linear_system(A, b):
    T = None
    return T


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
    # b = - np.ones(n_v*n_a)

    # T = np.linalg.solve(A, b)

    T = np.linalg.lstsq(A, b)
    T = T[0]

    # for i in range(10):
    #     new_T = T - np.linalg.lstsq(A, (np.dot(A,T) - b))[0]
    #     T = new_T

    print(np.allclose(np.dot(A, T), b))
    T_matrix = T_to_T_matrix(T)


    plt.figure()
    plt.plot(np.dot(A, T))
    plt.plot(b)

    plt.legend(['b_result', 'b'])
    # plt.title()
    plt.ylim([-2, 1])



    v_, a_ = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(v_, a_, T_matrix.transpose(), levels=20)
    plt.xlabel('v')
    plt.ylabel('a')
    plt.title(f'solve_simple_pde_D_{D}_v_thr_{v_thr}')
    plt.colorbar()
