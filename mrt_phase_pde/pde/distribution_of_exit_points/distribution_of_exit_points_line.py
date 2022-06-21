import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_pde.pde.distribution_of_exit_points.distribution_of_exit_points import solve_for_P
from mrt_phase_pde.pde.distribution_of_exit_points.settings import x_min, x_max, y_min, y_max, n_x, n_y, D, x, y
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution


def normalize(p):
    sum_p = np.sum(p) * np.diff(y)[0]
    if sum_p != 0.0:
        p = 1/sum_p * p

    return p


def get_p_for_point(i, j , P_matrix):
    p = np.zeros(len(y))
    for l, a in enumerate(y):
        if a > y[j]:
            break
        else:
            P = all_P[a]
            p[l] = P[i, j]

    return normalize(p)


def get_all_P(y=y):
    all_P = {}

    for a in y:
        x, y, P_matrix = solve_for_P(a)

        all_P[a] = P_matrix

    return all_P


if __name__ == '__main__':

    all_P = get_all_P()

    P = [[None for a in y] for v in x]

    for i, v in enumerate(x):
        for j, a in enumerate(y):
            P[i][j] = get_p_for_point(i, j, all_P)

    epd = ExitPointDistribution(x, y, P)
    epd.save(f'epd_D_{D}_pde.pickle')
