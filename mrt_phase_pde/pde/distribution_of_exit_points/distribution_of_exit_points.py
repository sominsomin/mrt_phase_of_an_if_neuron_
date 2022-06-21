import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.config import equation_config
from mrt_phase_pde.pde.distribution_of_exit_points.settings import x_min, x_max, y_min, y_max, n_x, n_y, D, x, y

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']


h = np.diff(x)[0]
epsilon = 2

idx_mid = int((len(x) - 1) / 2)

idx_epsilon_up = idx_mid + epsilon
idx_epsilon_low = idx_mid - epsilon

x_all, y_all = np.meshgrid(x, y)

x_all = x_all.flatten()
y_all = y_all.flatten()

left = np.where((x_all == x_min))
bottom = np.where((y_all == y_min))
right = np.where((x_all == x_max))
top = np.where((y_all == y_max))


def f(x, y):
    return mu - x - y


def g(x, y):
    return -y


def get_L_dagger_b(a):
    L_dagger = []
    b = np.zeros(len(x_all))
    # b = - np.ones(len(x_all))

    for i in range(0, len(x_all)):
        row = np.zeros(n_x * n_y)

        x_ = x_all[i]
        y_ = y_all[i]

        # idx_x_up is index x_ + h
        # idx_x_low is index x_ - h

        idx_x_up = (np.where(x == x_)[0][0] + 1)
        idx_x_low = (np.where(x == x_)[0][0] - 1)
        idx_y_up = (np.where(y == y_)[0][0] + 1)
        idx_y_low = (np.where(y == y_)[0][0] - 1)

        idx_y_a = np.where(y_all == a)

        # row[i] = -2 * D / h ** 2

        if i in right[0] and i in idx_y_a[0]:
            row[i] = 1

            b[i] = 1

        elif i in left[0]:
            row[i] = 1

            b[i] = 0

        elif i in right[0]:
            row[i] = 1

            b[i] = 0

        elif i in bottom[0]:
            row[i] = 1

            b[i] = 0

        elif i in top[0]:
            row[i] = 1

            b[i] = 0
        else:
            idx_up = np.where((x_all == x_) & (y_all == y[idx_y_up]))
            idx_down = np.where((x_all == x_) & (y_all == y[idx_y_low]))
            idx_left = np.where((x_all == x[idx_x_low]) & (y_all == y_))
            idx_right = np.where((x_all == x[idx_x_up]) & (y_all == y_))

            row[idx_right] += f(x_, y_) * 1 / (2 * h)
            row[idx_left] -= f(x_, y_) * 1 / (2 * h)

            row[idx_up] += g(x_, y_) * 1 / (2 * h)
            row[idx_down] -= g(x_, y_) * 1 / (2 * h)

            row[idx_right] += D / h ** 2
            row[i] -= 2 * D / h ** 2
            row[idx_left] += D / h ** 2

        L_dagger.append(row)

    L_dagger = np.array(L_dagger)

    return L_dagger, b


def P_to_P_matrix(P):
    P_matrix = np.zeros((len(x), len(y)))

    for i in range(len(x_all)):
        l = np.where(x_all[i] == x)
        m = np.where(y_all[i] == y)

        P_matrix[l, m] = P[i]

    return P_matrix


def solve_for_P(a):
    L_dagger, b = get_L_dagger_b(a)

    P = np.linalg.lstsq(L_dagger, b)
    P = P[0]

    # P = np.linalg.solve(L_dagger, b)

    print(np.allclose(np.dot(L_dagger, P), b))

    P_matrix = P_to_P_matrix(P)

    return x, y, P_matrix


if __name__=='__main__':
    a = 1.5

    L_dagger, b = get_L_dagger_b(a)

    P = np.linalg.lstsq(L_dagger, b)
    P = P[0]

    # P = np.linalg.solve(L_dagger, b)

    print(np.allclose(np.dot(L_dagger, P), b))

    P_matrix = P_to_P_matrix(P)

    __x, __y = np.meshgrid(x, y)

    plt.figure()
    plt.plot(np.dot(L_dagger, P))
    plt.plot(b)
    plt.legend(['b_numeric', 'b_calc', 'b'])
    plt.ylim([-2, 1])

    plt.figure()
    plt.contourf(__x, __y, P_matrix.transpose(), levels=20) #, v_min=-30, v_max=30)
    plt.colorbar()
    for bound in [bottom, left, right, top]:
        plt.scatter(x_all[bound], y_all[bound])
    plt.xlabel('x')
    plt.ylabel('y')
