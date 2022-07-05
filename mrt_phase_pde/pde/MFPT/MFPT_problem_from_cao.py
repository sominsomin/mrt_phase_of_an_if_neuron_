import numpy as np
import matplotlib.pyplot as plt

from config import equation_config

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = 0.0

x_min = -1
x_max = 1
y_min = 0
y_max = 10

n_x = int(x_max - x_min) * 20 + 1
n_y = int(y_max - y_min) * 20 + 1

x = np.linspace(x_min, x_max, n_x)
y = np.linspace(y_min, y_max, n_y)

h = np.diff(x)[0]

idx_mid = int((len(x) - 1) / 2)

x_all, y_all = np.meshgrid(x, y)

x_all = x_all.flatten()
y_all = y_all.flatten()

left = np.where((x_all == x_min))
bottom = np.where((y_all == y_min))
right = np.where((x_all == x_max))
top = np.where((y_all == y_max))

jump = np.where((x_all == x[idx_mid]))
right_jump = np.where((x_all == x[idx_mid - 1]))


def f(x, y):
    return mu - x - y


def g(x, y):
    return -y


def get_A_b():
    L_dagger = []
    b = - np.ones(len(x_all))

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

        # row[i] = -2 * D / h ** 2

        if i == 0:
            row[i] = 1

            b[i] = 0
        elif i in left[0]:
            idx_right = np.where((x_all == x[idx_x_up]) & (y_all == y_))
            # row[idx_right] = D / h ** 2

            row[idx_right] = 1
            row[i] = -1

            b[i] = 0

        elif i in right[0] and y_ <= mu - v_th:
            idx_left = np.where((x_all == x[idx_x_low]) & (y_all == y_))
            # row[idx_left] = D / h ** 2

            # row[idx_left] = -1
            row[i] = 1

            b[i] = 0

        elif i in right[0] and y_ > mu - v_th:
            idx_left = np.where((x_all == x[idx_x_low]) & (y_all == y_))

            row[i] = -1
            row[idx_left] = 1

            b[i] = 0

        elif i in bottom[0]:
            idx_up = np.where((x_all == x_) & (y_all == y[idx_y_up]))
            # row[idx_up] = D / h ** 2

            row[idx_up] = 1
            row[i] = -1

            b[i] = 0

        elif i in top[0]:
            idx_down = np.where((x_all == x_) & (y_all == y[idx_y_low]))
            # row[idx_down] = D / h ** 2

            row[idx_down] = -1
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


def T_to_T_matrix(T):
    T_matrix = np.zeros((len(x), len(y)))

    for i in range(len(x_all)):
        l = np.where(x_all[i] == x)
        m = np.where(y_all[i] == y)

        T_matrix[l, m] = T[i]

    return T_matrix


def get_T():
    L_dagger, b = get_A_b()

    T = np.linalg.lstsq(L_dagger, b)
    T = T[0]

    return T


def get_T_matrix():
    T = get_T()

    T_matrix = T_to_T_matrix(T)

    return x, y, T_matrix


if __name__=='__main__':
    L_dagger, b = get_A_b()

    T = get_T()

    # T = np.linalg.solve(L_dagger, b)

    print(np.allclose(np.dot(L_dagger, T), b))

    T_matrix = T_to_T_matrix(T)

    __x, __y = np.meshgrid(x, y)

    plt.figure()
    plt.plot(np.dot(L_dagger, T))
    plt.plot(b)
    plt.legend(['b_numeric', 'b_calc', 'b'])
    plt.ylim([-2, 1])

    plt.figure()
    plt.contourf(__x, __y, T_matrix.transpose(), levels=20) #, v_min=-30, v_max=30)
    plt.colorbar()
    for bound in [bottom, left, right, top, jump, right_jump]:
        plt.scatter(x_all[bound], y_all[bound])
    plt.xlabel('x')
    plt.ylabel('y')
