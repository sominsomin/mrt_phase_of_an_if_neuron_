import numpy as np
import matplotlib.pyplot as plt

# x is low
# y is up

T_bar = 16.225796508539315
alpha = 0.1
D = 0.01125

x_min = -np.pi / 2
x_max = np.pi / 2
y_min = -np.pi/2
y_max = np.pi/2

n_x = int(x_max - x_min) * 10 + 1
n_y = int(y_max - y_min) * 10 + 1

m = 51

n_x = m
n_y = m

x = np.linspace(-np.pi/2, np.pi/2, n_x)
y = np.linspace(np.pi/2, -np.pi/2, n_y)

h = np.diff(x)[0]
epsilon = 2

idx_mid = int((len(x) - 1)/2)

idx_epsilon_up = idx_mid + epsilon
idx_epsilon_low = idx_mid - epsilon

x_all, y_all = np.meshgrid(x, y)

x_all = x_all.flatten()
y_all = y_all.flatten()

mask = np.ones(len(x_all), dtype=bool)
remove = np.where((x_all < x[idx_epsilon_up]) & (x_all > x[idx_epsilon_low]) & (y_all < y[idx_epsilon_up]) & (y_all > y[idx_epsilon_low]))
mask[remove] = False

x_all = x_all[mask]
y_all = y_all[mask]

outer_left = np.where((x_all == x_min))
outer_bottom = np.where((y_all == y_min))
outer_right = np.where((x_all == x_max))
outer_top = np.where((y_all == y_max))

# fuer y ist es anders rum da y vektor von pi/2 bis -pi/2
inner_left = np.where((x_all == x[idx_epsilon_low]) & (y_all <= y[idx_epsilon_low]) & (y_all >= y[idx_epsilon_up]))
inner_bottom = np.where((y_all == y[idx_epsilon_low]) & (x_all <= x[idx_epsilon_up]) & (x_all >= x[idx_epsilon_low]))
inner_right =  np.where((x_all == x[idx_epsilon_up]) & (y_all <= y[idx_epsilon_low]) & (y_all >= y[idx_epsilon_up]))
inner_top = np.where((y_all == y[idx_epsilon_up]) & (x_all <= x[idx_epsilon_up]) & (x_all >= x[idx_epsilon_low]))

jump = np.where((y_all>=y[idx_epsilon_up]) & (y_all<=y_max) & (x_all == x[idx_mid]))
right_jump = np.where((y_all>=y[idx_epsilon_up]) & (y_all<=y_max) & (x_all == x[idx_mid-1]))


def f(x, y):
    return np.cos(x) * np.sin(y) + alpha*np.sin(2 * x)


def g(x,y):
    return -np.sin(x) * np.cos(y) + alpha * np.sin(2 * y)


L_dagger = []
b = []

for i in range(0, len(x_all)):
    row = np.zeros(n_x * n_y)

    x_ = x_all[i]
    y_ = y_all[i]

    # idx_x_up is index x_ + h
    # idx_x_low is index x_ - h

    idx_x_up = (np.where(x == x_)[0][0] + 1)
    idx_x_low = (np.where(x == x_)[0][0] - 1)
    idx_y_up = (np.where(y == y_)[0][0] - 1)
    idx_y_low = (np.where(y == y_)[0][0] + 1)

    row[i] = -4 * D / h**2

    if i == 0:
        pass
        # row[i] = 1
        # b.append(100)
    elif i in outer_left[0] or i in inner_right[0]:
        idx_right = np.where((x_all == x[idx_x_up]) & (y_all == y_))
        row[idx_right] = 2 * D / h**2

    elif i in outer_right[0] or i in inner_left[0]:
        idx_left = np.where((x_all == x[idx_x_low]) & (y_all == y_))
        row[idx_left] = 2 * D / h**2

    elif i in outer_bottom[0] or i in inner_top[0]:
        idx_up = np.where((x_all == x_) & (y_all == y[idx_y_up]))
        row[idx_up] = 2 * D / h**2

    elif i in outer_top[0] or i in inner_bottom[0]:
        idx_down = np.where((x_all == x_) & (y_all == y[idx_y_low]))
        row[idx_down] = 2 * D / h**2

    else:
        idx_up = np.where((x_all == x_) & (y_all == y[idx_y_up]))
        idx_down = np.where((x_all == x_) & (y_all == y[idx_y_low]))
        idx_left = np.where((x_all == x[idx_x_low]) & (y_all == y_))
        idx_right = np.where((x_all == x[idx_x_up]) & (y_all == y_))

        row[idx_right] += f(x_,y_) * 1/(2 * h)
        row[idx_left] -= f(x_, y_) * 1/(2 * h)

        row[idx_up] += g(x_,y_) * 1/(2 * h)
        row[idx_down] -= g(x_, y_) * 1/(2 * h)

        row[idx_up] += D / h**2
        # row[i] -= 2 * D / h**2
        row[idx_down] += D / h ** 2

        row[idx_right] += D / h ** 2
        # row[i] -= 2 * D / h ** 2
        row[idx_left] += D / h ** 2

    L_dagger.append(row)

b = - np.ones(len(x_all))

T_0 = 0 # specify northwest corner
# b[0] = T_0
b[1] = -1 - T_0 * (g(x_all[1], y_all[1]) / (h) + D/h**2)
b[m] = -1 + T_0 * (f(x_all[m], y_all[m]) / (h) - D/h**2)
T_bar = 16.225796508539315 #; % from simulations
b[jump] = -1 + T_bar * (f(x_all[jump], y_all[jump])/(2*h) + D/h**2)
b[right_jump] = -1 + T_bar * (f(x_all[right_jump],y_all[right_jump])/(2*h) - D/h**2)


def T_to_T_matrix(T):
    T_matrix = np.zeros((len(x), len(y)))

    for i in range(len(x_all)):
        l = np.where(x_all[i] == x)
        m = np.where(y_all[i] == y)

        T_matrix[l, m] = T[i]

    return T_matrix

#
# T = np.linalg.lstsq(L_dagger, b)
# T = T[0]

L_dagger = np.array(L_dagger)
T = np.linalg.solve(L_dagger, b)

print(np.allclose(np.dot(L_dagger, T), b))

T_matrix = T_to_T_matrix(T)


__x, __y = np.meshgrid(x, y)


plt.figure()
plt.plot(np.dot(L_dagger, T))
plt.plot(b)
plt.legend(['b_numeric', 'b_calc', 'b'])
plt.ylim([-2, 1])

plt.figure()
plt.contourf(__x, __y, T_matrix.transpose(), vmin=-20, v_max=20, levels=30)
plt.colorbar()
for bound in [inner_left, inner_bottom, outer_bottom, outer_left, inner_top, inner_right, outer_right, outer_top, jump, right_jump]:
    plt.scatter(x_all[bound], y_all[bound])
plt.xlabel('x')
plt.ylabel('y')
