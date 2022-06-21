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

m = 101

n_x = m
n_y = m

x = np.linspace(-np.pi/2, np.pi/2, n_x)
y = np.linspace(-np.pi/2, np.pi/2, n_y)

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

r = np.sqrt(len(remove))

x_all = x_all[mask]
y_all = y_all[mask]

outer_left = np.where((x_all == x_min))
outer_bottom = np.where((y_all == y_min))
outer_right = np.where((x_all == x_max))
outer_top = np.where((y_all == y_max))

inner_left = np.where((x_all == x[idx_epsilon_low]) & (y_all <= y[idx_epsilon_up]) & (y_all >= y[idx_epsilon_low]))
inner_bottom = np.where((y_all == x[idx_epsilon_low]) & (x_all <= x[idx_epsilon_up]) & (x_all >= x[idx_epsilon_low]))
inner_right =  np.where((x_all == x[idx_epsilon_up]) & (y_all <= y[idx_epsilon_up]) & (y_all >= y[idx_epsilon_low]))
inner_top = np.where((y_all == y[idx_epsilon_up]) & (x_all <= x[idx_epsilon_up]) & (x_all >= x[idx_epsilon_low]))

jump = np.where((y_all>=y[idx_epsilon_up]) & (y_all<=y_max) & (x_all == x[idx_mid]))
right_jump = np.where((y_all>=y[idx_epsilon_up]) & (y_all<=y_max) & (x_all == x[idx_mid-1]))


def f(x, y):
    return np.cos(x) * np.sin(y) + alpha*np.sin(2 * x)


def g(x,y):
    return -np.sin(x) * np.cos(y) + alpha * np.sin(2 * y)


right_norm = []
for i in range(1,len(y_all)-m):
    if y_all[i] == y_all[i+m]:
        right_norm.append(i)

left_norm = []
for i in range(len(y_all)-1, m, -1):
    if y_all[i] == y_all[i-m]:
        left_norm.append(i)


L_dagger = []
b = []

for i in range(len(x_all)):
    row = np.zeros(n_x * n_y)

    x_ = x_all[i]
    y_ = y_all[i]

    up = i - 1
    down = i + 1

    if i in outer_top[0]:
        pass
    elif i in inner_bottom[0]:
        pass
    elif i in outer_bottom[0]:
        row[up] = 2*D / h**2
    elif i in inner_top[0] :
        row[up] = 2 * D / h ** 2
    elif i>=2:
        row[i] = g(x_all[i], y_all[i]) / (2*h) + D/h**2

    if i in outer_bottom[0]:
        pass
    elif i in inner_top[0]:
        pass
    elif i in outer_top[0]:
        row[down] = 2 * D / h**2
    elif i in inner_bottom[0]:
        row[down] = 2 * D / h**2
    else:
        row[down] = - g(x_all[i], y_all[i]) / (2*h) + D/h**2

    if i in right_norm:
        right = i + m
    else:
        right = i + (m-r)

    if i in left_norm:
        left = i - m
    else:
        left = i - (m-r)

    if i not in outer_left[0] and i not in inner_right[0] and i >= m+1:
        row[left] = -f(x_all[i], y_all[i]) / (2*h) + D/h**2

    if i not in outer_right[0] and i not in inner_left[0]:
        row[left] = f(x_all[i], y_all[i]) / (2*h) + D/h**2


    L_dagger.append(row)


def T_to_T_matrix(T):
    T_matrix = np.zeros((len(x), len(y)))

    for i in range(len(x_all)):
        l = np.where(x_all[i] == x)
        m = np.where(y_all[i] == y)

        T_matrix[l, m] = T[i]

    return T_matrix


T = np.linalg.lstsq(L_dagger, b)
T = T[0]

print(np.allclose(np.dot(L_dagger, T), b))

T_matrix = T_to_T_matrix(T)


__x, __y = np.meshgrid(x, y)


plt.figure()
plt.plot(np.dot(L_dagger, T))
plt.plot(b)
plt.legend(['b_numeric', 'b_calc', 'b'])
plt.ylim([-2, 1])

plt.figure()
plt.contourf(__x, __y, T_matrix.transpose(), levels=20)
plt.colorbar()
for bound in [lower_inner_bound_x, lower_inner_bound_y, lower_bound_y, lower_bound_x, upper_inner_bound_y, upper_inner_bound_x, upper_bound_x, upper_bound_y, jump, right_jump]:
    plt.scatter(x_all[bound], y_all[bound])
plt.xlabel('x')
plt.ylabel('y')
