import numpy as np
import matplotlib.pyplot as plt

w = 1
gamma = 1
c = 0
D = 0.01

T_bar = 4.5

rho_min = 0.5
rho_max = 1.5

phi_min = 0
phi_max = 2 * np.pi

n_phi = 20
n_rho = 20

phi = np.linspace(phi_min, phi_max, n_phi)
rho = np.linspace(rho_min, rho_max, n_rho)

delta_rho = np.diff(rho)[0]
delta_phi = np.diff(phi)[0]

# rho, phi = np.meshgrid(rho, phi)
#
# rho = rho.flatten()
# phi = phi.flatten()

# for i, rho_ in enumerate(rho):
    # plt.scatter(rho[i], phi[i])


def Q(rho):
    return rho**2 - 1


def f(rho, phi):
    return w - gamma * c * Q(rho)


def g(rho, phi):
    return -gamma * rho * (rho**2 - 1)


J = len(rho) - 1
L = len(phi) - 1


def flatten(j, l):
    """
    2d index to 1d array index
    """
    return j * (L + 1) + l


L_dagger = []
b = []


rho_min_idx = []
rho_max_idx = []
phi_min_idx = []
phi_max_idx = []

for j, _rho in enumerate(rho):
    for l, _phi in enumerate(phi):
        row = np.zeros((len(rho) * len(phi)))

        if _phi == phi[-1]:
            row[flatten(j, l)] += 1
            # row[flatten(j - 1, l)] -= 1

            b.append(0)
        elif _rho == rho_min:
            row[flatten(j, l + 1)] += 1
            row[flatten(j, l)] -= 1

            b.append(0)
        elif _rho == rho_max:
            row[flatten(j, l)] += 1
            row[flatten(j, l - 1)] -= 1

            b.append(0)
        elif _phi == 0:
            row[flatten(j, l+1)] += 1
            row[flatten(j, l)] -= 1

            b.append(0)
        else:
            # a_1 = (g(_rho, _phi) + D/_rho) / (2 * delta_rho)
            a_1 = (D / _rho) / (2 * delta_rho)
            a_2 = (f(_rho, _phi)) / (2 * delta_phi)
            a_3 = D / (delta_rho**2)
            a_4 = D * 1/_rho**2 / (delta_phi**2)

            # e = w / (2 * delta_phi)

            # row[flatten(j + 1, l)] += a_1
            # row[flatten(j - 1, l)] -= a_1

            row[flatten(j, l + 1)] += a_2
            row[flatten(j, l - 1)] -= a_2

            # row[flatten(j + 1, l)] += 1 * a_3
            # row[flatten(j, l)] -= 2 * a_3
            # row[flatten(j - 1, l)] += 1 * a_3

            # row[flatten(j, l + 1)] += 1 * a_4
            # row[flatten(j, l)] -= 2 * a_4
            # row[flatten(j, l - 1)] += 1 * a_4

            b.append(-1)

        L_dagger.append(row)

        # row = np.zeros((len(rho) * len(phi)))
        # if _phi == 0:
        #     row[flatten(j, l)] += 1
        #     # row[flatten(j, l-1)] -= 1
        #
        #     L_dagger.append(row)
        #     b.append(0)


def T_to_T_matrix(T):
    T_matrix = np.zeros((n_rho, n_phi))

    for j, _v in enumerate(rho):
        for l, _a in enumerate(phi):
            i = flatten(j, l)
            T_matrix[j, l] = T[i]

    return T_matrix


T = np.linalg.lstsq(L_dagger, b)
T = T[0]

print(np.allclose(np.dot(L_dagger, T), b))
T_matrix = T_to_T_matrix(T)

rho_, phi_ = np.meshgrid(rho, phi)

plt.figure()
plt.plot(np.dot(L_dagger, T))
plt.plot(b)
plt.legend(['b_numeric', 'b_calc', 'b'])
plt.ylim([-2, 1])

plt.figure()
plt.contourf(rho_, phi_, T_matrix.transpose(), levels=20)
plt.xlabel('rho')
plt.ylabel('phi')
plt.colorbar()

# plt.figure()
# plt.scatter
