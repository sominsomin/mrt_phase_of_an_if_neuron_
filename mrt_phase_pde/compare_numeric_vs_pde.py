import numpy as np
# import scipy.linalg
# from scipy.linalg import lu
import matplotlib.pyplot as plt
import pickle

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

v_thr = 1


if __name__ == '__main__':

    with open('pde\\T_pde_D_0.1_v_thr_1.0.pkl', 'rb') as f:
        T_pde = pickle.load(f)

    with open('numeric\\T_D_0.1_v_thr_1.0_n_v_20_n_a_40.pkl', 'rb') as f:
        T_numeric = pickle.load(f)

    v_, a_ = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(v_, a_, T_pde.transpose() - T_numeric.transpose(), levels=20)
    plt.xlabel('v')
    plt.ylabel('a')
    plt.title(f'solve_pde_D_{D}_v_thr_{v_thr}')
    plt.colorbar()
