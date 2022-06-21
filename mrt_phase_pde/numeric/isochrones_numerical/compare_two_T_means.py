import numpy as np
import pickle
import matplotlib.pyplot as plt

filename_1 = 'T_mean\\T_D_0.25_v_thr_1.0_n_v_31_n_a_51_dt_0.01_n_trajectories_1000_n_thr_crossings_9.pkl'
filename_2 = 'T_mean\\T_D_0.25_v_thr_1.0_n_v_31_n_a_51_dt_0.01_n_trajectories_1000_n_thr_crossings_10.pkl'

with open(filename_1, 'rb') as f:
    T_1 = pickle.load(f)

with open(filename_2, 'rb') as f:
    T_2 = pickle.load(f)

v_min = -2
v_max = 1
a_min = 0
a_max = 5

n_v = int(v_max - v_min) * 10 + 1
n_a = int(a_max - a_min) * 10 + 1

# n_v = 20
# n_a = 40

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)


T_diff = T_2 - T_1

V, A = np.meshgrid(v, a)

plt.contourf(V, A, T_diff.transpose(), levels=30)
plt.xlabel('v')
plt.ylabel('a')
plt.colorbar()
