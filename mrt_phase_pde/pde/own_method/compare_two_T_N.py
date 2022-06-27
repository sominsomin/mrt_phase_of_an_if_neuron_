import matplotlib.pyplot as plt
import numpy as np
import math

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N


def find_nearest(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


D = 0.25

n_1 = 6
n_2 = 6

# T sim vs T pde 2 difference
T_1 = f'result\\T_N_{n_1}_D_{0.0}.pickle'
# T_1 = f'data\\T_0_D_{D}.pickle'
# T_2 = f'data\\T_0_D_{D}_sim_n_thr_10.pickle'
T_2 = f'result\\T_N_{n_2}_D_{0.25}.pickle'

t_1 = T_N.load(T_1)
t_2 = T_0.load(T_2)

# max_n_a = 31
a = np.linspace(0, 3, 31)
v = np.linspace(-1, 1, 21)

idx_a_t_1 = [find_nearest(t_1.a, _a) for _a in a]
idx_a_t_2 = [find_nearest(t_2.a, _a) for _a in a]
idx_v_t_1 = [find_nearest(t_1.v, _v) for _v in v]
idx_v_t_2 = [find_nearest(t_2.v, _v) for _v in v]

diff = t_2.T[np.ix_(idx_v_t_2, idx_a_t_2)] - t_1.T[np.ix_(idx_v_t_1, idx_a_t_1)]

__x, __y = np.meshgrid(t_2.v[idx_v_t_2], t_2.a[idx_a_t_2])

plt.contourf(__x, __y, diff.transpose())
plt.colorbar()

# plt.savefig(f'img\\compare_T_{n_1}_with_T_{n_2}_D_{D}.png')
