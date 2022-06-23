import matplotlib.pyplot as plt
import numpy as np

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N

D = 0.25

n_1 = 15
n_2 = 15

# T sim vs T pde 2 difference
T_1 = f'result\\T_N_{n_1}_D_0.0.pickle'
# T_1 = f'data\\T_0_D_{D}.pickle'
# T_2 = f'data\\T_0_D_{D}_sim_n_thr_10.pickle'
T_2 = f'result\\T_N_{n_2}_D_1.0.pickle'

t_1 = T_N.load(T_1)
t_2 = T_0.load(T_2)

max_n_a = 41

diff = t_2.T[:, :max_n_a] - t_1.T[:, :max_n_a]

__x, __y = np.meshgrid(t_2.v, t_2.a[:max_n_a])

plt.contourf(__x, __y, diff.transpose())
plt.colorbar()

# plt.savefig(f'img\\compare_T_{n_1}_with_T_{n_2}_D_{D}.png')
