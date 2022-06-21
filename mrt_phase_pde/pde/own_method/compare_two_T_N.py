import matplotlib.pyplot as plt
import numpy as np

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N

# T sim vs T pde 2 difference
T_1 = 'result\\T_N_8_D_0.25.pickle'
# T_1 = 'data\\T_0_D_0.25.pickle'
T_2 = 'data\\T_0_D_0.25_sim_n_thr_10.pickle'

t_1 = T_N.load(T_1)
t_2 = T_0.load(T_2)

max_n_a = 41

diff = t_2.T[:, :max_n_a] - t_1.T[:, :max_n_a]

__x, __y = np.meshgrid(t_2.v, t_2.a[:max_n_a])

plt.contourf(__x, __y, diff.transpose())
plt.colorbar()