import matplotlib.pyplot as plt
import numpy as np

from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N

D = 0.0

n = list(range(1, 10))
diff_mean = []
diff_max = []
diff_min = []

for i in n:
    n_1 = i
    n_2 = i-1

    T_1 = f'result\\T_N_{n_1}_D_{D}.pickle'
    T_2 = f'result\\T_N_{n_2}_D_{D}.pickle'

    t_1 = T_N.load(T_1)
    t_2 = T_0.load(T_2)

    max_n_a = 31

    diff = t_1.T[:, :max_n_a] - t_2.T[:, :max_n_a]

    diff_mean.append(np.mean(diff))
    diff_max.append(np.max(diff))
    diff_min.append(np.min(diff))

plt.plot(n, diff_mean, label='mean($T_N(v,a) - T_{N-1}(v,a)$)')
plt.plot(n, diff_max, label='max($T_N(v,a) - T_{N-1}(v,a)$)')
plt.plot(n, diff_min, label='min($T_N(v,a) - T_{N-1}(v,a)$)')
plt.xlabel('N')
plt.ylabel('$T_N(v,a) - T_{{N-1}}(v,a)$')
plt.title(f'$D = {D}$')
plt.legend()

plt.savefig(f'img\\T_N_diff_over_N_D_{D}.png')

