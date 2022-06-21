import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution

D = 0.25

epd_pde = ExitPointDistribution.load(f'epd_D_{D}_pde.pickle')
epd_sim = ExitPointDistribution.load(f'..\\own_method\\data\\epd_sim_D_{D}.pickle')

v = 0.1
a = 1.0

i = -5
j = 12

print(f'v = {epd_pde.v[i]}')
print(f'a = {epd_pde.a[j]}')

i_sim = np.where(epd_sim.v == epd_pde.v[i])[0][0]
j_sim = np.where(epd_sim.a == epd_pde.a[j])[0][0]

plt.plot(epd_pde.a, epd_pde[i][j])
plt.plot(epd_sim.a, epd_sim[i_sim][j_sim]*np.diff(epd_sim.a)[0])

plt.legend(['pde', 'sim'])

