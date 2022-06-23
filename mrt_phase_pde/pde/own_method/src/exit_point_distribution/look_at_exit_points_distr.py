import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution

D = 0.0

# epd_pde = ExitPointDistribution.load(f'epd_D_{D}_pde.pickle')
epd_sim = ExitPointDistribution.load(f'..\\..\\data\\epd_sim_D_{D}.pickle')

i = -1

for j, a in enumerate(epd_sim.a):
    plt.plot(epd_sim.a, epd_sim[i][j]*np.diff(epd_sim.a)[0])

plt.xlim([0, 5])
# plt.legend([i for i in range(len(epd_sim.a))])
plt.show()
