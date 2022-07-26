import numpy as np
import matplotlib.pyplot as plt
from mrt_phase_algorithmic.src.Isochrone.Isochrone import IsochroneMultipleBranchesBaseClass

phi = 0.4
D = 0.25
filename = f'..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'
# filename = f'..\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'

isochrone = IsochroneMultipleBranchesBaseClass.load(filename)

# for curves in isochrone.curve_history:

curves_history = isochrone.curve_history
return_time_list_history = isochrone.mean_return_time_pro_point_history

fig, (ax1, ax2) = plt.subplots(2, 1)

for i, curves in enumerate(curves_history[:]):
    if i == 0 or i < 0:
        continue
    for j, curve in enumerate(curves):
        if not curve:
            pass
        elif j == 1:
        # else:
            ax1.plot(curve[:, 0], curve[:, 1])
            ax2.plot(curve[:, 0], return_time_list_history[i][j])

v_ = np.linspace(-2, 1, 100)
ax2.plot(v_, [isochrone.target_rt for v in v_], 'g--')

ax1.set_xlim([-1, 1])
ax1.set_ylim([0, 2])
ax1.set_xlabel('v')
ax1.set_ylabel('a')

ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 3])
ax2.set_xlabel('v')
ax2.set_ylabel('t')

fig.tight_layout()
