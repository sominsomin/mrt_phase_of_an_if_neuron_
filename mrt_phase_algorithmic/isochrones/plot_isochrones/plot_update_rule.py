import numpy as np
import matplotlib.pyplot as plt
from mrt_phase_algorithmic.src.Isochrone.Isochrone import IsochroneMultipleBranchesBaseClass
from mrt_phase_algorithmic.src.util.save_util import read_curve_from_file
from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths, DataTypes

phi = 0.5
D = 0.0
# filename = f'..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'
filename = f'..\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'

_limit_cycle = read_curve_from_file(filepaths['limit_cycle_path'])
isochrone = IsochroneMultipleBranchesBaseClass.load(filename)

# for curves in isochrone.curve_history:

curves_history = isochrone.curve_history
return_time_list_history = isochrone.mean_return_time_pro_point_history

fig, (ax1, ax2) = plt.subplots(2, 1)

i = 35
j = 1

curve = curves_history[i][j]
rt_list = return_time_list_history[i+1][j]

curve = np.delete(curve.points, 26, axis=0)
rt_list = np.delete(np.array(rt_list), 26, axis=0)

v_ = np.linspace(-2, 1, 100)
ax2.plot(v_, [isochrone.target_rt for v in v_], 'g--')

horizontal = [0.94857053 for v in v_]

ax1.plot(_limit_cycle[:, 0], _limit_cycle[:, 1], 'g.')

# ax1.plot(v_, horizontal, 'b')
# ax1.plot(v_, horizontal, 'bx')
ax1.plot(curve[:, 0], curve[:, 1], 'b')
ax1.plot(curve[:, 0], curve[:, 1], 'bx')
ax2.plot(curve[:, 0], rt_list, 'b')
ax2.plot(curve[:, 0], rt_list, 'bx')

ax1.set_xlim([-1, 1])
ax1.set_ylim([0, 2])
ax1.set_xlabel('v')
ax1.set_ylabel('a')
ax1.legend(['deterministic limit cycle'])
ax1.set_title(f'iteration {i}')

ax2.set_xlim([-1, 1])
ax2.set_ylim([0, 5])
ax2.set_xlabel('v')
ax2.set_ylabel('return time $t$')

ax2.legend(['mean cycle time  $\\bar{T}$'])

fig.tight_layout()


plt.savefig(f'img\\explain_algorithm\\deterministic_example_{i}.png')
