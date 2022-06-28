import numpy as np
import matplotlib.pyplot as plt

from config import equation_config
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.Isochrone.Isochrone import IsochroneMultipleBranchesBaseClass


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

phi = 0.4
D = 0.25
filename = f'..\\..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'

isochrone = IsochroneMultipleBranchesBaseClass.load(filename)

_limit_cycle = read_curve_from_file(filepaths['limit_cycle_path'])

# for curves in isochrone.curve_history:

curves_history = isochrone.curve_history
return_time_list_history = isochrone.mean_return_time_pro_point_history

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(_limit_cycle[:, 0], _limit_cycle[:, 1], 'g.', label='deterministic limit cycle')
v_ = np.linspace(-2, 1, 100)
ax2.plot(v_, [isochrone.target_rt for v in v_], 'g--', label='target return time')

color_scheme = ['b', 'm']
label_names = ['deterministic isochrone', 'stochastic isochrone']

for l, i in enumerate([1, -1]):
    curves = curves_history[i]
    for j, curve in enumerate(curves):
        if not curve:
            pass
        elif j == 1:
        # else:
            ax1.plot(curve[:, 0], curve[:, 1], color_scheme[l], label=label_names[l])
            ax1.plot(curve[:, 0], curve[:, 1], color_scheme[l] + 'x')
            ax2.plot(curve[:, 0], return_time_list_history[i][j], color_scheme[l], label='return time ' + label_names[l])
            ax2.plot(curve[:, 0], return_time_list_history[i][j], color_scheme[l] + 'x')

ax1.set_xlim([-1, 1])
ax1.set_ylim([0, 3])
ax1.set_xlabel('v')
ax1.set_ylabel('a')
ax1.set_title(f'$D={D}$')
# , $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={delta_a}$

# ax1.legend(['deterministic limit cycle', 'deterministic isochrone', 'stochastic isochrone'])
ax1.legend()

ax2.set_xlim([-1, 1])
ax2.set_ylim([1, 2.5])
ax2.set_xlabel('v')
ax2.set_ylabel('return time t')

# ax2.legend(['target return time', 'return time deterministic isochrone', 'mean return time stochastic isochrone'])
ax2.legend()

plt.savefig(f'..\\img\\single_isochrone_comparison\\D_{D}_phi_{phi}.png')
fig.tight_layout()
