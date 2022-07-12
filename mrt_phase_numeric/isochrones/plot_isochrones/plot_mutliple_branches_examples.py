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

phi = 0.6
D = 0.0


def plot_isochrone_multiple_branches(D, phi):
    # filename = f'..\\..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'
    # filename = '..\\..\\..\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    filename = f'..\\..\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0\\phi_{phi}\\phi_{phi}_D_{D}.isochrone'

    isochrone = IsochroneMultipleBranchesBaseClass.load(filename)

    _limit_cycle = read_curve_from_file(filepaths['limit_cycle_path'])

    curves_history = isochrone.curve_history
    return_time_list_history = isochrone.mean_return_time_pro_point_history

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(_limit_cycle[:, 0], _limit_cycle[:, 1], 'g.', label='deterministic limit cycle')
    v_ = np.linspace(-2, 1, 100)
    ax2.plot(v_, [isochrone.target_rt for v in v_], 'g--', label='target return time')

    # label_names = ['deterministic isochrone', 'stochastic isochrone']

    color_scheme = ['r--', 'b--', 'm--']

    # not needed but maybe for comparison stuff
    for l, i in enumerate([-1]):
        curves = curves_history[i]
        for j, curve in enumerate(curves):
            if not curve:
                pass
            # elif j == 1:
            else:
                ax1.plot(curve[:, 0], curve[:, 1], color_scheme[j], label=f'isochrone branch {j}')
                ax1.plot(curve[:, 0], curve[:, 1], color_scheme[j] + 'x')
                ax2.plot(curve[:, 0], return_time_list_history[i][j], color_scheme[j], label=f'isochrone branch {j}')
                ax2.plot(curve[:, 0], return_time_list_history[i][j], color_scheme[j] + 'x')

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, 5])
    ax1.set_xlabel('v')
    ax1.set_ylabel('a')
    ax1.set_title(f'$D={D}$')
    # , $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={delta_a}$

    # ax1.legend(['deterministic limit cycle', 'deterministic isochrone', 'stochastic isochrone'])
    ax1.legend()

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([1.5, 2.5])
    ax2.set_xlabel('v')
    ax2.set_ylabel('return time t')

    # ax2.legend(['target return time', 'return time deterministic isochrone', 'mean return time stochastic isochrone'])
    ax2.legend()
    fig.tight_layout()

    plt.savefig(f'..\\img\\multiple_branches_examples\\multiple_branches_example_D_{D}_phi_{phi}.png')


if __name__ == '__main__':
    for _phi in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 0.9]:
        plot_isochrone_multiple_branches(D, _phi)