import numpy as np
import matplotlib.pyplot as plt

# from scipy.signal import savgol_filter

from config import equation_config
from mrt_phase_algorithmic.src.util.save_util import read_curve_from_file
from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths
from mrt_phase_algorithmic.isochrones.plot_isochrones.plot_util import load_isochrones, read_dat

data_location_timeseries_grid = '..\\data\\results\\isochrones\\from_timeseries_grid'
data_location_timeseries = '..\\data\\results\\isochrones\\from_timeseries'

_limit_cycle = read_curve_from_file(filepaths['limit_cycle_path'])

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

# D = 0.5

det_file_paths = '../../data/results/isochrones/from_timeseries_grid/deterministic/D_0.0'


def plot_isochrones(isochrones_list, draw=None):
    for key in isochrones_list.keys():
        curves = isochrones_list[key]
        for i, curve in enumerate(curves):
            if curve.points.any(): # and i == 1:
                # curve = savgol_filter(curve.points, 51, 3)
                # curve.points = curve.points[3:]
                # curve[:, 1] = savgol_filter(curve.points[:, 1], 5, 2)

                try:
                    # idx_v_zero = np.where(curve[:, 0] == 0.0)[0][0]
                    idx_v_zero = -1
                    v = curve[idx_v_zero, 0]
                    a = curve[idx_v_zero, 1]

                    if a < 3:
                        plt.text(v , a, f'${key.split("_")[1]} $') # $\\varphi$ =  \\cdot 2 \\pi$
                except:
                    pass

                if draw:
                    plt.plot(curve[:, 0], curve[:, 1], draw)
                else:
                    plt.plot(curve[:, 0], curve[:, 1])


def load_isochrones_lukas():
    path = '../../data/results/isochrones/from_lukas'
    paths = [
        f'{path}/isochrone_mu2.0_taua2.0_delta1.0_phase3.14_',
        f'{path}/isochrone_mu2.0_taua2.0_delta1.0_phase6.28.dat'
    ]

    isochrones = []
    for entry in paths:
        isochrone = read_dat(entry)
        isochrone = np.sort(isochrone, axis=1)
        isochrones.append(isochrone)

    return isochrones


if __name__=='__main__':
    _type = 'deterministic'

    isochrones_deterministic = load_isochrones(det_file_paths)

    plt.plot(_limit_cycle[:, 0], _limit_cycle[:, 1], 'g.', label='limit cycle')

    plt.title(f'$D={0.0}$, $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={delta_a}$')
    plt.xlim([-1, 1])
    plt.ylim([0, 3])
    plt.xlabel('v')
    plt.ylabel('a')

    plot_isochrones(isochrones_deterministic, 'b')

    isochrones_lukas = load_isochrones_lukas()
    plt.plot(isochrones_lukas[0][:, 0], isochrones_lukas[0][:, 1], 'b.') #, label='deterministic isochrones from Holzhausen et.al.')
    plt.plot(isochrones_lukas[1][:, 0], isochrones_lukas[1][:, 1], 'b.') #, label='deterministic isochrones')

    # legend = ['limit cycle']
    # legend.extend(legend_str)

    plt.legend()
    plt.savefig('..\\img\\isochrone_deterministic_comparison_wiht_lukas.png')

    plt.show()

