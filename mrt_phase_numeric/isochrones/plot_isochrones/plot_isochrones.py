import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter

from mrt_phase_numeric.src.config import equation_config
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths, DataTypes
from mrt_phase_numeric.src.Isochrone.Isochrone import IsochroneBaseClass


data_location_timeseries_grid = '..\\data\\results\\isochrones\\from_timeseries_grid'
data_location_timeseries = '..\\data\\results\\isochrones\\from_timeseries'

_limit_cycle = read_curve_from_file(filepaths['limit_cycle_path'])

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

D = 0.25
# D = 0.25

det_file_paths = '../../data/results/isochrones/from_timeseries_grid/deterministic/D_0.0'
stochastic = f'..\\..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'
# D_1 = '..\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_2.0'


def load_curves(data_location, phi):
    file = os.listdir(os.path.join(data_location, phi))[0]
    file_path = os.path.join(data_location, phi, file)
    isochrone = IsochroneBaseClass.load(file_path)

    curves = get_best_isochrone(isochrone)

    return curves


def get_best_isochrone(isochrone: IsochroneBaseClass):
    error_history = np.array(isochrone.error_history)
    if len(error_history) < 2:
        error_history = np.array(isochrone.percent_error_history)

    idx = np.where(error_history == np.nanmin(error_history))[0][0] - 1
    # idx = len(error_history) - 1

    curves = isochrone.curve_history[idx]
    rt_times = isochrone.mean_return_time_pro_point_history[idx+1]

    return curves, rt_times


def load_isochrones(data_location):
    phi_list = os.listdir(data_location)

    all_curves = dict()
    for phi in phi_list:
        print(phi)
        curves, rt_times = load_curves(data_location, phi)
        result = {'curves': curves,
                  'rt_times': rt_times}
        all_curves[phi] = result

    return all_curves


def plot_isochrones(isochrones_list):
    # exclude = ['phi_0.4']
    # exclude = ['phi_0.6']
    exclude = []

    for key in isochrones_list.keys():
        if key in exclude:
            continue

        result = isochrones_list[key]
        curves = result['curves']
        rt_times =  result['rt_times']
        for i, curve in enumerate(curves):
            if curve.points.any() and len(curve.points) > 5: # i == 1:
                # filter = np.where(rt_times[i] - np.mean(rt_times[i]) <= 0.05)
                # curve.points = curve.points[filter]
                # if len(curve.points) > 10:
                #     curve[:, 1] = savgol_filter(curve.points[:, 1], 7, 3)
                #

                # idx_v_zero = -1
                # v = curve[idx_v_zero, 0]
                # a = curve[idx_v_zero, 1]
                #
                # if a < 3:
                #     plt.text(v, a, f'${key.split("_")[1]} $')

                plt.plot(curve[:, 0], curve[:, 1], 'b') #, label='stochastic isochrones')


if __name__=='__main__':
    _type = 'deterministic'

    # filepaths[DataTypes.TIMESERIES_GRID_TYPE][DataTypes.DETERMINISTIC_TYPE]['curve_path']

    # isochrones_deterministic = load_isochrones(det_file_paths)
    isochrones_stochastic = load_isochrones(stochastic)

    plt.plot(_limit_cycle[:, 0], _limit_cycle[:, 1], 'g.')

    plt.title(f'$D={D}$, $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={delta_a}$')
    # plt.legend(['limit cycle'])
    plt.xlim([-1, 1])
    plt.ylim([0, 3])
    plt.xlabel('v')
    plt.ylabel('a')

    # plot_isochrones(isochrones_deterministic)

    plot_isochrones(isochrones_stochastic)
    # plot_isochrones(isochrones_stochastic_D_1)

    # plt.legend([])

    plt.savefig(f'..\\img\\all_isochrones_comparison\\D_{D}.png')

    plt.show()

