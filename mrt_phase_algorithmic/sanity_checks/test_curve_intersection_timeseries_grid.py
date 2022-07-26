import numpy as np
import matplotlib.pyplot as plt
import random
import time

from mrt_phase_algorithmic.mrt_phase.reset_and_fire.util.save_util import load_npy
from mrt_phase_algorithmic.mrt_phase.reset_and_fire.util.Isochrone.Isochrone import IsochroneTimeSeriesGrid
from mrt_phase_algorithmic.mrt_phase.reset_and_fire.config import mean_T_dict

timeseries_data_filename = f'../../data\\input\\timeseries_grid\\deterministic\\data_D_0.0_dt_0.02_v_range_0.01_a_range_0.01.npy'
# timeseries_data_filename = f'..\\data\\input\\timeseries_grid\\stochastic\\D_0.1\\'
timeseries = []

D = 0.01
mean_T = mean_T_dict[D]

config = {
    'D': D,
    'v_range': 0.02,
    'a_range': 0.02,
    'v_min': None,
    'mean_T': mean_T,
    'dt': 0.02,
    'update_jump': 0.1,
}


def plot_crossings_oop(point, _curve, _type='fast'):
    global timeseries
    isochrone = IsochroneTimeSeriesGrid(_init_curve=_curve, _data=timeseries, _config=config)

    trajectories = isochrone.get_trajectories(point)

    # plt.scatter(timeseries[:, 0], timeseries[:, 1])
    plt.plot(_curve[:, 0], _curve[:, 1], 'c')
    plt.plot(_curve[:, 0], _curve[:, 1], 'cx')
    plt.plot(point[0], point[1], 'rx')

    for trajectory in trajectories[0:1]:

        start = time.time()
        if _type:
            intersection_indice = isochrone.get_curve_intersection_indice(trajectory, _type=_type)
        else:
            intersection_indice = isochrone.get_curve_intersection_indice(trajectory)
        stop = time.time()

        print(f'{_type} : {stop-start}')

        plt.plot(trajectory[0, 0], trajectory[0, 1], 'yx') # plot point starting trajectory
        plt.plot(trajectory[:intersection_indice + 1, 0], trajectory[:intersection_indice + 1, 1], 'k-')
        plt.plot(trajectory[intersection_indice, 0], trajectory[intersection_indice, 1], 'mx')
        # plt.title(f'{_type}')

    plt.legend(['curve', 'curve points', 'point', 'start_point_trajectory', 'trajectory', 'intersection'])


if __name__ == '__main__':
    timeseries = load_npy(timeseries_data_filename)
    v = load_npy(timeseries_data_filename.split('.npy')[0] + '_v.npy')
    a = load_npy(timeseries_data_filename.split('.npy')[0] + '_a.npy')

    config['v'] = v
    config['a'] = a

    n_points = 30

    curve_test = [[x, y] for x, y in zip(np.linspace(0., 1.0, n_points),
                                         np.linspace(0., 2.0, n_points))]

    rand_factor = 0.0

    curve_test = [[point[0] + rand_factor * random.random(), point[1] + 0.0 * random.random()] for point in curve_test]

    curve_test = np.array(curve_test)

    idx_point = 15

    plt.figure()
    # plot_crossings([.7, 1.0], curve_test)
    plot_crossings_oop(curve_test[idx_point], curve_test)

    plt.figure()
    plot_crossings_oop(curve_test[idx_point], curve_test, 'slow')

    plt.show()
