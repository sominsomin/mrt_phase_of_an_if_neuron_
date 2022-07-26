import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_algorithmic.mrt_phase.reset_and_fire.util.save_util import load_npy
from mrt_phase_algorithmic.mrt_phase.reset_and_fire.from_timeseries.misc.isochrone_from_timeseries import (
    get_distinct_curves_starting_point,
    a_range
)

timeseries_data_filename = f'..\\data\\input\\timeseries\\D_0.1\\data_D_0.1_dt_0.02_N_1000000.npy'
timeseries = []


def n_indices_for_curve(_curve):
    global timeseries

    n_indices = []

    for point in _curve:
        indices = get_distinct_curves_starting_point(point, timeseries, v_range=0.05, a_range=0.05)
        n_indices.append(len(indices))

    return n_indices


if __name__ == '__main__':
    timeseries = load_npy(timeseries_data_filename)

    v = 0.5

    curve_test = [[v, a] for a in np.arange(0., 2.0, a_range)]

    curve_test = np.array(curve_test)

    plt.figure()

    n_indices = n_indices_for_curve(curve_test)

    plt.plot(curve_test[:, 1], n_indices)

    plt.xlabel('a')

    plt.title(f'n of curves for v={v}')

    plt.show()
