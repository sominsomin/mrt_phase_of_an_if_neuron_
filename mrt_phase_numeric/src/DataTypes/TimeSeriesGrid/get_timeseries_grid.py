import os
import numpy as np

from mrt_phase_numeric.src.update_equ.update import update_
from mrt_phase_numeric.src.DataTypes.TimeSeriesGrid.TimeSeriesGrid import TimeSeriesGrid
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths, DataTypes


folder_path = filepaths[DataTypes.TIMESERIES_GRID_TYPE][DataTypes.DETERMINISTIC_TYPE]['data_path']

D = 0.0
dt = 0.01

# _type = 'stochastic'
_type = 'deterministic'

timeseries_grid_config = {
    'D': D,
    'dt': dt,
    'n_timeseries': 1,
    'n_cycles': 3,
    'v_range': 0.01,
    'a_range': 0.01,
    'v_min': -3,
    'v_max': 1,
    'a_min': 0,
    'a_max': 5,
}

n_timeseries = timeseries_grid_config['n_timeseries']
n_cycles = timeseries_grid_config['n_cycles']

v_range = timeseries_grid_config['v_range']
a_range = timeseries_grid_config['a_range']

v_min = timeseries_grid_config['v_min']
v_max = timeseries_grid_config['v_max']

a_min = timeseries_grid_config['a_min']
a_max = timeseries_grid_config['a_max']

n_v = int((v_max - v_min) / v_range)
n_a = int((a_max - a_min) / a_range)

# TODO dont forget to remove round here
_v = [np.round(v, 2) for v in np.arange(v_min, v_max+v_range, v_range)]
_a = [np.round(a, 2) for a in np.arange(a_min, a_max+a_range, a_range)]


def get_trajectory(v_init, a_init):
    n_spikes = 0

    v_ = [v_init]
    a_ = [a_init]
    y_ = [0]

    while n_spikes <= n_cycles:
        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt, D=D)
        v_.append(v_new)
        a_.append(a_new)
        y_.append(y_new)

        if y_new == 1:
            n_spikes += 1

    # run loop once more for reset
    v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt)
    v_.append(v_new)
    a_.append(a_new)
    y_.append(y_new)

    return np.array(list(zip(v_, a_)))


if __name__ == '__main__':
    data = [[[] for j in range(len(_a))] for i in range(len(_v))]

    for i, v in enumerate(_v):
        print(f'v {v}')
        for j, a in enumerate(_a):
            for n in range(n_timeseries):
                trajectory = get_trajectory(v, a)
                data[i][j].append(trajectory)

    data = np.array(data, dtype='object')

    timeseries_grid = TimeSeriesGrid(_timeseries_grid_data=data, _v=_v, _a=_a,
                                     _timeseries_grid_config=timeseries_grid_config, _equation_config=equation_config)

    timeseries__grid_path = os.path.join(folder_path, f'D_{D}')
    if not os.path.exists(timeseries__grid_path):
        os.makedirs(timeseries__grid_path)

    timeseries_grid_filename = os.path.join(timeseries__grid_path, f'data_D_{D}_dt_{dt}_v_range_{v_range}_a_range_{a_range}.pkl')

    timeseries_grid.save(timeseries_grid_filename)
