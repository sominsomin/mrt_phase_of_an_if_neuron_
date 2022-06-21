import os
import numpy as np

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_numeric.src.DataTypes.TimeSeries.TimeSeries import TimeSeries
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths, DataTypes


folder_path = filepaths[DataTypes.TIMESERIES_TYPE][DataTypes.STOCHASTIC_TYPE]['data_path']

D = 0.1
dt = 0.01
n_cycles = 200000

config = {
    'D': D,
    'n_cycles': n_cycles,
    'dt': dt,
}

if __name__ == '__main__':
    v_init = 0
    a_init = 2.0
    v_, a_, y_ = integrate_forwards(v_init, a_init, n_cycles, D, dt)

    data = np.array(list(zip(v_, a_)))

    timeseries = TimeSeries(data, y_, _timeseries_config=config, _equation_config=equation_config)

    timeseries_path = os.path.join(folder_path, f'D_{D}')
    if not os.path.exists(timeseries_path):
        os.makedirs(timeseries_path)

    timeseries_filename = os.path.join(timeseries_path, f'data_D_{D}_dt_{dt}_N_{n_cycles}.pkl')
    timeseries.save(timeseries_filename)

    # import matplotlib.pyplot as plt
    # plt.scatter(data[:, 0], data[:, 1])
