import numpy as np
import pickle

from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths, DataTypes


def get_indices_slice(data, v_min, v_max, a_min, a_max):
    """
    get indices for points which are within given limits
    """
    indices = np.where((data[:, 0] > v_min) & (data[:, 0] < v_max) &
                       (data[:, 1] > a_min) & (data[:, 1] < a_max))
    return indices


def filter_indices(indices, n_timesteps_one_cycle):
    """
    reduce list of adjacent indices to only neighbouring indices
    """
    diff = list(np.diff(indices) > n_timesteps_one_cycle)
    diff.append(False)

    final_indices = indices[diff]

    return final_indices


class TimeSeries():
    timeseries = None
    equation_config = None
    timeseries_config = None

    v = None
    a = None
    y = None

    def __init__(self, _timeseries=None, _y=None, _timeseries_config=None, _equation_config=None):
        self.timeseries = _timeseries
        self.y = _y
        self.timeseries_config = _timeseries_config
        self.equation_config = _equation_config

        self._calc_mean_cycle_time()

    def _calc_mean_cycle_time(self):
        n_spikes = len(np.where(np.array(self.y) == 1)[0])
        total_t = self.timeseries_config['dt'] * len(self.timeseries)

        self.mean_T = total_t/n_spikes

    @classmethod
    def load(cls, filename):
        """
        load timeseries from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def save(self, filename=None):
        """
        save timeseries as pickle
        """
        if not filename:
            D = self.timeseries_config["D"]
            dt = self.timeseries_config["dt"]
            n_cycles = self.timeseries_config["n_cycles"]
            filepath = filepaths[DataTypes.TIMESERIES_TYPE][DataTypes.STOCHASTIC_TYPE]["data_path"]

            filename = f'{filepath}\\D_{D}\\data_D_{D}_dt_{dt}_N_{n_cycles}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    def __getitem__(self, index):
        return self.timeseries[index]

    def __len__(self):
        return len(self.timeseries)
