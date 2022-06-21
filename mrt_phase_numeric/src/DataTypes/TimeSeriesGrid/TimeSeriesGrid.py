import numpy as np
import pickle

from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths, DataTypes
from mrt_phase_numeric.src.DataTypes.DataBaseClass import DataBaseClass


class TimeSeriesGrid():
    timeseries_grid_data = None
    v = None
    a = None

    timeseries_grid_config = None
    equation_config = None

    def __init__(self, _timeseries_grid_data=None, _v=None, _a=None, _timeseries_grid_config=None, _equation_config=None):
        self.timeseries_grid_data = _timeseries_grid_data
        self.v = _v
        self.a = _a

        self.timeseries_grid_config = _timeseries_grid_config
        self.equation_config = _equation_config

    @classmethod
    def load(cls, filename):
        """
        load timeseries from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __getitem__(self, index):
        return self.timeseries_grid_data[index]

    def __len__(self):
        return len(self.timeseries_grid_data)

    def save(self, filename=None):
        """
        save timeseries as pickle
        """
        if not filename:
            D = self.timeseries_grid_config["D"]
            dt = self.timeseries_grid_config["dt"]
            v_range = self.timeseries_grid_config["v_range"]
            a_range = self.timeseries_grid_config["a_range"]
            if D == 0.0:
                filepath = filepaths[DataTypes.TIMESERIES_GRID_TYPE][DataTypes.DETERMINISTIC_TYPE]["data_path"]

            filename = f'{filepath}\\data_D_{D}_dt_{dt}_v_range_{v_range}_a_range_{a_range}.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
