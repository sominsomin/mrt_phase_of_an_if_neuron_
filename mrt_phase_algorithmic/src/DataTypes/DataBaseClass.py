import numpy as np
import pickle

from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths, DataTypes


class DataBaseClass:
    def __init__(self):
        pass

    def _unpack_config(self, _config):
        for key in _config:
            self.__setattr__(key, _config[key])