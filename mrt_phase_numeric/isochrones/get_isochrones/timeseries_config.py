import os

from config import equation_config
from mrt_phase_numeric.src.DataTypes.DataTypes import (
    DataTypes, InitTypes)

D = 1.0
mean_T = equation_config['mean_T_dict'][D]

data_path = f'..\\..\\data\\input\\timeseries\\D_{D}'

files = os.listdir(data_path)
data_filename = os.path.join(data_path, files[0])

isochrone_config = {
    'D': D,
    'v_range': 0.015,
    'a_range': 0.015,
    'target_rt': mean_T,
    'update_jump': 0.2,
    'max_n_trajectories': 5000,
    'early_stopping_n_updates': 10,
    'smooth_curve': False,
    'data_filename': data_filename,
    'ignore_limit_cycle_point': False,
}

init_config = {
    'init_type': InitTypes.INIT_FROM_DETERMINISTIC, # 'from_scratch',
    'data_type': DataTypes.TIMESERIES_TYPE,
    'multiple_branches': True,
    'data_filename': data_filename,
    'resample': False,
    'n_points': 20,
    'save_curves': True,
    'max_n_updates': 50,
}
