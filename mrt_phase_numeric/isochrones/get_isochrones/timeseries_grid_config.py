from config import equation_config
from mrt_phase_numeric.src.Isochrone.InitHelper import (InitTypes)
from mrt_phase_numeric.src.Isochrone.Isochrone import (
    DataTypes)
from mrt_phase_numeric.definitions import ROOT_DIR

D = 0.0
mean_T = equation_config['mean_T_dict'][D]

data_filename = f'{ROOT_DIR}\\data\\input\\timeseries_grid\\deterministic\\D_0.0\\data_D_0.0_dt_0.01_v_range_0.01_a_range_0.01.pkl'


isochrone_config = {
    'D': D,
    'target_rt': mean_T,
    'update_jump': 0.05,
    'early_stopping_n_updates': 3,
    'smooth_curve': False,
    'data_filename': data_filename,
    'v_min': -2,
    'a_max': 3,
    'skip_reference_points': True,
}

init_config = {
    'init_type': InitTypes.INIT_FROM_OBJECT, # 'from_scratch',
    # 'init_type': INIT_FROM_OBJECT,
    'data_type': DataTypes.TIMESERIES_GRID_TYPE,
    'multiple_branches': True,
    'data_filename': data_filename,
    'n_points': 50,
    'resample': True,
    'save_curves': True,
    'max_n_updates': 30,
}

