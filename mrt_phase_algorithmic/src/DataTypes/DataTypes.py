import os
from mrt_phase_algorithmic.definitions import ROOT_DIR

class DataTypes:
    TIMESERIES_TYPE = 'timeseries'
    TIMESERIES_GRID_TYPE = 'timeseries_grid'
    STOCHASTIC_TYPE = 'stochastic'
    DETERMINISTIC_TYPE = 'deterministic'


class InitTypes:
    INIT_FROM_SCRATCH = 'init_from_scratch'
    INIT_FROM_OBJECT = 'init_from_object'
    INIT_FROM_LAST = 'init_from_last'
    INIT_FROM_DETERMINISTIC = 'init_from_deterministic'


MULTIPLE_BRANCHES_TYPE = 'multiple_branches'

data_folder_path = os.path.join(ROOT_DIR, 'data')

filepaths = {
    'limit_cycle_path': os.path.join(data_folder_path, 'input', 'limit_cycle', 'limit_cycle.txt'),
    DataTypes.TIMESERIES_TYPE: {
        DataTypes.STOCHASTIC_TYPE: {
            'curve_path': os.path.join(data_folder_path, 'results', 'isochrones', 'from_timeseries', DataTypes.STOCHASTIC_TYPE),
            'data_path': os.path.join(data_folder_path, 'input', 'timeseries'),
        }
    },
    DataTypes.TIMESERIES_GRID_TYPE: {
        DataTypes.STOCHASTIC_TYPE: {
            'curve_path': os.path.join(data_folder_path, 'results', 'isochrones', 'from_timeseries_grid', DataTypes.STOCHASTIC_TYPE),
        },
        DataTypes.DETERMINISTIC_TYPE: {
            'curve_path': os.path.join(data_folder_path, 'results', 'isochrones', 'from_timeseries_grid', DataTypes.DETERMINISTIC_TYPE),
            'data_path': os.path.join(data_folder_path, 'input', 'timeseries_grid', DataTypes.DETERMINISTIC_TYPE),
        }
    }
}

