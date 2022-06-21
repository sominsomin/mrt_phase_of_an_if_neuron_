import os

import numpy as np

from mrt_phase_numeric.src.Curve.curve_util import resample_curve_conserve_mid
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.src.Isochrone.Isochrone import IsochroneTimeSeries, \
    IsochroneTimeSeriesGrid, IsochroneTimeSeriesMultipleBranches, IsochroneTimeSeriesGridMultipleBranches, \
    IsochroneBaseClass
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths, DataTypes, InitTypes
from mrt_phase_numeric.src.DataTypes.TimeSeries.TimeSeries import TimeSeries
from mrt_phase_numeric.src.DataTypes.TimeSeriesGrid.TimeSeriesGrid import TimeSeriesGrid
from mrt_phase_numeric.src.Curve.Curve import Curve


class IsochroneInitHelper:
    isochrone_config = None
    config = None

    data_type = None  # data input type
    noise_type = None  # deterministic or stochastic
    isochrone_type = None  # isochrone class

    multiple_branches = False
    save_curves = False

    load_object_path = None

    object_path = None
    data_filename = None
    limit_cycle_file_path = None
    save_curve_path = None

    n_points = None

    data = None
    limit_cycle = None
    init_type = None  # from scratch, deterministic, stochastic

    resample = False  # resample curve after init

    def __init__(self, _isochrone_config: dict, _config: dict):
        self.isochrone_config = _isochrone_config
        self.config = _config

        self._unpack_config()
        self.set_paths()

        self.load_data()

    def _unpack_config(self):
        _config = self.config
        for key in _config:
            self.__setattr__(key, _config[key])

    def set_paths(self):
        D = self.isochrone_config['D']

        if D == 0.0:
            self.noise_type = DataTypes.DETERMINISTIC_TYPE
        else:
            self.noise_type = DataTypes.STOCHASTIC_TYPE

        self.save_curve_path = f'{filepaths[self.data_type][self.noise_type]["curve_path"]}\\D_{D}'

        self.limit_cycle_file_path = filepaths['limit_cycle_path']

    def set_save_paths(self, phi):
        D = self.isochrone_config['D']
        if self.init_type == InitTypes.INIT_FROM_OBJECT:
            if os.path.exists(os.path.join(self.save_curve_path, f'phi_{phi}')):
                files = [f for f in os.listdir(os.path.join(self.save_curve_path, f'phi_{phi}')) if f.endswith('.isochrone')]
                if any(files):
                    self.object_path = os.path.join(self.save_curve_path, f'phi_{phi}', files[0])
            else:
                self.object_path = None
        else:
            self.object_path = os.path.join(self.save_curve_path, f'phi_{phi}', f'phi_{phi}_D_{D}.isochrone')

    def set_load_paths(self, phi):
        if self.init_type == InitTypes.INIT_FROM_DETERMINISTIC:
            self.load_object_path = os.path.join(filepaths[DataTypes.TIMESERIES_GRID_TYPE][DataTypes.DETERMINISTIC_TYPE]["curve_path"],
                                                 'D_0.0', f'phi_{phi}', f'phi_{phi}_D_0.0.isochrone')

    def load_data(self):
        self.limit_cycle = read_curve_from_file(self.limit_cycle_file_path)

        if self.data_type == DataTypes.TIMESERIES_TYPE:
            D = self.isochrone_config['D']
            self.data = TimeSeries.load(self.data_filename)
            self.isochrone_config['D'] = self.data.timeseries_config['D']
            self.isochrone_config['dt'] = self.data.timeseries_config['dt']
            self.isochrone_config['v_min'] = np.min([np.ceil(self.data[:, 0].min()), 0.1])

        if self.data_type == DataTypes.TIMESERIES_GRID_TYPE:
            self.data = TimeSeriesGrid.load(self.data_filename)
            self.isochrone_config['v'] = self.data.v
            self.isochrone_config['a'] = self.data.a

            self.isochrone_config['v_range'] = self.data.timeseries_grid_config['v_range']
            self.isochrone_config['a_range'] = self.data.timeseries_grid_config['a_range']
            self.isochrone_config['dt'] = self.data.timeseries_grid_config['dt']

    def init_isochrone(self, _phi):
        self._unpack_config()

        if self.init_type in [InitTypes.INIT_FROM_SCRATCH, InitTypes.INIT_FROM_LAST, InitTypes.INIT_FROM_DETERMINISTIC]:
            return self.init_new(_phi)
        elif self.init_type == InitTypes.INIT_FROM_OBJECT:
            return self.init_from_object(_phi)

    def init_from_object(self, _phi=None):
        if _phi:
            self.set_save_paths(_phi)
            object_path = self.object_path
        else:
            object_path = f'pkl\\{self.data_type}.pkl'

        if object_path:
            if os.path.exists(object_path):
                isochrone = IsochroneBaseClass.load(filename=object_path)
                isochrone.config = self.isochrone_config

                # curves = self.adjust_curve(isochrone)

                self.data_filename = isochrone.data_filename
                self.load_data()
                isochrone.data = self.data

                return isochrone
            else:
                self.init_type = InitTypes.INIT_FROM_DETERMINISTIC
                return self.init_new(_phi)
        else:
            self.init_type = InitTypes.INIT_FROM_DETERMINISTIC
            return self.init_new(_phi)

    def adjust_curve(self, isochrone: IsochroneBaseClass):
        indice = np.where(self.limit_cycle[:, 0] >= isochrone.phi)[0][0]
        limit_cycle_point = self.limit_cycle[indice]

        curve_1 = isochrone.curves[1]
        idx = np.where(curve_1[:, 0] >= isochrone.phi)[0][0]

        diff_a = limit_cycle_point[1] - curve_1[idx][1]

        curves = []
        for curve in isochrone.curves:
            curve[:, 1] = curve[:, 1] + diff_a
            curves.append(curve)

        return curves

    def resolve_class(self):
        if self.data_type == DataTypes.TIMESERIES_TYPE:
            if self.multiple_branches:
                return IsochroneTimeSeriesMultipleBranches
            else:
                return IsochroneTimeSeries
        elif self.data_type == DataTypes.TIMESERIES_GRID_TYPE:
            if self.multiple_branches:
                return IsochroneTimeSeriesGridMultipleBranches
            else:
                return IsochroneTimeSeriesGrid

    def init_new(self, _phi):
        self.set_save_paths(_phi)
        self.set_load_paths(_phi)

        if self.multiple_branches:
            curves = self.init_multiple_branches(_phi)
            curve = curves[1]
        else:
            curve = self.init_curve(_phi)

        isochrone_config = self.set_isochrone_config(_phi, curve)

        cls = self.resolve_class()
        isochrone = cls(curve, self.data, isochrone_config)

        if self.multiple_branches:
            isochrone.set_curves(curves)

        return isochrone

    def set_isochrone_config(self, _phi: float, _curve: Curve):
        isochrone_config = self.isochrone_config
        isochrone_config['limit_cycle_point_index'] = _curve.reference_point_index
        isochrone_config['phi'] = _phi
        isochrone_config['save_curve_path'] = self.save_curve_path
        isochrone_config['update_jump'] = isochrone_config['update_jump'] * self.data.equation_config['delta_a']

        return isochrone_config

    def init_curve(self, phi: float):
        if self.init_type == InitTypes.INIT_FROM_SCRATCH:
            _curve = self.init_horizontal_curve(phi, v_min=self.isochrone_config['v_min'])

        elif self.init_type == InitTypes.INIT_FROM_LAST or self.init_type == InitTypes.INIT_FROM_DETERMINISTIC:
            isochrone = IsochroneBaseClass.load(filename=self.load_object_path)

            _curve = isochrone.curves[1]
            _limit_cycle_point_index = _curve.reference_point_index

            if self.resample:
                _curve, _limit_cycle_point_index = resample_curve_conserve_mid(_curve, n_samples=self.n_points,
                                                                               x_min=self.isochrone_config['v_min'], x_max=1.0,
                                                                               limit_cycle_point_index=_limit_cycle_point_index,
                                                                               x_step=0.1)

                _curve = Curve(_curve)
                _curve._set_reference_point_index(_limit_cycle_point_index)

        return _curve

    def init_horizontal_curve(self, phi, v_min=0.0, v_max=1.0, v_step=0.1, ):
        indice = int(np.floor(len(self.limit_cycle) * phi / 1.0))
        limit_cycle_point = self.limit_cycle[indice]

        curve = Curve.init_horizontal(limit_cycle_point[1], v_min, v_max)
        # insert point into curve
        curve.points = np.insert(curve.points, 0, limit_cycle_point, axis=0)
        curve.points = np.sort(curve.points, axis=0)
        curve.reference_point_index = np.where(curve[:, 0] == limit_cycle_point[0])[0][0]

        return curve

    def init_multiple_branches(self, phi):
        v_th = self.data.equation_config['v_th']
        delta_a = self.data.equation_config['delta_a']

        if self.init_type == InitTypes.INIT_FROM_SCRATCH:
            curve = self.init_horizontal_curve(phi, v_min=self.isochrone_config['v_min'])
            curve.reference_point_on_limit_cycle = True

            a = curve.points[0, 1]

            curve_upper = Curve.init_horizontal(a=a + delta_a, v_min=self.isochrone_config['v_min'], v_max=1.0)
            curve_upper.n_branch = 2
            curve_upper.reference_point_index = np.where(curve_upper[:, 0] >= 0.0)[0][0]

            curve_corner = Curve.init_horizontal(a=a - delta_a, v_min=self.isochrone_config['v_min'], v_max=1.0)
            curve_corner.n_branch = 0
            curve_corner.reference_point_index = len(curve_corner) - 1

            curves = [curve_corner, curve, curve_upper]

        elif self.init_type == InitTypes.INIT_FROM_DETERMINISTIC:
            isochrone = IsochroneBaseClass.load(filename=self.load_object_path)

            curve = isochrone.curves[1]

            curve_upper = Curve(np.array([[point[0] - v_th, point[1] + 1.0 * delta_a] for point in curve]))

            self.n_points = int(v_th - self.isochrone_config['v_min']) * 10 + 1

            curve_upper.resample_curve(self.n_points, _x_min=self.isochrone_config['v_min'], _x_max=v_th)
            curve_upper.n_branch = 2
            curve_upper.reference_point_index = np.where(curve_upper[:, 0] == 0.0)[0][0]

            curves = [isochrone.curves[0], curve, curve_upper]

        return curves


class GetIsochroneHelper:
    """
    helper class to run Isochrone calculations
    """

    config = None
    isochrone_init_helper = None

    max_n_iterations = None

    def __init__(self, _config):
        self.config = _config

        self._unpack_config()

    def _unpack_config(self):
        _config = self.config
        for key in _config:
            self.__setattr__(key, _config[key])

    def reset_simulation(self, _phi_inits, _phi_init_index):
        print('reset simulation')
        _phi_init_index += 1
        phi_init = _phi_inits[_phi_init_index]
        print(f'{phi_init}')
        isochrone = self.isochrone_init_helper.init_isochrone(phi_init)

        return isochrone, _phi_init_index


def mock_test(isochrone):
    """
    for some quick and dirty testing
    """
    mock_curve = np.array([[0, 1], [1, 2]])
    mock_t_list = [1, 2]
    isochrone.curve = mock_curve
    isochrone.mean_return_time_pro_point = mock_t_list
    isochrone.mean_return_time_pro_point_history.append(mock_t_list)
    isochrone.curve_history.append(mock_curve)
    isochrone.error_history.append(1)

    return isochrone
