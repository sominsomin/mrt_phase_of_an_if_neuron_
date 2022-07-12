import os
import copy
import pickle

import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.DataTypes.timeseries_util import (
    get_indices_slice,
    filter_indices,
    get_timeseries_slice_next_spike_interval,
    get_timeseries_slices)
from mrt_phase_numeric.src.line_intersection.line_intersection import (get_curve_intersection,
                                                                       get_curve_intersection_faster,
                                                                       get_curves_overlap)
from mrt_phase_numeric.src.DataTypes.DataTypes import DataTypes
from mrt_phase_numeric.src.Curve.Curve import Curve


def trim_curve(curve, t_list):
    curve = np.array([entry for i, entry in enumerate(curve) if t_list[i] is not np.nan])
    t_list = np.array([entry for i, entry in enumerate(t_list) if t_list[i] is not np.nan])
    return curve, t_list


class IsochroneBaseClass:
    """
    Isochrone Base Class
    """
    type = 'base_class'

    multiple_branches = False

    curve = []
    curve_history = []
    data = []
    mean_return_time_pro_point = []
    mean_return_time_pro_point_history = [[]]
    all_return_times_list_history = []
    all_rt_history = None
    error_history = [np.nan]
    config = None

    max_n_trajectories = 2000
    update_jump = 0.05
    tol = 0.001

    min_dist_points_curve = 0.05
    max_dist_points_curve = 0.1

    phi = None
    save_curve_path = None
    D = None
    target_rt = None
    dt = None
    percent_error = None
    limit_cycle_point_index = None
    limit_cycle_point = None
    skip_reference_points = False
    n_timesteps_one_cycle = None
    max_n_indices = None
    v_range = None
    a_range = None

    v_min = None
    a_min = None

    v_th = 1

    v = None
    a = None

    n_points = None

    smooth_curve = False
    early_stopping_n_updates = None
    ignore_limit_cycle_point = False

    data_filename = None

    n_updates = 0  # internal update counter

    debug_mode = False

    def __init__(self, _init_curve: np.array, _data, _config: dict):
        self.init_empty_values()

        self.curve = _init_curve
        self.data = _data
        self.config = _config

        self._unpack_config()
        self._init_values()
        self._set_values()

        if not self.multiple_branches:
            self.curve_history.append(_init_curve)

    def init_empty_values(self):
        """
        make sure that some class values are empty
        problems with poi
        """
        # init to empty
        self.curve_history = []
        self.mean_return_time_pro_point_history = [[]]
        self.error_history = [np.nan]

    def _unpack_config(self):
        """
        unpack config dict and set as variables
        """
        _config = self.config
        for key in _config:
            self.__setattr__(key, _config[key])

    def _init_values(self):
        """
        init some values after config has been read
        """
        if self.target_rt is None:
            self.target_rt = 2.0
        self.n_timesteps_one_cycle = int(self.target_rt / self.dt)
        self.max_n_indices = 10 * self.n_timesteps_one_cycle

        if self.limit_cycle_point_index is not None:
            self.limit_cycle_point = self.curve[self.limit_cycle_point_index]

        self.n_points = int((self.v_th - self.v_min) / 0.1)

        if self.ignore_limit_cycle_point:
            self.limit_cycle_point_index = None

    def _set_values(self):
        """
        set some initial values based on config
        """
        if self.target_rt is None:
            raise Exception(Warning, 'mean return time not set')

        if self.v is not None and self.a is not None:
            self.v_min = np.min(self.v)
            self.a_min = np.min(self.a)

    def set_target_T(self):
        return self.target_rt

    def update_isochrone_single_iteration(self):
        """
        update isochrone
        get mean return time for each point on curve
        update curve based on crossing time
        """
        self.setup()

        return_times = self.get_return_times()
        self.target_rt = self.set_target_T(return_times)
        curve_new = self.update_curve(_return_times_list=return_times)

        self.save_history(curve_new, return_times)
        self.trim_curve()

    def save_history(self, curve_new: Curve, _t_list: list):
        """
        append results of iteration to history
        """
        self.mean_return_time_pro_point = _t_list
        self.mean_return_time_pro_point_history.append(_t_list)
        self.curve = curve_new
        self.curve_history.append(curve_new)
        self.error_history.append(self.error)

        self.n_updates += 1

    def setup(self):
        """
        skeleton function for overriding
        """
        pass

    def get_return_times(self, _curve: Curve=None):
        """
        for each point in curve, calculate return time to the curve
        """
        if _curve is None:
            _curve = self.curve

        return self.get_return_times_for_curve(_curve=_curve)

    def get_return_times_for_curve(self, _curve: Curve=None):
        """
        for each point in curve, calculate return time to the curve
        """
        return_times_list = list()

        for i, point in enumerate(_curve):
            print(i, point)

            if self.debug_mode:
                plt.clf()
                if hasattr(self, "curves"):
                    color_scheme = ['r', 'b', 'm']
                    for l, curve in enumerate(self.curves):
                        _plot(curve, color_scheme[l], label=f'isochrone branch {l}')
                else:
                    _plot(self.curve)
                plt.plot(point[0], point[1], 'rx')
                plt.xlabel('v')
                plt.ylabel('a')
                plt.legend()

            rt = self.get_return_time_for_point(_curve, point)
            print(rt)

            return_times_list.append(rt)

        return return_times_list

    def get_return_time_for_point(self, _curve: Curve, _point):
        """
        mock function for class
        """
        print('Warning, define return time function')
        pass

    def trim_curve(self):
        self.curve, self.mean_return_time_pro_point = trim_curve(self.curve, self.mean_return_time_pro_point)
        if not self.ignore_limit_cycle_point:
            self.limit_cycle_point_index = np.where(self.limit_cycle_point == self.curve)[0][0]

    def update_curve(self, _curve: Curve=None, _return_times_list: list=None):
        """
        pass curve to update curve function
        """
        if not _curve:
            _curve = self.curve
        curve_new = self.update_single_curve(_curve, _return_times_list)
        self.curve = curve_new

        return curve_new

    def update_single_curve(self, _curve: Curve, _return_times_list: list):
        """
        update curve according to given return_times_list
        """
        # copy to avoid problems with pointers
        curve_new = copy.copy(_curve)

        for i, point in enumerate(_curve):
            # skip if point is on limit cycle
            if self.skip_reference_points:
                if hasattr(_curve, 'reference_point_index'):
                    if _curve.reference_point_on_limit_cycle:
                        if i == _curve.reference_point_index:
                            print(f'reference point skipped: {point}')
                            continue

            t = _return_times_list[i]
            if np.isnan(t):
                curve_new[i] = point
                continue

            diff = np.abs(self.target_rt - t)

            _point = copy.copy(point)

            if diff > self.tol:
                if t < self.target_rt:
                    da = self.update_jump * diff
                else:
                    da = -self.update_jump * diff

                # point_new = [_point[0] + dv, _point[1]]
                point_new = [_point[0], _point[1] + da]
            else:
                point_new = _point

            curve_new[i] = point_new

        return curve_new

    def has_reached_early_stopping(self):
        """
        returns True if early stopping condition has been reached
        """
        if not self.early_stopping_n_updates:
            return False
        if self.n_updates < self.early_stopping_n_updates:
            return False

        min_error_index = np.where(np.nanmin(self.error_history) == self.error_history)[0][0]

        if len(self.error_history) - 1 - min_error_index > self.early_stopping_n_updates:
            print('early stopping reached')
            return True
        else:
            return False

    def save(self):
        """
        save object to file
        """
        # deepcopy first to overwrite self.data
        object_copy = copy.deepcopy(self)
        object_copy.data = []

        folder_path = os.path.join(self.save_curve_path, f'phi_{self.phi}')
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        filename = os.path.join(folder_path, f'phi_{self.phi}_D_{self.D}.isochrone')

        with open(filename, 'wb') as f:
            pickle.dump(object_copy, f)

        print(f'file saved to {filename}')

    @classmethod
    def load(cls, filename: str):
        """
        load isochrone from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)

    @property
    def error(self):
        """
        calculate error of curve
        """
        return np.nanmean(np.abs(self.target_rt - np.array(self.mean_return_time_pro_point)))


class IsochroneSingleBranchBaseClass(IsochroneBaseClass):
    max_n_spikes = 1

    def get_curve_intersection_indice(self, _curve: Curve, _start_indice: int = 0, _data=None, _data_type: str = None,
                                      _type='fast'):
        if _data_type == DataTypes.TIMESERIES_GRID_TYPE:
            _start_indice = 0
        elif _data_type == DataTypes.TIMESERIES_TYPE:
            _data = self.data

        timeseries_slice_next_spike, next_spike_indice, next_next_spike_indice = get_timeseries_slice_next_spike_interval(
            _start_indice=_start_indice, _timeseries=_data, max_n_indices=self.max_n_indices) or (None, None, None)
        if not next_spike_indice:
            return None

        if _type == 'fast':
            intersection = get_curve_intersection_faster(_curve, timeseries_slice_next_spike)

        if intersection:
            intersection_indice = intersection[1]
            intersection_indice = intersection_indice + next_spike_indice
            return intersection_indice
        else:
            print('no intersection found')
            return None

    def get_curve_intersection_indice(self, _curve, _start_indice=0, _data=None, _data_type=None):
        if _data_type == DataTypes.TIMESERIES_GRID_TYPE:
            _start_indice = 0
        elif _data_type == DataTypes.TIMESERIES_TYPE:
            _data = self.data

        n_spikes = 0

        while n_spikes < self.max_n_spikes:
            timeseries_slice_next_spike, next_spike_indice, next_next_spike_interval = get_timeseries_slice_next_spike_interval(
                _start_indice=_start_indice,
                _timeseries=_data,
                max_n_indices=self.max_n_indices) or (None, None)

            n_spikes += 1

            intersection = get_curve_intersection_faster(self.curve, timeseries_slice_next_spike)

            if intersection:
                intersection_indice = intersection[1] + next_spike_indice
                return intersection_indice

        print('couldn\"t find intersection')
        return None


class IsochroneMultipleBranchesBaseClass(IsochroneBaseClass):
    multiple_branches = True
    type = 'multiple_branches'
    curves = None
    max_n_spikes = 4

    def set_curves(self, curves):
        self.curves = curves
        self.curve_history.append(curves)

    def set_target_T(self, return_times):
        # set mean rt time of curve 1 as target
        return np.nanmean(return_times[1])

    def setup(self):
        # self.set_phase_conditions()
        pass

    def set_phase_conditions(self):
        curve_1_zero = np.where(self.curves[1][:, 0] == 0.0)
        curve_1_v_th = np.where(self.curves[1][:, 0] == 1.0)

        if any(curve_1_zero):
            curve_1_zero_idx = curve_1_zero[0][0]
            self.curves[0][self.curves[0].reference_point_index, 1] = self.curves[1][curve_1_zero_idx, 1] - self.v_th
        if any(curve_1_v_th):
            curve_1_v_th_idx = curve_1_v_th[0][0]
            self.curves[2][self.curves[2].reference_point_index, 1] = self.curves[1][curve_1_v_th_idx, 1] + self.v_th

    def get_return_times(self):
        return_times_all_curves = []

        for i, curve in enumerate(self.curves):
            rt = self.get_return_times_for_curve(_curve=curve)
            return_times_all_curves.append(rt)

        self.all_return_times_list_history.append(return_times_all_curves)

        return_times_all_curves = self.mean_rt_times(return_times_all_curves)

        return return_times_all_curves

    def mean_rt_times(self, return_times: list):
        rt_curves = []
        for curve_rt in return_times:
            if any(curve_rt):
                rt_times_mean = []
                for rt in curve_rt:
                    # calculate mean of t
                    if isinstance(rt, list):
                        # cast None to np.nan for nan_mean
                        rt = [_t if _t else np.nan for _t in rt]
                        rt = np.nanmean(rt)
                    rt_times_mean.append(rt)
                rt_curves.append(rt_times_mean)
            else:
                rt_curves.append(curve_rt)

        return rt_curves

    def get_curve_intersection_indice(self,
                                      _curve: Curve,
                                      _start_indice=0,
                                      _data=None, _data_type=None):
        if _data_type == DataTypes.TIMESERIES_GRID_TYPE:
            _start_indice = 0
        elif _data_type == DataTypes.TIMESERIES_TYPE:
            _data = self.data

        timeseries_slices, next_spike_indices = get_timeseries_slices(_start_indice, _data, self.max_n_spikes,
                                                                      self.max_n_indices)

        if timeseries_slices is None:
            return None

        for n_thr_crossings, timeseries_slice in enumerate(timeseries_slices):

            n_branch = self.get_possible_n_branch_for_crossing(_curve.n_branch, n_thr_crossings)

            if n_branch is None:
                if self.debug_mode: _plot(timeseries_slice, 'b--')
                continue
            # skip if n_branch is higher than there are branches, can happen when trajectory walks around
            elif n_branch >= len(self.curves):
                # print(f'n_branch {n_branch}, n_thr {n_thr_crossings}, current_n_brach: {_curve.n_branch}')
                return None
            # skip if branch doesn't exist, e.g. when there's no zero branch
            elif not self.curves[n_branch].points.any():
                continue

            target_branch = self.curves[n_branch]

            dx, dy = get_curves_overlap(target_branch, timeseries_slice)

            if np.abs(dy) <= 0.01 and dy <= 0:
                # print('close to edge')
                timeseries_len_previous_traj = int(
                    np.sum([len(timeseries_slices[i]) for i in range(n_thr_crossings)]))

                if self.data_type == DataTypes.TIMESERIES_GRID_TYPE:
                    intersection_indice = len(timeseries_slice) + timeseries_len_previous_traj
                else:
                    intersection_indice = len(timeseries_slice) + timeseries_len_previous_traj + _start_indice

                return intersection_indice

            try:
                # intersection = get_curve_intersection(self.curves[n_branch], timeseries_slice)
                intersection = get_curve_intersection_faster(target_branch, timeseries_slice)
            except:
                intersection = None
                print('intersection failed')

            if intersection:
                # add length of previous trajectories
                timeseries_len_previous_traj = int(
                    np.sum([len(timeseries_slices[i]) for i in range(n_thr_crossings)]))
                if self.data_type == DataTypes.TIMESERIES_GRID_TYPE:
                    intersection_indice = intersection[1] + timeseries_len_previous_traj
                else:
                    intersection_indice = intersection[1] + timeseries_len_previous_traj + _start_indice

                if self.debug_mode:
                    _plot(timeseries_slice[:intersection[1]], 'b--')
                    plt.plot(timeseries_slice[intersection[1], 0],
                             timeseries_slice[intersection[1], 1], 'ro')
                    # _plot(self.curves[n_branch])

                return intersection_indice

            if self.debug_mode: _plot(timeseries_slice, 'b--')

        print('couldnt find intersection')
        return None

    def get_possible_n_branch_for_crossing(self, n_current_branch: int, n_thr_crossings: int):
        """
        get possible branches which can be crossed (almost correct: #ende = #start - 1 + #n_th_crossing )

        n_thr_crossings : n_current_branch : n_branch
        """
        target_n = n_current_branch - 1 + n_thr_crossings

        if target_n < 0:
            return None
        else:
            return target_n

    def update_curve(self, _curve: Curve=None, _return_times_list=None):
        """
        pass curve to update curve function
        """
        all_new_curves = []

        for i, curve in enumerate(self.curves):
            # if i == 0:
            #     pass
            curve_new = self.update_single_curve(curve, _return_times_list[i])
            self.curves[i] = curve_new
            all_new_curves.append(curve_new)

        return all_new_curves

    @property
    def error(self):
        """
        calculate error of curve
        """
        error = []
        for rt_list in self.mean_return_time_pro_point:
            error.append(np.nanmean(np.abs(self.target_rt - np.array(rt_list))))
        return np.nanmean(error)

    @property
    def mean_deviation(self):
        """
        calculate error of curve
        """
        error = []
        for rt_list in self.mean_return_time_pro_point:
            rt_list = np.array(rt_list)
            error.append(np.nanmean(np.abs(np.nanmean(rt_list) - rt_list)))
        return np.nanmean(error)

    def mean_deviation_n(self, n: int = -1):
        """
        calculate error of curve
        """
        error = []
        for rt_list in self.mean_return_time_pro_point_history[n]:
            rt_list = np.array(rt_list)
            error.append(np.nanmean(np.abs(np.nanmean(rt_list) - rt_list)))
        return np.nanmean(error)

    def trim_curve(self):
        for i, curve in enumerate(self.curves):
            self.curves[i], self.mean_return_time_pro_point[i] = curve.trim_curve(self.mean_return_time_pro_point[i])


def _plot(data, *args, **kwargs): #, draw_style=None):
    # if len(data) != 0:
    #     if draw_style is None:
    #         plt.plot(data[:, 0], data[:, 1])
    #     else:
    #         plt.plot(data[:, 0], data[:, 1], draw_style)
    plt.plot(data[:, 0], data[:, 1], *args, **kwargs)


def _plot_(data):
    for curve in data:
        _plot(curve)


class IsochroneTimeSeries(IsochroneSingleBranchBaseClass):
    """
    Isochrone based on timeseries data input
    """
    data_type = DataTypes.TIMESERIES_TYPE

    def get_return_time_for_point(self, _curve: Curve, _point):
        indices = self.get_distinct_curves_starting_point(_point)

        print(f'len {len(indices)}, limit: {self.max_n_trajectories}')

        if self.max_n_trajectories:
            indices = indices[:self.max_n_trajectories]
        if indices.size == 0:
            return np.nan

        t_list = []
        for indice in indices:
            if self.debug_mode:
                plt.clf()
                # if hasattr(self, "curves"):
                #     _plot_(self.curves)
                # else:
                #     _plot(self.curve)
                if hasattr(self, "curves"):
                    color_scheme = ['r', 'b', 'm']
                    for l, curve in enumerate(self.curves):
                        _plot(curve, color_scheme[l], label=f'isochrone branch {l}')
                else:
                    _plot(self.curve)
                plt.plot(_point[0], _point[1], 'rx')
                plt.xlabel('v')
                plt.ylabel('a')
                plt.legend()

            t = self.get_time_until_line_crossing(_curve, indice)
            if t is not None:
                t_list.append(t)

        return t_list

    def get_distinct_curves_starting_point(self, point):
        indices = get_indices_slice(self.data, point[0] - self.v_range, point[0] + self.v_range,
                                    point[1] - self.a_range,
                                    point[1] + self.a_range)
        indices = indices[0]
        if indices.size > 0:
            indices = filter_indices(indices, self.n_timesteps_one_cycle)

        return indices

    def get_time_until_line_crossing(self, _curve: Curve, _indice: int):
        intersection_indice = self.get_curve_intersection_indice(_curve=_curve, _start_indice=_indice,
                                                                 _data=self.data, _data_type=self.data_type)

        if intersection_indice:
            if self.debug_mode and self.data_type == DataTypes.TIMESERIES_TYPE:
                plt.plot(self.data[intersection_indice, 0],
                         self.data[intersection_indice, 1], 'ro')
            t = (intersection_indice - _indice) * self.dt
            return t
        else:
            return None


class IsochroneTimeSeriesGrid(IsochroneSingleBranchBaseClass):
    """
    Isochrone based on timeseries grid data input
    """
    data_type = DataTypes.TIMESERIES_GRID_TYPE

    def get_return_time_for_point(self, _curve: Curve, point):
        trajectories = self.get_trajectories(point)
        if trajectories is None:
            return np.nan

        t = self.get_time_until_line_crossing_trajectory(_curve, trajectories)
        return t

    def get_trajectories(self, point):
        if point[0] < self.v_min or point[1] < self.a_min:
            return None
        else:
            _i = np.where(self.v <= point[0])
            _j = np.where(self.a <= point[1])

            try:
                i = _i[0][-1]
                j = _j[0][-1]
                trajectories = self.data[i, j]

                return trajectories
            except:
                return None

    def get_time_until_line_crossing_trajectory(self, _curve: Curve, _trajectories):
        rt_list = []

        for trajectory in _trajectories:
            # if self.debug_mode: _plot(trajectory)
            intersection_indice = self.get_curve_intersection_indice(_curve=_curve, _start_indice=0,
                                                                     _data=trajectory, _data_type=self.data_type)

            if intersection_indice:
                if self.debug_mode: plt.plot(trajectory[intersection_indice, 0], trajectory[intersection_indice, 1], 'bx')
                intersection_indice = self.get_curve_intersection_indice(_curve=_curve, _start_indice=0,
                                                                         _data=trajectory,
                                                                         _data_type=self.data_type)
                t = intersection_indice * self.dt
                rt_list.append(t)

        if any(rt_list):
            t_mean = np.mean(rt_list)
            return t_mean
        else:
            return np.nan


class IsochroneTimeSeriesMultipleBranches(IsochroneMultipleBranchesBaseClass, IsochroneTimeSeries):
    """
    just overwriting get intersection function for the moment
    """
    type = 'timeseries_multiple_branches'


class IsochroneTimeSeriesGridMultipleBranches(IsochroneMultipleBranchesBaseClass, IsochroneTimeSeriesGrid):
    """
    just overwriting get intersection function for the moment
    """
    type = 'timeseries_grid_multiple_branches'
