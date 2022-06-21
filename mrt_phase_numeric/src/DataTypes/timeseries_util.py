import numpy as np
import matplotlib.pyplot as plt


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


def get_timeseries_jumps_indeces(timeseries, indice: int, max_n_indices: int):
    """
    return indices where reset is triggered
    """
    return np.where(np.diff(timeseries[indice:indice + max_n_indices, 0]) <= -1.0)[0]


def get_next_spike_indice_from_indice(timeseries, indice: int, max_n_indices: int):
    """
    return indice where timeseries has been reset to 0
    """
    # get indice before next spike
    # difference in v is smaller than -1.0 for resets
    jump_in_timeseries = get_timeseries_jumps_indeces(timeseries, indice, max_n_indices)

    if any(jump_in_timeseries[0]):
        next_spike_indice = jump_in_timeseries[0][0]

        # add indice since we only look at a slice
        # add one to get next value after spike
        next_spike_indice = indice + next_spike_indice + 1
        if timeseries[next_spike_indice, 0] != 0.0:
            raise Exception('getting next spike indice failed')

        return next_spike_indice
    else:
        return None


def get_distinct_curves_starting_point(point, timeseries, n_timesteps_one_cycle, v_range, a_range):
    """
    get all indices for trajectories starting nearby point (given by v_range and a_range)
    """
    indices = get_indices_slice(timeseries, point[0] - v_range, point[0] + v_range, point[1] - a_range,
                                point[1] + a_range)
    indices = indices[0]
    if indices.size > 0:
        indices = filter_indices(indices, n_timesteps_one_cycle)

    return indices


def get_timeseries_slice_next_spike_interval(_start_indice: int, _timeseries, max_n_indices: int):
    """
    slice timeseries between next spike and spike after
    """
    next_spike_indice = get_next_spike_indice_from_indice(_timeseries, _start_indice, max_n_indices)
    if next_spike_indice is None:
        print('couldnt find next spike, maybe increase max_n_indices')
        return None
    next_next_spike_indice = get_next_spike_indice_from_indice(_timeseries, next_spike_indice, max_n_indices)

    if not next_spike_indice or not next_next_spike_indice:
        return None

    return _timeseries[next_spike_indice:next_next_spike_indice], next_spike_indice, next_next_spike_indice


def get_timeseries_slice_this_spike_interval(_start_indice: int, _timeseries, max_n_indices: int):
    """
    slice timeseries between _indice until spike
    """
    next_spike_indice = get_next_spike_indice_from_indice(_timeseries, _start_indice, max_n_indices)
    if next_spike_indice is None:
        print('couldnt find next spike, maybe increase max_n_indices')
        return None

    return _timeseries[_start_indice:next_spike_indice], next_spike_indice


def get_timeseries_slices(_start_indice: int, _timeseries, n_spikes: int=0, max_n_indices: int=None):
    """
    slice timeseries between _indice until spike
    """
    next_spike_indices = get_timeseries_jumps_indeces(_timeseries, _start_indice, max_n_indices)

    if any(next_spike_indices):
        if len(next_spike_indices) < n_spikes:
            print('end of timeseries reached, double check')
            # try again but more data
            next_spike_indices = get_timeseries_jumps_indeces(_timeseries, _start_indice, 2 * max_n_indices)
            if len(next_spike_indices) < n_spikes:
                return None, None
    else:
        print('end of timeseries reached, double check')
        return None, None

    next_spike_indices = [next_spike_indice + 1 for next_spike_indice in next_spike_indices[:n_spikes]]
    timeseries_slices = np.split(_timeseries[_start_indice:_start_indice+max_n_indices], next_spike_indices)

    return timeseries_slices[:n_spikes], next_spike_indices


if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mrt_phase_numeric.src.DataTypes.TimeSeries.TimeSeries import TimeSeries

    timeseries = TimeSeries().load(f'..\\data\\input\\timeseries\\D_0.5\\data_D_0.5_dt_0.01_N_100000.pkl')

    n_spikes_max = 3

    start_indice = 10000

    timeseries_slices, next_spike_indices = get_timeseries_slices(start_indice, timeseries, n_spikes=n_spikes_max, max_n_indices=1000)

    plt.plot(timeseries[start_indice:start_indice+next_spike_indices[-1], 0], timeseries[start_indice:start_indice+next_spike_indices[-1], 1])

    for i, timeseries_slice in enumerate(timeseries_slices):
        plt.plot(timeseries_slice[:, 0], timeseries_slice[:, 1], 'x')
