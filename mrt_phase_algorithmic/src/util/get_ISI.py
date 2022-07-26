import numpy as np


def get_ISI(y_, dt):
    return np.diff(list_to_spike_times(y_, dt))


def list_to_spike_times(y_, dt):
    # take list of 0 and 1
    spike_times_indexes = np.where(np.array(y_) > 0)
    spike_times = [(i+1)*dt for i in spike_times_indexes[0]]
    return spike_times
