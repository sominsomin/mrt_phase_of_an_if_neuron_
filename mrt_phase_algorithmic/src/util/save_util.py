import numpy as np


def load_npy(filename):
    with open(filename, 'rb+') as f:
        timeseries = np.load(f, allow_pickle=True)

    return timeseries


def read_curve_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        _curve = [eval(line) for line in lines]

    return np.array(_curve)


def read_limit_cycle_point_index_from_file(filename):
    with open(filename, 'r') as file:
        result = file.readlines()[0]

    return int(result)


def write_curve_to_file(filename, curve_):
    with open(filename, 'w') as file:
        for point in curve_:
            file.write(f'[{point[0]}, {point[1]}]\n')
            # file.write(str(point) + '\n')


def write_t_list_to_file(filename, t_list):
    with open(filename, 'w') as file:
        for t in t_list:
            file.write(f'[{t}]\n')


def write_data_to_file(filename, data):
    with open(filename, 'w') as file:
        file.write(str(data))


def read_t_list_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        t_list = [eval(line) for line in lines]

    return np.array(t_list)
