import numpy as np


def resample_curve_conserve_mid(_curve, n_samples=None, x_min=None, x_max=None,
                                limit_cycle_point_index=None, x_step=None):

    x_limit_cycle_point = _curve[limit_cycle_point_index, 0]
    n_lower = np.max([int((x_limit_cycle_point - x_min) / (x_max - x_min) * n_samples), 3])
    n_upper = n_samples - n_lower

    if limit_cycle_point_index is not None:
        if limit_cycle_point_index < 2:
            # resampling when there are not enough points
            curve_lower = _curve[:limit_cycle_point_index+3]
            # resample first with range greater than limit_cycle_point_index, then resample again
            curve_lower_resampled = resample_curve(curve_lower, 3, x_min, curve_lower[-1, 0])
            # take first element of resampled curve and limit cycle point
            curve_lower = np.append([curve_lower_resampled[0]], [_curve[limit_cycle_point_index]], axis=0)
            curve_lower_resampled = resample_curve(curve_lower, n_lower, x_min, curve_lower[-1, 0], x_step=x_step)
        else:
            curve_lower = _curve[:limit_cycle_point_index+1]
            curve_lower_resampled = resample_curve(curve_lower, n_lower, x_min, curve_lower[-1, 0], x_step=x_step)

        curve_upper = _curve[limit_cycle_point_index:]
        curve_upper_resampled = resample_curve(curve_upper, n_upper, curve_upper[0, 0], x_max)

        curve_resampled = np.append(curve_lower_resampled, curve_upper_resampled[1:], axis=0)
        limit_cycle_point_index = len(curve_lower_resampled) - 1
    else:
        curve_resampled = resample_curve(_curve, n_samples, x_min, x_max, x_step=x_step)

    return curve_resampled, limit_cycle_point_index


def extend_out_of_range_values_min(_curve, x_min):
    x = [point[0] for point in _curve]
    y = [point[1] for point in _curve]

    _x = [x_min]

    if len(_curve) > 2:
        n_jump = 2
    else:
        n_jump = 1

    m = (_curve[n_jump][1] - _curve[0][1]) / (_curve[n_jump][0] - _curve[0][0])
    _y = [_curve[0][1] - m * (_curve[0][0] - x_min)]

    _x.extend(x)
    _y.extend(y)

    x = _x
    y = _y

    return x, y


def extend_out_of_range_values_max(_curve, x_max):
    x = [point[0] for point in _curve]
    y = [point[1] for point in _curve]

    _x = [x_max]

    if len(_curve) > 2:
        n_jump = 2
    else:
        n_jump = 1

    m = (_curve[-1][1] - _curve[-1-n_jump][1]) / (_curve[-1][0] - _curve[-1-n_jump][0])
    _y = [_curve[-1][1] + m * (x_max - _curve[-1][0])]

    x.extend(_x)
    y.extend(_y)

    return x, y


def resample_curve(_curve, n_samples=None, x_min=None, x_max=None, x_step=None):

    if _curve[0][0] > x_min:
        x, y = extend_out_of_range_values_min(_curve, x_min)
        _curve = np.array(list(zip(x, y)))
    if _curve[-1][0] < x_max:
        x, y = extend_out_of_range_values_max(_curve, x_max)
        _curve = np.array(list(zip(x, y)))

    if any(np.diff(_curve[:, 0]) < 0):
        negative_indices = np.where(np.diff(_curve[:, 0]) < 0)[0]
        _curve_list = _curve.tolist()

        for negative_indice in negative_indices:
            _curve_list.pop(negative_indice)

        _curve = np.array(_curve_list)

    x = [point[0] for point in _curve]
    y = [point[1] for point in _curve]

    if not n_samples:
        n_samples = int(len(_curve)) * 2

    if x_min is None:
        x_min = np.min(x)
    if x_max is None:
        x_max = np.max(x)

    if x_step:
        x_new = np.arange(x_min, x_max + x_step, x_step)
    else:
        x_new = np.linspace(x_min, x_max, n_samples)

    y_new = np.interp(x_new, x, y)

    new_curve = [(x_new[i], y_new[i]) for i in range(len(x_new))]

    return np.array(new_curve)


def resample_curve_if_sparse(_curve, _tol):
    new_curve = []

    for i in range(len(_curve)-1):
        diff_to_next = np.array(_curve[i+1]) - np.array(_curve[i])
        dist_to_next = np.linalg.norm(diff_to_next)

        new_curve.append(_curve[i])

        if dist_to_next > _tol:
            new_point = np.array(_curve[i]) + dist_to_next/2
            new_curve.append((new_point[0], new_point[1]))

    new_curve.append(_curve[-1])

    return np.array(new_curve)


def init_curve_from_scratch(limit_cycle_, v_init_, n_points=30, v_min=0.0, v_max=1.0, a_min=0.0, a_max=2.0):
    indice = np.where(limit_cycle_[:, 0] >= v_init_)[0][0]
    limit_cycle_point = limit_cycle_[indice]

    n_lower = int((v_init_ - v_min) / (v_max - v_min) * n_points)
    n_upper = n_points - n_lower

    limit_cycle_point_index = n_lower - 1

    curve_upper = [[x, y] for x, y in zip(np.linspace(limit_cycle_point[0], v_max, n_upper),
                                          np.linspace(limit_cycle_point[1], a_max, n_upper))]
    curve_lower = [[x, y] for x, y in zip(np.linspace(v_min, limit_cycle_point[0], n_lower),
                                          np.linspace(a_min, limit_cycle_point[1], n_lower))]

    curve_lower.extend(curve_upper[1:])

    return np.array(curve_lower), limit_cycle_point_index


if __name__ == '__main__':
    from mrt_phase_algorithmic.src.util.save_util import read_curve_from_file, read_limit_cycle_point_index_from_file
    import matplotlib.pyplot as plt

    limit_cycle = read_curve_from_file(f'../../data/input/limit_cycle/limit_cycle.txt')

    curve = read_curve_from_file('..\\data\\results\\isochrones\\from_timeseries_grid\\stochastic\\D_0.5\\v_0.1\\v_0.1_D_0.5.txt')
    limit_cycle_point_index = read_limit_cycle_point_index_from_file('..\\data\\results\\isochrones\\from_timeseries_grid\\stochastic\\D_0.5\\v_0.1\\v_0.1_D_0.5_limit_cycle_point_index.txt')

    curve_resampled, limit_cycle_point_index_resampled = resample_curve_conserve_mid(curve, n_samples=10,
                                                                           x_min=0.0, x_max=1.0,
                                                                           limit_cycle_point_index=limit_cycle_point_index)

    curve_resampled_2, limit_cycle_point_index_resampled_2 = resample_curve_conserve_mid(curve[limit_cycle_point_index:], n_samples=10,
                                                           x_min=0.0, x_max=1.0,
                                                           limit_cycle_point_index=0)

    plt.plot(curve[:, 0], curve[:, 1])
    plt.plot(curve_resampled[:, 0], curve_resampled[:, 1])
    plt.plot(curve_resampled_2[:, 0], curve_resampled_2[:, 1])

    plt.ylim([0, 2])

    a = np.array([[0.01, -0.04680897],
         [0.50035444, 0.96699088],
         [0.57173237, 1.11456361],
         [0.65311031, 1.1],
         [0.64311031, 1.26213635],
          [0.74311031, 1.16213635],
         ])

    a_resampled, a_l = resample_curve_conserve_mid(a, 10, x_min=0.0, x_max=1.0, limit_cycle_point_index=4)

    plt.plot(a[:, 0], a[:, 1], 'r--')
    plt.plot(a_resampled[:, 0], a_resampled[:, 1], 'm--')


    b, l = init_curve_from_scratch(limit_cycle, 0.5,
                               n_points=15,
                               a_min=0.0, a_max=2.0,
                               v_min=0.5-0.1, v_max=0.5+0.1)
