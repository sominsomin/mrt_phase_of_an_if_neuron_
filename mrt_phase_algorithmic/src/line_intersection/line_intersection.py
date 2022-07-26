import shapely
from shapely.geometry import LineString, Point
import matplotlib.pyplot as plt
import numpy as np


def intersects(point_1, point_2, point_3, point_4):
    line1 = LineString([point_1, point_2])
    line2 = LineString([point_3, point_4])

    return line1.intersects(line2)


def intersect_shapely(curve_1, curve_2):
    line1 = LineString(curve_1)
    line2 = LineString(curve_2)

    return list(line1.intersection(line2))[0]


def get_coarse_curve(_curve, _n_splits):
    stepsize = int(len(_curve) / _n_splits)

    curve_temp = [_curve[i * stepsize] for i in range(_n_splits - 1)]
    curve_temp.append(_curve[-1])

    curve_temp = np.array(curve_temp)

    return curve_temp, stepsize


def plot_test(curve_1, curve_2, curve_2_coarse, curve_1_small, curve_2_small):
    """
    helper function for debugging get_curve_intersection_faster
    """
    plt.figure()

    plt.plot(curve_1[:, 0], curve_1[:, 1], 'm-')
    plt.plot(curve_2[:, 0], curve_2[:, 1], 'm-')
    plt.plot(curve_2_coarse[:, 0], curve_2_coarse[:, 1], 'r-')
    plt.plot(curve_1_small[:, 0], curve_1_small[:, 1], 'g-')
    plt.plot(curve_2_small[:, 0], curve_2_small[:, 1], 'g-')

    plt.show()


def slice_curves(curve_1, curve_2, stepsize, i_coarse, j_coarse, i_range):
    # get smaller curves where the intersection has been found

    # restrict slicing
    curve_1_slice_low = np.max([i_coarse - i_range, 0])
    curve_1_slice_high = np.min([i_coarse + i_range, len(curve_1) - 1])

    curve_1_small = curve_1[curve_1_slice_low:curve_1_slice_high]

    # +1 because of the slicing
    curve_2_slice_low = np.max([(j_coarse - 1) * stepsize, 0])
    curve_2_slice_high = np.min([((j_coarse * stepsize) + 1), len(curve_2) - 1])

    curve_2_small = curve_2[curve_2_slice_low:curve_2_slice_high]

    return curve_1_small, curve_2_small


def get_curve_intersection_faster(curve_1, curve_2):
    """
    create a coarse version of curve_1 (resample to smaller one) and find intersection from there at first,
    then rerun with smaller version of the curve
    """
    if not curves_overlap(curve_1, curve_2):
        return None

    curves_switched = False

    # longer curve should be curve_2, switch variables in case
    if len(curve_1) > len(curve_2):
        _curve_1 = curve_2
        _curve_2 = curve_1
        curve_1 = _curve_1
        curve_2 = _curve_2

        curves_switched = True

    n_splits = 20
    curve_2_coarse, stepsize = get_coarse_curve(curve_2, n_splits)

    i_coarse, j_coarse = get_curve_intersection(curve_1, curve_2_coarse) or (None, None)

    if not i_coarse:
        # could happen that because of the sampling curves pass each other
        # calculate on complete curve
        i, j = get_curve_intersection(curve_1, curve_2) or (None, None)
        if i is None:
            return None
        if curves_switched:
            return j, i
        else:
            return i, j
    else:
        # it can be that through downsampling the curve can walk around in reality
        for i_range in [2, 4]:
            # TODO there is some doubling here
            # get smaller slices of curve_1 and curve_2 where the intersection supposedly happens
            curve_1_small, curve_2_small = slice_curves(curve_1, curve_2, stepsize, i_coarse, j_coarse, i_range)
            _i, _j = get_curve_intersection(curve_1_small, curve_2_small) or (None, None)
            if _i:
                break

        if not _i:
            # print('curve intersection failing on slice, retry on complete curve')
            i, j = get_curve_intersection(curve_1, curve_2) or (None, None)
            if i is None:
                print('couldn\'t find intersection, passing by')
                return None
            if i:
                if curves_switched:
                    return j, i
                else:
                    return i, j

        # TODO probably i is i_coarse + _i or smth
        i = i_coarse
        j = (j_coarse - 1)*stepsize + _j

        if curves_switched:
            return j, i
        else:
            return i, j


def curves_overlap(curve_1, curve_2):
    dx, dy = get_curves_overlap(curve_1, curve_2)

    if dx >= 0 and dy >= 0:
        return True
    else:
        return False


def get_curves_overlap(curve_1, curve_2):
    curve_1_min_x = np.min(curve_1[:, 0])
    curve_1_max_x = np.max(curve_1[:, 0])
    curve_1_min_y = np.min(curve_1[:, 1])
    curve_1_max_y = np.max(curve_1[:, 1])

    curve_2_min_x = np.min(curve_2[:, 0])
    curve_2_max_x = np.max(curve_2[:, 0])
    curve_2_min_y = np.min(curve_2[:, 1])
    curve_2_max_y = np.max(curve_2[:, 1])

    dx = min(curve_1_max_x, curve_2_max_x) - max(curve_1_min_x, curve_2_min_x)
    dy = min(curve_1_max_y, curve_2_max_y) - max(curve_1_min_y, curve_2_min_y)

    return dx, dy


def get_curve_intersection(curve_1, curve_2):
    """
    take two curves (list of tuples) and find intersection point
    returns index i (curve_1) and j (curve_2)
    """
    # stepping through each point in curve_1 and check everytime if it has crossed curve_2 yet

    if not curves_overlap(curve_1, curve_2):
        return None

    for i in range(1, len(curve_1)):
        for j in range(1, len(curve_2)):
            point_1 = curve_1[i-1]
            point_2 = curve_1[i]
            point_3 = curve_2[j-1]
            point_4 = curve_2[j]

            if intersects(point_1, point_2, point_3, point_4):
                return i, j

    return None


def get_curve_intersection_custom(curve_1, curve_2):
    if not curves_overlap(curve_1, curve_2):
        return None

    for i in range(1, len(curve_1)):
        for j in range(1, len(curve_2)):
            if curves_overlap(curve_1[i-1:i+1], curve_2[j-1:j+1]):
                return i, j

    raise Warning('intersection not found')


if __name__ == '__main__':
    x = np.arange(0, 1000)
    f = np.arange(0, 1000)
    g = np.sin(np.arange(0, 10, 0.01) * 2) * 1000 + 1

    curve_1 = np.column_stack((x, f))
    curve_2 = np.column_stack((x, g))

    get_curve_intersection(curve_1, curve_2)

    get_curve_intersection_custom(curve_1, curve_2)
