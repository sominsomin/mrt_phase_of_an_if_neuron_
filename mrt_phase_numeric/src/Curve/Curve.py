import warnings
import numpy as np
import pickle
from mrt_phase_numeric.src.Curve.curve_util import resample_curve


class Curve:
    n_branch = 1  # if there are multiple branches, curves are numbered

    reference_point_index=None # point which lies on limit cycle
    reference_point = None

    reference_point_on_limit_cycle = False

    def __init__(self, points: int, n_branch: int=None):
        self.points = np.array(points)
        if n_branch:
            self.n_branch = n_branch

    @classmethod
    def init_horizontal(cls, a: float, v_min: float = 0.0, v_max: float = 1.0, v_step: float = 0.1):
        n_samples = int((v_max - v_min) * 10 + 1)

        curve = Curve(np.array([[v, a] for v in np.linspace(v_min, v_max, n_samples)]))

        return curve

    def _set_reference_point_index(self, _reference_point_index: int):
        self.reference_point_index = _reference_point_index
        self.reference_point = self.points[_reference_point_index]

    def resample_curve(self, _n_points: int, _x_min: float, _x_max: float, _x_step: float=None):
        self.points = resample_curve(self.points, n_samples=_n_points, x_min=_x_min, x_max=_x_max, x_step=_x_step)

    def trim_curve(self, return_time_list: []):
        """
        remove point with a np.nan return time
        """
        curve_trimmed = []
        rt_list_trimmed = []
        reference_point_index_trimmed = self.reference_point_index

        for i, rt in enumerate(return_time_list):
            if rt is np.nan:
                if i < self.reference_point_index:
                    reference_point_index_trimmed = reference_point_index_trimmed - 1
                elif i == self.reference_point_index and len(self.points) > 1:
                    warnings.warn('deleting safe point')
                elif i > self.reference_point_index:
                    pass
            else:
                curve_trimmed.append(self.points[i])
                rt_list_trimmed.append(rt)

        curve_trimmed = np.array([entry for i, entry in enumerate(self.points) if return_time_list[i] is not np.nan])
        t_list = np.array([entry for i, entry in enumerate(return_time_list) if return_time_list[i] is not np.nan])

        self.points = curve_trimmed
        self.reference_point_index = reference_point_index_trimmed
        return self, t_list

    def __iter__(self):
        return CurveIterator(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, index):
        return self.points[index]

    def __setitem__(self, index, value):
        self.points[index] = value

    def __repr__(self):
        return f'{self.points}'

    def save(self, filename):
        """
        save object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename: str):
        """
        load curve from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


class CurveIterator:
    def __init__(self, curve):
        self._curve = curve
        self._index = -1

    def __next__(self):
        if self._index < len(self._curve.points)-1:
            self._index += 1
            return self._curve.points[self._index]
        raise StopIteration


class Curves:
    curves = None

    def __init__(self, _curves: [Curve]):
        self.curves = _curves

    def __iter__(self):
        return CurveIterator(self)

    def __len__(self):
        return len(self.curves)

    def __getitem__(self, index):
        return self.curves[index]

    def __setitem__(self, index, value):
        self.curves[index] = value

    def __repr__(self):
        return f'{self.curves}'

    def save(self, filename):
        """
        save object to file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """
        load curve from object
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__=='__main__':
    # curve = Curve([(1, 1), (2, 2), (3, 3)])
    curve = Curve([[0, 0], [1, 1], [2, 2], [3, 3]])

    # curve.resample_curve(10)

    for point in curve:
        print(point)

    curve._set_reference_point_index(1)
    print(curve.reference_point_index)

    curves = Curves([curve, curve])


