import pytest
from mrt_phase_algorithmic.mrt_phase.reset_and_fire.util.Curve import Curve


@pytest.mark.parametrize("points,expected",
                         [
                             ([(1, 1), (2, 2), (3, 3)], [(1, 1), (2, 2), (3, 3)]),
                             ([[1, 1], [2, 2], [3, 3]], [[1, 1], [2, 2], [3, 3]])
                         ])
def test_curve(points,expected):
    curve = Curve(points)

    for i, point in enumerate(curve):
        print(i)
        assert point == expected[i]