import numpy as np
import matplotlib.pyplot as plt
from mrt_phase_numeric.mrt_phase.reset_and_fire.util.Curve.Curve import Curve
from mrt_phase_numeric.mrt_phase.reset_and_fire.util.Isochrone.Isochrone import IsochroneTimeSeriesGridMultipleBranches
from mrt_phase_numeric.mrt_phase.reset_and_fire.util.update import integrate_backwards, integrate_forwards, integrate_forwards_time

curve_path = f'..\\..\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0\\v_0.1\\v_0.1_D_0.0.pkl'

isochrone = IsochroneTimeSeriesGridMultipleBranches.load(curve_path)

mean_T = isochrone.target_rt

for curve in isochrone.curves:
    plt.plot(curve[:, 0], curve[:, 1])

for point in isochrone.curves[1]:
    v, a = integrate_backwards(point[0], point[1], mean_T, dt=0.01, D=0.0)
    plt.plot(v[-1], a[-1] , 'x')

for point in isochrone.curves[1]:
    v, a = integrate_backwards(point[0], point[1], mean_T/2, dt=0.01, D=0.0)
    plt.plot(v[-1], a[-1] , 'x')

# for point in isochrone.curves[1]:
#     v, a, y  = integrate_forwards_time(point[0], point[1], mean_T, dt=0.01, D=0.0)
#     plt.plot(v[-1], a[-1] , 'x')

plt.xlim([-5, 1])

plt.legend(['isochrone_0', 'isochrone_1', 'isochrone_2'])
