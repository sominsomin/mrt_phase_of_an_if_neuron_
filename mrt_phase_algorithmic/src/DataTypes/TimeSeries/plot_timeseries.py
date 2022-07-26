import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

from mrt_phase_algorithmic.src.DataTypes.TimeSeries.TimeSeries import TimeSeries
from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths, DataTypes
from mrt_phase_algorithmic.src.Isochrone.Isochrone import IsochroneTimeSeries

D = 1.0

folder_path = filepaths[DataTypes.TIMESERIES_TYPE][DataTypes.STOCHASTIC_TYPE]['data_path']
timeseries_path = os.path.join(folder_path, f'D_{D}')
timeseries_filename = os.path.join(timeseries_path, os.listdir(timeseries_path)[0])

timeseries = TimeSeries.load(timeseries_filename)

dv = 0.01
da = 0.01

isochrone = IsochroneTimeSeries([], timeseries, {'dt': 0.01, 'v_range': dv, 'a_range': da, 'v_th' : 1.0, 'v_min': -3.0})

v_min = np.min(timeseries.timeseries[:, 0])
v_max = np.max(timeseries.timeseries[:, 0])

a_min = np.min(timeseries.timeseries[:, 1])
a_max = np.max(timeseries.timeseries[:, 1])


v = np.arange(-2, 1, 2*dv)
a = np.arange(0, 4, 2*da)

N = np.zeros((len(v), len(a)))
for i, v_ in enumerate(v):
    print(v_)
    for j, a_ in enumerate(a):
        print(a_)
        N[i, j] = len(isochrone.get_distinct_curves_starting_point([v_, a_]))


_v, _a = np.meshgrid(v, a)

# plt.scatter(timeseries.timeseries[:, 0], timeseries.timeseries[:,1])
# plt.hist2d(timeseries.timeseries[:, 0], timeseries.timeseries[:, 1], bins =[v, a])

levels = np.linspace(0, 15000, 21)
ticks = np.linspace(0, 15000, 11)

plt.contourf(_v, _a, N.transpose(), levels=levels)
plt.colorbar(ticks=ticks)

plt.title(f'$D = {D}$, n threshold crossings = 200000 \n $dv = {dv}$, $da = {da}$')
plt.savefig(f'img\\distribution_example_D_{D}.png')

# l_f = LogFormatter(10) #, labelOnlyBase=False)
# plt.contourf(_v, _a, N.transpose(), locator=ticker.LogLocator())
# plt.colorbar(format=l_f)


with open(f'N_D_{D}_dv_{dv}_{da}.pickle', 'wb') as f:
    pickle.dump(N, f)

# max_n = 1000
# plt.figure()
# plt.scatter(timeseries.timeseries[:max_n, 0], timeseries.timeseries[:max_n, 1])
