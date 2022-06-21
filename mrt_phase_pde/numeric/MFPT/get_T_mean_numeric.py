import numpy as np
import matplotlib.pyplot as plt
import pickle

from mrt_phase_numeric.src.update_equ.update import update_
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file


limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D = 0.5
dt = 0.01

v_min = -1
v_max = 1
a_min = 0
a_max = 4

n_v = int(v_max - v_min) * 10 + 1
n_a = int(a_max - a_min) * 10 + 1

# n_v = 20
# n_a = 40

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 500


def update_until_line_cross(v, a):
    if v <= v_thr:
        not_crossed_yet = True
        n_timesteps = 0

        v_new = v
        a_new = a

        while not_crossed_yet:
            v_new, a_new, y_new, offset = update_(v_new, a_new, dt=dt, D=D)
            n_timesteps += 1

            if v_new >= v_thr:
                not_crossed_yet = False

        return n_timesteps

    else:
        return update_until_line_cross_greater_v_max(v, a)


def update_until_line_cross_greater_v_max(v, a):
    not_crossed_yet = True
    n_timesteps = 0

    v_new = v
    a_new = a
    y_new = 0

    thr_crossed = False

    while not_crossed_yet:
        v_new, a_new, y_new, offset = update_(v_new, a_new, y=y_new, dt=dt, D=D)
        n_timesteps += 1

        if v_new >= v_thr and thr_crossed:
            not_crossed_yet = False

        if y_new == 1:
            thr_crossed = True

    return n_timesteps


def get_mean_rt(v, a):
    rt = []
    for i in range(n_trajectories):
        n_timesteps = update_until_line_cross(v, a)
        rt.append(n_timesteps * dt)

    return np.mean(rt)


def get_mean_T():
    T = np.zeros((len(v), len(a)))

    for i, _v in enumerate(v):
        for j, _a in enumerate(a):
            print(i, j)
            mean_rt = get_mean_rt(_v, _a)
            T[i, j] = mean_rt

    return T


phi_ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

if __name__ == '__main__':
    T = get_mean_T()

    V, A = np.meshgrid(v, a)

    # with open(f'T_D_{D}_v_thr_{v_thr}.pkl', 'wb') as f:
    #     pickle.dump(T, f)

    plt.contourf(V, A, T.transpose(), levels=20)
    plt.xlabel('v')
    plt.ylabel('a')
    plt.colorbar()

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1])

    plt.title(f'mean time to cross threshold (n trajectories {n_trajectories}), D = {D}, v_thr = {v_thr}')

    with open(f'T_D_{D}_v_thr_{v_thr}_n_v_{n_v}_n_a_{n_a}.pkl', 'wb') as f:
        pickle.dump(T, f)

    # curves = []
    # for phi in phi_:
    #     filename = os.path.join(filepaths[DataTypes.TIMESERIES_GRID_TYPE][DataTypes.DETERMINISTIC_TYPE]["curve_path"],
    #                  'D_0.0', f'phi_{phi}', f'phi_{phi}_D_0.0.isochrone')
    #     isochrone = IsochroneBaseClass.load(filename)
    #     for curve in isochrone.curves:
    #         curves.append(curve)
    #         try:
    #             plt.plot(curve[:,0], curve[:, 1])
    #         except:
    #             pass
