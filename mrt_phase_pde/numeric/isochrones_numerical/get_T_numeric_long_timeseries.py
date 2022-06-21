import numpy as np
import matplotlib.pyplot as plt
import pickle

from mrt_phase_numeric.src.update_equ.update import update_, integrate_forwards
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones


limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D = 0.0
dt = 0.01

v_min = -1
v_max = 0.99
a_min = 0
a_max = 4

n_v = int(np.ceil(v_max - v_min)) * 20 + 1
n_a = int(np.ceil(a_max - a_min)) * 10 + 1

# n_v = 20
# n_a = 40

v = np.linspace(v_min, v_max, n_v)
a = np.linspace(a_min, a_max, n_a)

v_thr = 1.0
n_trajectories = 1

n_thr_crossings = 4


def update_until_line_cross(v, a):
    v_, a_, y_ = integrate_forwards(v, a, n_thr_crossings, D, dt)

    n_timesteps = len(v_)
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
        print(f'v : {_v}')
        for j, _a in enumerate(a):
            print(f'a : {_a}')
            mean_rt = get_mean_rt(_v, _a)
            T[i, j] = mean_rt

    return T


if __name__ == '__main__':
    T = get_mean_T()

    V, A = np.meshgrid(v, a)

    plt.contourf(V, A, T.transpose(), levels=30)
    plt.xlabel('v')
    plt.ylabel('a')
    plt.colorbar()

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1])

    plt.title(f'mean first passage time to do {n_thr_crossings} threshold-crossings\n(n trajectories {n_trajectories}), $D = {D}$, $v_{{thr}} = {v_thr}$')

    plt.savefig(f'../img/long_timeseries_mfpt_solution_D_{D}_n_trjectories_{n_trajectories}_n_trh_crossings_{n_thr_crossings}.png')
    # plt.savefig(f'../img/comparison_long_timeseries_mfpt_solution_D_{D}_n_trjectories_{n_trajectories}_n_trh_crossings_{n_thr_crossings}.png')

    with open(
            f'T_mean\\T_D_{D}_v_thr_{v_thr}_n_v_{n_v}_n_a_{n_a}_dt_{dt}_n_trajectories_{n_trajectories}_n_thr_crossings_{n_thr_crossings}.pkl', 'wb') as f:
        pickle.dump(T, f)


    def plot_isochrones(isochrones_list, draw=None):
        legend_str = []
        for key in isochrones_list.keys():
            curves = isochrones_list[key]
            for i, curve in enumerate(curves):
                if curve.points.any():
                    if draw:
                        plt.plot(curve[:, 0], curve[:, 1], draw)
                    else:
                        plt.plot(curve[:, 0], curve[:, 1])

    det_file_paths = '..\\..\\..\\mrt_phase_numeric\\data/results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    stochastic = f'..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    # isochrones = load_isochrones(det_file_paths)
    isochrones = load_isochrones(det_file_paths)
    plot_isochrones(isochrones, 'g--')
    plt.xlim([-1, 1])
    #
    # plt.savefig(f'../img/comparison_long_timeseries_mfpt_solution_D_{D}_n_trjectories_{n_trajectories}_n_trh_crossings_{n_thr_crossings}.png')
    #
