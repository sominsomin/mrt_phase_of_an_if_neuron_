import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_own_method.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_algorithmic.src.DataTypes.DataTypes import filepaths
from mrt_phase_algorithmic.src.util.save_util import read_curve_from_file
from config import equation_config
from mrt_phase_own_method.pde.own_method.extract_curves_from_T_N.extract_curves import get_curves_from_cs, find_nearest
from mrt_phase_own_method.pde.own_method.T_bar import T_bar
from mrt_phase_algorithmic.src.Isochrone.InitHelper import IsochroneInitHelper
from mrt_phase_algorithmic.isochrones.get_isochrones.timeseries_config import \
        isochrone_config, init_config
from config import equation_config

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

# dont forget to set D in timeseries_config
D = 0.5

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)


def reduce_curve(curve):
    curve_downsampled = []
    v = np.linspace(np.min(curve[:, 0]), np.max(curve[:, 0]), int((np.max(curve[:, 0]) - np.min(curve[:, 0])) * 10 + 1))

    for _v in v:
        idx = find_nearest(curve[:, 0], _v)
        curve_downsampled.append([_v, curve[idx, 1]])

    curve_downsampled = np.array(curve_downsampled)

    return curve_downsampled


if __name__ == '__main__':
    T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')

    min_T = np.min(T_N.T)
    max_T = np.max(T_N.T)
    mean_T = T_bar[D]

    stepsize = 0.1
    levels = np.arange(min_T, max_T, mean_T * stepsize)

    mid_idx = int(len(levels)/2) + 3

    level_2 = levels[mid_idx] + mean_T
    level_1 = levels[mid_idx]
    level_0 = levels[mid_idx] - mean_T

    levels = [level_2, level_1, level_0]
    levels.sort()

    all_curves = []
    for i, level in enumerate(levels):
        __x, __y = np.meshgrid(T_N.v, T_N.a)
        cs = plt.contour(__x, __y, T_N.T.transpose(), levels=[level])  # , v_min=-30, v_max=30)
        curve = get_curves_from_cs(cs)[0]
        curve = reduce_curve(curve)
        all_curves.append(curve)

    isochrone_init_helper = IsochroneInitHelper(_isochrone_config=isochrone_config, _config=init_config)
    isochrone = isochrone_init_helper.init_from_curve(all_curves)
    rt_all = isochrone.get_return_times()

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='deterministic limit cycle')
    colorscheme = ['r', 'b', 'y']

    for i, rt in enumerate(rt_all):
        ax1.plot(all_curves[i][:, 0], all_curves[i][:, 1], colorscheme[i])
        ax1.plot(all_curves[i][:, 0], all_curves[i][:, 1], colorscheme[i] + 'x')

        rt = rt - np.nanmean(rt) + T_bar[D]
        ax2.plot(all_curves[i][:, 0], rt, colorscheme[i])
        ax2.plot(all_curves[i][:, 0], rt, colorscheme[i] + 'x')

    v_ = np.linspace(-2, 1, 100)
    ax2.plot(v_, [T_bar[D] for v in v_], 'g--', label='$\\bar{T}$')

    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, 30])
    ax1.set_xlabel('v')
    ax1.set_ylabel('a')

    ax2.set_xlim([-1, 1])
    ax2.set_ylim([0, 2])
    ax2.set_xlabel('v')
    ax2.set_ylabel('mean return time t')

    ax1.set_title(f'$D={D}$, $\mu={mu}$, $\\tau_a={tau_a}$, $\Delta_a={delta_a}$')

    ax1.legend()
    ax2.legend()

    fig.tight_layout()

    plt.savefig(f'img\\check_mrt_D_{D}_{mid_idx}.png')
