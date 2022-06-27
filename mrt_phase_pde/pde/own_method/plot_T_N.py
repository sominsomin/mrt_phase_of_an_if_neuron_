import copy

import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.update_equ.update import integrate_forwards
from mrt_phase_numeric.src.config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_0 import T_0
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones, plot_isochrones


D = 0.0


if __name__ == '__main__':
    prob = ExitPointDistribution.load(f'data/epd_sim_D_{D}.pickle')
    # T_0 = T_0.load(f'data/T_0_D_{D}_sim.pickle')
    # T_N = T_N_class.load(f'result/T_N_15_D_{D}.pickle')
    T_N = T_N_class.load(f'data/T_0_D_{D}_sim_n_thr_15.pickle')
    # T_0 = T_0.load('data/T_0_D_0.25_sim_n_thr_1.pickle')

    __x, __y = np.meshgrid(T_N.v, T_N.a)

    plt.figure()
    plt.contourf(__x, __y, T_N.T.transpose(), levels=30)  # , v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 4])
    # plt.title(
    #     f'$T_{{{T_N.n_thr}}}(v,a)$, $D={D}$')
        # f'\n $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={Delta_a}$')

    # plt.savefig(f'img\\T_N_{l + 1}_D_{D}.png')

    det_file_paths = '..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries_grid\\deterministic\\D_0.0'
    stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    plot_isochrones(isochrones, plt, 'g--')
    plt.xlim([-1,1])