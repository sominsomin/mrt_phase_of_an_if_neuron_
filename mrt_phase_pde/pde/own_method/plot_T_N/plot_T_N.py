import numpy as np
import matplotlib.pyplot as plt

from config import equation_config
from mrt_phase_pde.pde.own_method.src.exit_point_distribution.exit_point_distribution import ExitPointDistribution
from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.isochrones.plot_isochrones.plot_util import load_isochrones, plot_isochrones


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
Delta_a = equation_config['delta_a']

D = 0.0
N = 1


if __name__ == '__main__':
    # prob = ExitPointDistribution.load(f'data/epd_sim_D_{D}.pickle')
    # T_0 = T_0.load(f'data/T_0_D_{D}_sim.pickle')
    # T_N = T_N_class.load(f'result/T_N_15_D_{D}.pickle')
    T_N = T_N_class.load(f'..\\data\\T_{N}_D_{D}_sim_n_thr_{N + 1}.pickle')
    # T_N = T_N_class.load(f'result/T_N_0_D_{D}.pickle')

    # resize T_N for plot
    a = T_N.a
    v = T_N.v
    a_max_plot = 3

    idx_max_a = np.where(a == a_max_plot)[0][0] + 1
    a = a[:idx_max_a]
    T_N = T_N[:, :idx_max_a]

    __x, __y = np.meshgrid(v, a)

    plt.figure()
    plt.contourf(__x, __y, T_N.transpose(), levels=30)  # , v_min=-30, v_max=30)
    plt.colorbar()
    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, a_max_plot])
    plt.title(
        f'$T_{{{N}}}(v,a)$, $D={D}$')
        # f'\n $v_{{thr}}={v_th}$, $\\mu={mu}$, $\\tau_a={tau_a}$, $\\Delta_a={Delta_a}$')

    # plt.savefig(f'img\\comparision_to_numerics_T_N_D_{D}.png')

    det_file_paths = '../../../../mrt_phase_numeric/data/results/isochrones/from_timeseries_grid/deterministic/D_0.0'
    stochastic = f'..\\..\\..\\mrt_phase_numeric\\data\\results\\isochrones\\from_timeseries\\stochastic\\D_{D}'

    isochrones = load_isochrones(det_file_paths)
    # isochrones = load_isochrones(stochastic)
    # plot_isochrones(isochrones, plt, 'g--')
    plt.xlim([-1,1])