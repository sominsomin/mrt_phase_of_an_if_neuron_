import numpy as np
import matplotlib.pyplot as plt

from mrt_phase_numeric.src.config import equation_config
from mrt_phase_numeric.src.update_equ.update import integrate_forwards

D = 0.0
mean_T = equation_config['mean_T_dict'][D]
dt = mean_T/100


mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']


if __name__ == '__main__':
    v_init = 0
    a_init = 0

    v_, a_, y_ = integrate_forwards(v_init=v_init, a_init=a_init, max_n_cycles=1000, D=D, dt=dt)

    v_ = np.array(v_)
    a_ = np.array(a_)

    v_last_zero = np.where(np.array(v_) == 0)[0][-1]

    limit_cycle = np.array(list(zip(v_[v_last_zero:], a_[v_last_zero:])))

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'x')
    plt.xlabel('v(t)')
    plt.ylabel('a(t)')
    # plt.xlim([0, 1])
    # plt.ylim([0, 2])
    plt.title(f'limit cycle for \n D={D}, mu={mu}, tau_a={tau_a}, Delta_a={delta_a}')

    # plt.savefig(f'limit_cycle_scatter.png')

    plt.show()

    with open('../../data/input/limit_cycle/limit_cycle_version_3.txt', 'w') as file:
        for line in limit_cycle:
            file.write(f'({line[0]},{line[1]})' + '\n')
