import matplotlib.pyplot as plt
import numpy as np

from config import equation_config
from mrt_phase_numeric.src.update_equ.update import update_
from mrt_phase_numeric.src.util.get_ISI import get_ISI
from mrt_phase_numeric.src.util.save_util import read_curve_from_file

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']

D = 0.0


def plot_basic(dt=0.01):
    limit_cycle = read_curve_from_file(f'../../data/input/limit_cycle/limit_cycle.txt')

    v_ = [0]
    a_ = [0]
    y_ = [0]

    n_max = 10
    i = 0

    while i <= n_max:
        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt, D=D)
        v_.append(v_new)
        a_.append(a_new)
        y_.append(y_new)

        if y_new == 1:
            i += 1

    t = [i*dt for i in range(len(v_))]

    fig, axs = plt.subplots(2, 1)

    # axs[0].title(f'D {D}')
    title = f'$D={D}$, $v_{{thr}}={v_th}$, $\mu={mu}$, $\\tau_a={tau_a}$, $\Delta_a={delta_a}$'

    axs[0].plot(t, v_, 'r')
    axs[0].set_xlabel('t')
    axs[0].set_ylabel('v(t)')
    axs[0].set_title(title)

    axs[1].plot(t, a_, 'r')
    axs[1].set_xlabel('t')
    axs[1].set_ylabel('a(t)')

    # axs[2].plot(v_, a_, '-')
    # axs[2].set_xlabel('v')
    # axs[2].set_ylabel('a')

    plt.tight_layout()

    plt.savefig(f'img\\v_a_example_D_{D}_n_{n_max}.png')

    plt.figure()
    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.')

    start = 0
    for i in np.where(np.array(y_) == 1)[0]:
        plt.plot(v_[start+1:i], a_[start+1:i], 'r-')
        start = i
    plt.plot(v_[start + 1:], a_[start + 1:], 'r-')

    plt.legend(['limit cycle', 'trajectory'])

    plt.xlabel('v')
    plt.ylabel('a')
    plt.title(title)
    # plt.xlim([-0.3, 1])
    plt.ylim([-0.05, 2.5])

    plt.savefig(f'img\\v_a_scatter_D_{D}_n_{n_max}.png')

    plt.show()


def get_ISI_():
    dt = 0.01

    v_ = [0]
    a_ = [0]
    y_ = [0]

    n_spikes = 10000
    i = 0

    while i <= n_spikes:
        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt, D=D)
        v_.append(v_new)
        a_.append(a_new)
        y_.append(y_new)

        print(i)

        if y_new == 1:
            i += 1

    ISI = get_ISI(y_, dt)

    mean_ISI = np.mean(ISI[1000:])

    mean_cycle_time = len(y_) * dt / n_spikes

    return ISI, mean_ISI, mean_cycle_time


if __name__ == '__main__':
    # plot_basic()

    ISI, mean_ISI, mean_cycle_time = get_ISI_()

    print('mean ISI: ', mean_ISI)
    print('mean_cycle_time: ', mean_cycle_time)

