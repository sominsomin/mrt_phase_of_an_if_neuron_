import random
import numpy as np

from mrt_phase_numeric.src.config import equation_config

mu = equation_config['mu']
v_th = equation_config['v_th']
tau_a = equation_config['tau_a']
delta_a = equation_config['delta_a']


# TODO when to reset v, directly when spike hits or afterwards
def update_(v, a, y=0, offset=0, dt=None, D=None):
    noise = random.gauss(0, 1)

    # if spike then reset v_new
    if y == 1:
        v_new = 0.0
    else:
        v_new = v + (mu - v - a) * dt + np.sqrt(2 * D * dt) * noise

    a_new = a + 1 / tau_a * (-a + tau_a * delta_a * y * 1 / dt) * dt

    if v_new > v_th:
        y_new = 1

        # get a value exactly at threshold crossing
        a_new = np.interp([v, v_th], [v, v_new], [a, a_new])[1]
        v_new = 1.0

        offset += 1
    else:
        y_new = 0

    return v_new, a_new, y_new, offset


def update_backwards(v, a, offset=0, dt=None, D=None):
    noise = random.gauss(0, 1)

    v_new = v + (mu - v - a) * dt + np.sqrt(2 * D * dt) * noise
    a_new = a + 1 / tau_a * (-a) * dt

    return v_new, a_new, offset


def integrate_backwards(v_init, a_init, T, dt, D):
    v_ = [v_init]
    a_ = [a_init]

    n_max = int(T / dt)

    for i in range(n_max):
        v_new, a_new, offset = update_backwards(v_[-1], a_[-1], dt=-dt, D=D)
        v_.append(v_new)
        a_.append(a_new)

    return v_, a_


def integrate_forwards_time(v_init, a_init, T, D, dt):
    v_ = [v_init]
    a_ = [a_init]
    y_ = [0]

    n_max = int(T / dt)

    for i in range(n_max):
        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt, D=D)
        v_.append(v_new)
        a_.append(a_new)
        y_.append(y_new)

    return v_, a_, y_


def integrate_forwards(v_init, a_init, max_n_cycles, D, dt):
    v_ = [v_init]
    a_ = [a_init]
    # y_ = [0]

    i = 0

    if v_init > 1.0:
        y_ = [1]
        i += 1
    else:
        y_ = [0]

    while i < max_n_cycles:
        v_new, a_new, y_new, offset = update_(v_[-1], a_[-1], y=y_[-1], dt=dt, D=D)
        v_.append(v_new)
        a_.append(a_new)
        y_.append(y_new)

        if y_new == 1:
            i += 1
            # print(i)

    return v_, a_, y_
