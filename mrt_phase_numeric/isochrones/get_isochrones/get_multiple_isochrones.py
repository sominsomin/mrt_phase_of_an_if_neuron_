import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

from mrt_phase_numeric.src.Isochrone.isochrone_util import get_isochrone
from mrt_phase_numeric.src.Isochrone.InitHelper import IsochroneInitHelper
from mrt_phase_numeric.src.DataTypes.DataTypes import DataTypes

DEBUG_MODE = True
# DEBUG_MODE = False

# TYPE = DataTypes.TIMESERIES_TYPE
TYPE = DataTypes.TIMESERIES_GRID_TYPE

if TYPE == DataTypes.TIMESERIES_TYPE:
    from mrt_phase_numeric.isochrones.get_isochrones.timeseries_config import \
        isochrone_config, init_config
elif TYPE == DataTypes.TIMESERIES_GRID_TYPE:
    from mrt_phase_numeric.isochrones.get_isochrones.timeseries_grid_config import \
        isochrone_config, init_config

############## simulation parameters ##############

converge_tol = 0.001
max_n_iterations = 200

########## init values

isochrone = None
isochrone_init_helper = None
error = 1

# phi_inits = [0.2, 0.3, 0.4, 0.5]
phi_inits = [0.0]
    #0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# phi_inits = [0.2]
phi_init_index = 0


def reset_simulation(_phi_inits, _phi_init_index):
    print('reset simulation')
    _phi_init_index += 1
    phi_init = _phi_inits[_phi_init_index]
    print(f'{phi_init}')
    isochrone = isochrone_init_helper.init_isochrone(phi_init)

    return isochrone, _phi_init_index


def update_plot(frame_number):
    global isochrone
    global isochrone_init_helper
    global error
    global phi_init_index

    v_init = phi_inits[phi_init_index]

    if isochrone.n_updates >= max_n_iterations: #or isochrone.has_reached_early_stopping():
        print('max n iterations reached')
        print(f'{v_init}')

        isochrone.save()

        # error = 1
        isochrone, phi_init_index = reset_simulation(phi_inits, phi_init_index)
        if isochrone.error is None or np.isnan(isochrone.error):
            error = 1
        else:
            error = isochrone.error

    if error > converge_tol:
        isochrone.update_isochrone_single_iteration()

        # make sure its not overwritten, a bit hacky
        isochrone.all_rt_history = isochrone.all_return_times_list_history
        if isochrone_init_helper.save_curves:
            isochrone.save()

        if isochrone_init_helper.multiple_branches:
            curve_new = isochrone.curves[1]
            t_list = isochrone.mean_return_time_pro_point[1]
            error = isochrone.mean_deviation
        else:
            curve_new = isochrone.curve
            t_list = isochrone.mean_return_time_pro_point
            error = isochrone.error

        print(isochrone.n_updates)
        print(error)
        print(t_list)

        v = curve_new[:, 0]
        a = curve_new[:, 1]

        line[0].set_data(v, a)
        line[1].set_data(v, a)
        line[2].set_data(v, t_list)
        line[3].set_data(v, t_list)

        v_ = np.arange(-5, 1, 0.01)
        line[13].set_data(v_, [isochrone.target_rt for i in range(len(v_))])

        if isochrone_init_helper.multiple_branches:
            try:
                line[5].set_data(isochrone.curves[0][:, 0], isochrone.curves[0][:, 1])
                line[6].set_data(isochrone.curves[0][:, 0], isochrone.curves[0][:, 1])

                line[9].set_data(isochrone.curves[0][:, 0], isochrone.mean_return_time_pro_point[0])
                line[10].set_data(isochrone.curves[0][:, 0], isochrone.mean_return_time_pro_point[0])
            except:
                line[5].set_data([], [])
                line[6].set_data([], [])

                line[9].set_data([], [])
                line[10].set_data([], [])

            try:
                line[7].set_data(isochrone.curves[2][:, 0], isochrone.curves[2][:, 1])
                line[8].set_data(isochrone.curves[2][:, 0], isochrone.curves[2][:, 1])

                line[11].set_data(isochrone.curves[2][:, 0], isochrone.mean_return_time_pro_point[2])
                line[12].set_data(isochrone.curves[2][:, 0], isochrone.mean_return_time_pro_point[2])
            except:
                line[7].set_data([], [])
                line[8].set_data([], [])

                line[11].set_data([], [])
                line[12].set_data([], [])
    else:
        print('converged')

        isochrone.save()

        error = 1
        isochrone, phi_init_index = reset_simulation(phi_inits, phi_init_index)

    return line


def setup_plot(limit_cycle, isochrone_config: dict):
    fig, (ax1, ax2) = plt.subplots(2, 1)

    line3, = ax1.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.')

    line1, = ax1.plot([], [], 'r')
    line1_x, = ax1.plot([], [], 'rx')
    line2, = ax2.plot([], [], 'r')
    line2_x, = ax2.plot([], [], 'rx')
    line4, = ax1.plot([], [], 'y')
    line4_x, = ax1.plot([], [], 'yx')
    line5, = ax1.plot([], [], 'b')
    line5_x, = ax1.plot([], [], 'bx')
    line6, = ax2.plot([], [], 'y')
    line6_x, = ax2.plot([], [], 'yx')
    line7, = ax2.plot([], [], 'b')
    line7_x, = ax2.plot([], [], 'bx')

    v_ = np.arange(-5, 1, 0.01)

    target_rt = isochrone_config['target_rt']
    line8, = ax2.plot(v_, [target_rt for i in range(len(v_))], 'g--')

    global line
    line = [line1, line1_x, line2, line2_x, line3, line4, line4_x, line5, line5_x, line6, line6_x, line7, line7_x, line8]

    v_min = isochrone_config['v_min']
    a_max = 3
    a_min = 0

    ax1.set_xlim([v_min, 1])
    ax1.set_ylim([a_min, a_max])
    ax1.set_xlabel('v')
    ax1.set_ylabel('a')

    ax2.set_xlim([v_min, 1])
    ax2.set_ylim([0, 5])
    ax2.set_xlabel('v')
    ax2.set_ylabel('t')

    fig.tight_layout()

    return fig, ax1, ax2


if __name__ == '__main__':
    if DEBUG_MODE: isochrone_config['debug_mode'] = True
    isochrone_init_helper = IsochroneInitHelper(_isochrone_config=isochrone_config, _config=init_config)

    if not DEBUG_MODE:
        fig, ax1, ax2 = setup_plot(isochrone_init_helper.limit_cycle, isochrone_config)

        phi_init = phi_inits[phi_init_index]
        isochrone = isochrone_init_helper.init_isochrone(phi_init)

    if not DEBUG_MODE:
        animate = animation.FuncAnimation(fig, update_plot, interval=100, blit=True,)
                                          # save_count=30)

        # animate.save(f'v_{v_init}_deterministic_version_3.mp4')

    if DEBUG_MODE:
        for phi_init in phi_inits:
            print(phi_init)
            isochrone = isochrone_init_helper.init_isochrone(phi_init)
            if DEBUG_MODE: isochrone.debug_mode = True
            get_isochrone(isochrone, phi=phi_init, _save=isochrone_init_helper.save_curves, _tol=converge_tol, max_n_iterations=max_n_iterations)
