import matplotlib.pyplot as plt

from mrt_phase_pde.pde.own_method.src.T_0.T_N import T_N as T_N_class
from mrt_phase_numeric.src.DataTypes.DataTypes import filepaths
from mrt_phase_numeric.src.util.save_util import read_curve_from_file
from scipy.signal import savgol_filter
from mrt_phase_pde.pde.own_method.extract_curves_from_T_N.extract_curves import get_curve_for_T_N

limit_cycle_file_path = filepaths['limit_cycle_path']
limit_cycle = read_curve_from_file(limit_cycle_file_path)

D_list = [0.0, 0.25]

if __name__ == '__main__':

    plt.figure()
    draw_style = ['b-', 'r-', 'm-'] #, 'm:', 'c.' ]

    for j, D in enumerate(D_list):
        T_N = T_N_class.load(f'..\\result\\T_N_6_D_{D}.pickle')
        T_N_curves = get_curve_for_T_N(T_N, D)

        for i, curve in enumerate(T_N_curves):
            curve[:, 1] = savgol_filter(curve[:, 1], len(curve), 4)

            if i == 0:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j], label=f'D={D}')
            else:
                plt.plot(curve[:, 0], curve[:, 1], draw_style[j])

    plt.plot(limit_cycle[:, 0], limit_cycle[:, 1], 'g.', label='limit cycle')

    plt.xlabel('v')
    plt.ylabel('a')
    plt.ylim([0, 4])
    plt.title(f'Isochrones')
    plt.legend()

    plt.show()
