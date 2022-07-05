import os
import numpy as np
from mrt_phase_numeric.src.Isochrone.Isochrone import IsochroneBaseClass


def get_best_isochrone(isochrone: IsochroneBaseClass):
    error_history = np.array(isochrone.error_history)
    if len(error_history) < 2:
        error_history = np.array(isochrone.percent_error_history)

    idx = np.where(error_history == np.nanmin(error_history))[0][0]
    idx = len(error_history) - 1

    curves = isochrone.curve_history[idx]

    return curves


def load_curves(data_location, phi):
    file = os.listdir(os.path.join(data_location, phi))[0]
    file_path = os.path.join(data_location, phi, file)
    isochrone = IsochroneBaseClass.load(file_path)

    curves = get_best_isochrone(isochrone)

    return curves


def load_isochrones(data_location):
    phi_list = os.listdir(data_location)

    all_curves = dict()
    for phi in phi_list:
        print(phi)
        all_curves[phi] = load_curves(data_location, phi)

    return all_curves


def read_dat(filename):
    with open(filename, 'r') as file:
        data = []
        for line in file.readlines():
            line = line.split(' ')
            line[1] = line[1].replace('\n', '')
            line[0] = float(line[0])
            line[1] = float(line[1])

            data.append(line)

    return np.array(data)


def plot_isochrones(isochrones_list, plt, draw=None):
    for key in isochrones_list.keys():
        curves = isochrones_list[key]
        for i, curve in enumerate(curves):
            if curve.points.any():
                if draw:
                    # plt.plot(curve[:, 0], curve[:, 1], draw, label='numeric isochonres')
                    plt.plot(curve[:, 0], curve[:, 1], draw, label='numeric isochrones')
                else:
                    plt.plot(curve[:, 0], curve[:, 1])