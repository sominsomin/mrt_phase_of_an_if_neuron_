import numpy as np
from mrt_phase_numeric.src.Isochrone.Isochrone import IsochroneBaseClass


def get_isochrone(isochrone: IsochroneBaseClass, phi: float = None, _save: bool = False, _tol: float = 0.01, max_n_iterations: int = None):
    # for running code without update_plot function
    i = 0
    percent_error = 1

    while percent_error > _tol:
        if max_n_iterations:
            if isochrone.n_updates >= max_n_iterations or isochrone.has_reached_early_stopping():
                print('max n iterations reached')
                print(f'{phi}')

        isochrone.update_isochrone_single_iteration()

        t_list = isochrone.mean_return_time_pro_point
        percent_error = isochrone.error
        t_list = [np.round(t, 3) if t is not np.nan else np.nan for t in t_list]

        i += 1
        print(i)
        print(percent_error)
        print(t_list)

        if _save:
            # make sure its not overwritten, a bit hacky
            isochrone.all_rt_history = isochrone.all_return_times_list_history
            isochrone.save()

    return isochrone
