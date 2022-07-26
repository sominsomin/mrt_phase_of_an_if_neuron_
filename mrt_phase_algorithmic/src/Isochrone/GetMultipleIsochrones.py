import numpy as np


class GetIsochroneHelper:
    """
    helper class to run Isochrone calculations
    """

    config = None
    isochrone_init_helper = None
    isochrone = None

    tol = 0.01
    max_n_iterations = None

    def __init__(self, isochrone_config: dict, init_config: dict):
        self.init_config = init_config
        self.isochrone_config = self.isochrone_config

        self.isochrone_init_helper(isochrone_config,init_config)

        self._unpack_config()

    def _unpack_config(self):
        _config = self.init_config
        for key in _config:
            self.__setattr__(key, _config[key])

    def init_isochrone(self):
        self.isochrone = self.isochrone_init_helper(self.isochrone_config, self.init_config)

    def get_isochrone(self):
        return self.isochrone

    def reset_simulation(self, _v_inits: float, _v_init_index: int):
        print('reset simulation')
        _v_init_index += 1
        v_init = _v_inits[_v_init_index]
        print(f'{v_init}')
        isochrone = self.isochrone_init_helper.init_isochrone(v_init)

        return isochrone, _v_init_index

    def get_isochrone(self, isochrone):
        # for running code without update_plot function
        i = 0
        percent_error = 1

        while percent_error > self.tol:
            if self.max_n_iterations:
                if isochrone.n_updates >= self.max_n_iterations or isochrone.has_reached_early_stopping():
                    print('max n iterations reached')
                    print(f'{isochrone.phi}')
                    if self.isochrone_init_helper.save_curves:
                        isochrone.save_best_curve(skip_first=0)

            isochrone.update_isochrone_single_iteration()
            t_list = isochrone.mean_return_time_pro_point
            percent_error = isochrone.error

            t_list = [np.round(t, 3) if t is not np.nan else np.nan for t in t_list]

            i += 1
            print(i)
            print(percent_error)
            print(t_list)

        if self.isochrone_init_helper.save_curves:
            isochrone.save_last_curve()

        return isochrone