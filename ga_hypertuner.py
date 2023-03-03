import numpy as np


class GaHypertuner:
    default_ga_parameters = {"pop_size": 20, "fscale": 0.5, "gmax": 50, "stop_value": 0.99, "direction": "max",
                             "cp": 0.5}

    def tune(self, ga_parameters: dict, model_func
             , model_parameters: dict
             , boundaries: list
             , x_train, y_train
             , scoring
             , stop_criteria: bool = False, stop_value: int = None
             , k: int = 5, stratified: bool = False
             , verbosity: int = 1
             , show_progress_plot: bool = False):
        self.check_ga_params(ga_parameters)

    @staticmethod
    def check_ga_params(ga_parameters):
        ga_parameters_range = {"pop_size": [5, np.Inf], "fscale": [0, 1], "gmax": [1, np.Inf],
                               "direction": ["min", "max"], "cp": [0, 1]}
        for k in list(ga_parameters.keys()):
            r = ga_parameters_range[k]
            if k != "direction":
                if not r[0] < ga_parameters < r[1]:
                    raise GaParamsException(GaParamsException.OUT_OF_RANGE, k, str(r))
            else:
                if "direction" != "max" or "direction" != "min":
                    raise GaParamsException(GaParamsException.OUT_OF_RANGE, k, str(r))

        if "pop_size" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.OUT_OF_RANGE, "pop_size")
        if "fscale" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.OUT_OF_RANGE, "fscale")
        if "gmax" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.OUT_OF_RANGE, "gmax")
        if "direction" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.OUT_OF_RANGE, "direction")
        if "cp" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.OUT_OF_RANGE, "cp")

    @staticmethod
    def check_m_parameters(model_parameters, boundaries):
        if set(model_parameters) != set(boundaries):
            raise MParamsException(MParamsException.KEYS_NOT_EQUAL)

        for k in list(boundaries.keys()):
            if len(boundaries[k]) != 2:
                raise MParamsException(MParamsException.BOUNDARY_VALUE, k)

        for k in list(model_parameters.keys()):
            if len(model[k]) != 2:
                raise MParamsException(MParamsException.BOUNDARY_VALUE, k)

