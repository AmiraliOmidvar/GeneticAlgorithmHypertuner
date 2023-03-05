import numpy as np
from ga_hypertuner.exceptions import GaParamsException, MParamsException, GaHypertunerParamException
from ga_hypertuner.ga import GA


class GaHypertuner:
    default_ga_parameters = {"pop_size": 20, "fscale": 0.5, "gmax": 50, "direction": "min",
                             "cp": 0.5}
    best_params = None

    def tune(self, ga_parameters: dict, model, model_parameters: dict
             , boundaries: dict
             , x_train, y_train
             , scoring
             , stop_value: int = None
             , stratified: bool = False
             , k: int = 5
             , verbosity: int = 1
             , show_progress_plot: bool = False):

        v_list = [verbosity]
        stop_criteria = False

        self._check_ga_params(ga_parameters)
        self._check_m_parameters(model_parameters, boundaries)
        self._check_ga_hypertuner_parameters(stop_value, v_list, stratified, show_progress_plot)

        verbosity = v_list[0]
        if stop_value is not None:
            stop_criteria = True

        ga = GA(ga_parameters, model, model_parameters
                , boundaries, x_train, y_train
                , scoring, stop_criteria=stop_criteria
                , stop_value=stop_value, stratified=stratified
                , k=k, verbosity=verbosity, show_progress_plot=show_progress_plot)

        return ga.main()

    @staticmethod
    def _check_ga_params(ga_parameters):
        ga_parameters_range = {"pop_size": [5, np.Inf], "fscale": [0, np.inf], "gmax": [1, np.Inf],
                               "direction": ["min", "max"], "cp": [0, 1]}
        for k in list(ga_parameters.keys()):
            r = ga_parameters_range[k]
            if k != "direction":
                if type(ga_parameters[k]) is not int and type(ga_parameters[k]) is not float:
                    raise GaParamsException(GaParamsException.PARAMETER_WRONG_TYPE, k, "number")
                if not r[0] < ga_parameters[k] < r[1]:
                    raise GaParamsException(GaParamsException.PARAMETER_OUT_OF_RANGE, k, str(r))
            else:
                if ga_parameters["direction"] != "max" and ga_parameters["direction"] != "min":
                    raise GaParamsException(GaParamsException.PARAMETER_OUT_OF_RANGE, k, str(r))

        if "pop_size" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.PARAMETER_SHOULD_EXIST, "pop_size")
        if "fscale" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.PARAMETER_SHOULD_EXIST, "fscale")
        if "gmax" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.PARAMETER_SHOULD_EXIST, "gmax")
        if "direction" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.PARAMETER_SHOULD_EXIST, "direction")
        if "cp" not in list(ga_parameters.keys()):
            raise GaParamsException(GaParamsException.PARAMETER_SHOULD_EXIST, "cp")

    @staticmethod
    def _check_m_parameters(model_parameters, boundaries):
        if set(model_parameters) != set(boundaries):
            raise MParamsException(MParamsException.KEYS_NOT_EQUAL)

        for k in list(boundaries.keys()):
            if len(boundaries[k]) != 2:
                raise MParamsException(MParamsException.BOUNDARY_VALUE, k)
            if boundaries[k][0] >= boundaries[k][1]:
                raise MParamsException(MParamsException.BOUNDARY_INVALID, k)

        for k in list(model_parameters.keys()):
            if len(model_parameters[k]) != 2:
                raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)
            if type(model_parameters[k]) != list:
                raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)
            if type(model_parameters[k][1]) != type:
                raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)
            if model_parameters[k][0] is not None and type(model_parameters[k][0]) != model_parameters[k][1]:
                raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)

    @staticmethod
    def _check_ga_hypertuner_parameters(stop_value, verbosity, stratified, show_progress_plot):
        if type(stop_value) != float and stop_value is not None and type(stop_value) != int:
            raise GaHypertunerParamException(GaHypertunerParamException.PARAMETER_WRONG_TYPE, "stop_value", "number")
        if type(verbosity[0]) != int:
            raise GaHypertunerParamException(GaHypertunerParamException.PARAMETER_WRONG_TYPE, "verbosity", "int")
        if type(stratified) != bool:
            raise GaHypertunerParamException(GaHypertunerParamException.PARAMETER_WRONG_TYPE, "stratified", "bool")
        if type(show_progress_plot) != bool:
            raise GaHypertunerParamException(GaHypertunerParamException.PARAMETER_WRONG_TYPE, "show_progress_plot",
                                             "bool")

        if verbosity[0] not in [0, 1, 2, 3]:
            verbosity[0] = 1
            GaHypertunerParamException.warning(GaHypertunerParamException.VERBOSITY_WARNING)
