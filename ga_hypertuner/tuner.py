import numpy as np
from ga_hypertuner.exceptions import GaParamsException, MParamsException, GaHypertunerParamException
from ga_hypertuner.ga import GA
from typing import Union


class Tuner:
    """
    Main class. user interacts with this class and given arguments are checked and passed to GA class.

    :ivar default_ga_parameters: a default dictionary for ga_parameters.
    """
    default_ga_parameters = {"pop_size": 20, "fscale": 0.5, "gmax": 20, "direction": "max",
                             "cp": 0.5}

    @staticmethod
    def tune(x_train, y_train, model
             , ga_parameters: dict, model_parameters: dict
             , boundaries: dict
             , scoring
             , stop_value: int = None
             , stratified: bool = False
             , k: int = 5
             , verbosity: int = 1
             , show_progress_plot: bool = False):

        """
        Main method to call to start tuning algorithm.

        :param ga_parameters: Parameters of the genetic algorithm. For more information on the parameters, see below.
        :type ga_parameters: dict

        :ga_parameters:
            * *direction* (``int``): Determines whether the score of models should be maximized or minimized. Accepted values are ["max","min"].
            * *pop_size* (``int``): Size of population in each generation. Increasing this value will reduce the chance of local optima. Accepted values are integers greater than 5. Default is 20.
            * *gmax* (``int``): Maximum number of generations. After this many generations, the algorithm will stop and return the best params. Accepted values are integers greater than 1. Default is 50.
            * *fscale* (``int``): A scaling factor that controls the amount of effect that differences between parameters of population members have. larger values will result in larger convergence rate.
            When convergence rate is higher, it will take less time for algorithm to reach local optimum, but the local optimum have lesser chance of being global. Reducing it will opposite result Accepted values are floats between 0 and 1. Default is 0.5.
            * *cp* (``int``): The probability that a child will inherit a parameter from a parent instead of a trial vector. Accepted values are floats between 0 and 1. Default is 0.5.

        :param model: Model class that its hyperparameters are being optimized. Any model class that scikit cross-validate module can accept.

        :param model_parameters: hyperparameters that are being optimized. This is a dictionary with parameters of the machine learning model as keys and a list either like [None, Parameter Type] (for optimization of parameter) or [Static Value, Parameter] (for passing the parameter as a static value that will not be changed).
        :type model_parameters: dict

        :param boundaries: Boundary search for hyperparameters. This is a dictionary with parameters of the machine learning model as keys and a list like [start,end].
        :type boundaries: dict

        :param x_train: Training features for the given model. This data will be used to train the model without slicing or sampling.
        :type x_train: Dataframe

        :param y_train: Training target for the given model. This data will be used to train the model without slicing or sampling.
        :type y_train: Dataframe

        :param scoring: The scoring criteria that the algorithm tries to optimize. Accepted values are scores that scikit cross validation accepts.
        :type scoring: str

        :param stop_value: The score that, when reached, the algorithm will stop. Default is None.
        :type stop_value:Union[int, float]

        :param k: Number of splits for k-fold cross validation. Accepted values are integers greater than 1. Default is 5.
        :type k: int

        :param stratified: Whether to use stratified cross validation or not. Default is False.
        :type stratified: bool

        :param verbosity: Determines the amount of information that is returned after each generation is generated. Accepted values are 0, 1, 2, or 3. Default is 1.
        :type verbosity: int

        :param show_progress_plot: Whether the progress plot of the score for each generation should be shown at the end of each generation.
        :type show_progress_plot: bool

        :return: a dictionary containing the best hyperparameters.
        """

        # making verbosity mutable, so it can be changed in scope of static methods
        v_list = [verbosity]
        stop_criteria = False

        # check parameters
        Tuner._check_ga_params(ga_parameters)
        Tuner._check_m_parameters(model_parameters, boundaries)
        Tuner._check_ga_hypertuner_parameters(stop_value, v_list, stratified, show_progress_plot)

        # set values for verbosity and
        verbosity = v_list[0]
        if stop_value is not None:
            stop_criteria = True

        # start algorithm
        ga = GA(ga_parameters, model, model_parameters
                , boundaries, x_train, y_train
                , scoring, stop_criteria=stop_criteria
                , stop_value=stop_value, stratified=stratified
                , k=k, verbosity=verbosity, show_progress_plot=show_progress_plot)

        return ga.main()

    @staticmethod
    def _check_ga_params(ga_parameters):

        """
        Checks ga_parameters.
        :param ga_parameters: Parameters of the genetic algorithm. For more information on the parameters.
        :type ga_parameters: dict
        :return: None
        """

        ga_parameters_range = {"pop_size": [5, np.Inf], "fscale": [0, np.inf], "gmax": [1, np.Inf],
                               "direction": ["min", "max"], "cp": [0, 1]}
        for k in list(ga_parameters.keys()):
            r = ga_parameters_range[k]
            if k != "direction":
                if type(ga_parameters[k]) is not int and type(ga_parameters[k]) is not float:
                    raise GaParamsException(GaParamsException.PARAMETER_WRONG_TYPE, k, "number")
                if not r[0] <= ga_parameters[k] <= r[1]:
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

        """
        Checks model parameters.
        :param model_parameters: hyperparameters that are being optimized. This is a dictionary with parameters of the machine learning model as keys and a list either like [None, Parameter Type] (for optimization of parameter) or [Static Value, Parameter] (for passing the parameter as a static value that will not be changed).
        :type model_parameters: dict

        :param boundaries: Boundary search for hyperparameters. This is a dictionary with parameters of the machine learning model as keys and a list like [start,end].
        :type boundaries: dict
        :return: None
        """

        for k in list(boundaries.keys()):
            if len(boundaries[k]) != 2:
                raise MParamsException(MParamsException.BOUNDARY_VALUE, k)
            if boundaries[k][0] >= boundaries[k][1]:
                raise MParamsException(MParamsException.BOUNDARY_INVALID, k)

        for k in list(model_parameters.keys()):
            if type(model_parameters[k]) == list:
                if type(model_parameters[k][1]) != type and model_parameters[k][0] is None:
                    raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)
                if model_parameters[k][0] is not None and type(model_parameters[k][0]) != model_parameters[k][1]:
                    raise MParamsException(MParamsException.PARAMETER_WRONG_FORMAT, k)
                if model_parameters[k][0] is None:
                    if k not in list(boundaries.keys()):
                        raise MParamsException(MParamsException.KEYS_NOT_EQUAL)

    @staticmethod
    def _check_ga_hypertuner_parameters(stop_value, verbosity, stratified, show_progress_plot):
        """
        Check tuner parameters.
        :param stop_value: The score that, when reached, the algorithm will stop. Default is None.
        :type stop_value:Union[int, float]

        :param verbosity: Determines the amount of information that is returned after each generation is generated. Accepted values are 0, 1, 2, or 3. Default is 1.
        :type verbosity: int

        :param stratified: Whether to use stratified cross validation or not. Default is False.
        :type stratified: bool

        :param show_progress_plot: Whether the progress plot of the score for each generation should be shown at the end of each generation.
        :type show_progress_plot: bool

        :return: None
        """
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
