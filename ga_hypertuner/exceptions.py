import warnings


class GaParamsException(Exception):
    PARAMETER_SHOULD_EXIST = " Should be given as ga parameters"
    PARAMETER_OUT_OF_RANGE = " Is out of range, the range for this is "
    PARAMETER_WRONG_TYPE = " : Wrong type, should be "
    """
    Exception raised for invalid GA parameters.

    :param message: Explanation of the error.
    :type message: str
    
    :param parameter: Name of the invalid parameter
    :type parameter: str
    
    :param range_: Range of valid values for the parameter
    :type range_: str
    """

    def __init__(self, message, parameter, range_: str = ""):
        self.message = message
        self.parameter = parameter
        self.range_ = range_

    def __str__(self):
        return self.parameter + self.message + self.range_


class MParamsException(Exception):
    KEYS_NOT_EQUAL = "Parameters with no static value should have a specified boundary"
    BOUNDARY_VALUE = " : Boundaries should have two values, a start and an end"
    BOUNDARY_INVALID = " : Boundaries values are invalid"
    PARAMETER_WRONG_FORMAT = " : Wrong format, format of parameter should be either [None,Type] or a static value"

    """
    Exception raised for invalid Model parameters.

    :param message: Explanation of the error.
    :type message: str

    :param parameter: Name of the invalid parameter
    :type parameter: str
    """

    def __init__(self, message, parameter: str = ""):
        self.message = message
        self.parameter = parameter

    def __str__(self):
        return self.parameter + self.message


class GaHypertunerParamException(Exception):
    PARAMETER_WRONG_TYPE = " : Wrong type, should be "
    VERBOSITY_WARNING = "Invalid verbosity level provided. Using default value of 1."

    """
    Exception raised for invalid tuner parameters.

    :param message: Explanation of the error.
    :type message: str

    :param parameter: Name of the invalid parameter
    :type parameter: str

    :param type_: Type of valid values for the parameter
    :type type_: str
    """

    def __init__(self, message, parameter="", type_=""):
        self.message = message
        self.parameter = parameter
        self.type_ = type_

    def __str__(self):
        return self.parameter + self.message + self.type_

    @staticmethod
    def warning(message):
        warnings.warn(message)

