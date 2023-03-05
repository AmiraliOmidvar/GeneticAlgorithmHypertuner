import warnings


class GaParamsException(Exception):
    PARAMETER_SHOULD_EXIST = " should be given as ga parameters"
    PARAMETER_OUT_OF_RANGE = " is out of range, the range for this is "
    PARAMETER_WRONG_TYPE = " : wrong type, should be "

    def __init__(self, message, parameter, range_: str = ""):
        self.message = message
        self.parameter = parameter
        self.range_ = range_

    def __str__(self):
        return self.parameter + self.message + self.range_


class MParamsException(Exception):
    KEYS_NOT_EQUAL = "parameter dictionary and boundary dictionary should have the same keys"
    BOUNDARY_VALUE = " : boundaries should have two values, a start and an end"
    BOUNDARY_INVALID = " : boundaries values are invalid"
    PARAMETER_WRONG_FORMAT = " : wrong format, format of parameter should be either [None,Type] or [Value,Type]"

    def __init__(self, message, parameter: str = ""):
        self.message = message
        self.parameter = parameter

    def __str__(self):
        return self.parameter + self.message


class GaHypertunerParamException(Exception):
    PARAMETER_WRONG_TYPE = " : wrong type, should be "

    def __init__(self, parameter, type_):
        self.parameter = parameter
        self.type_ = type_

    def __str__(self):
        return self.parameter + self.PARAMETER_WRONG_TYPE + self.type_
