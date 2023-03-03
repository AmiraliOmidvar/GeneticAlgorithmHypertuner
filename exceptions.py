import warnings


class GaParamsException(Exception):
    SHOULD_EXIST = " should be given as ga parameters"
    OUT_OF_RANGE = " is out of range, the range for this is "

    def __init__(self, message, parameter, range_: str = ""):
        self.message = message
        self.parameter = parameter
        self.range_ = range_

    def __str__(self):
        return self.parameter + self.message + self.range_


class MParamsException(Exception):
    KEYS_NOT_EQUAL = "parameter dictionary and boundary dictionary should have the same keys"
    BOUNDARY_VALUE = " : boundaries should have two values, a start and an end"
    M

    def __init__(self, message, parameter: str = ""):
        self.message = message
        self.parameter = parameter

    def __str__(self):
        return self.parameter + self.message
