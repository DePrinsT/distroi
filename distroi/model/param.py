"""
Module defining the implementation of a model component parameter.
"""

import numpy as np


class Param:
    """
    Defines a model component parameter.

    :ivar float value: The value of the parameter.
    :ivar tuple[float, float] range: Range to which a parameter should be restricted during fitting routines. Can be
            +-np.inf to indicate the lack of a bound.
    :ivar bool tuneable: Whether or not the parameter is tuneable by a fitting routine.
    """

    def __init__(self, name=None, value=0, range=(-np.inf, np.inf), tuneable=False) -> None:
        """
        Constructor method. See class docstring for information on initialzization parameters and instance properties.
        """
        self.value = value
        self.range = range
        self.tuneable = tuneable
