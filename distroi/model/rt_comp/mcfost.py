"""A module to define an interface with the MCFOST RT code.

Defines classes for interfacing with different models produced with the MCFOST RT code (Pinte et al. 2009:
https://ui.adsabs.harvard.edu/abs/2009A%26A...498..967P/abstract). Requires the user to have installed MCFOST on the
machine running DISTROI, and have it available in the command line interface (via e.g. '$ mcfost <parameter_file>').

Notes:
------
All RT model class implementations must inherit from the abstract class defined in the 'rt_model' module.
"""

from distroi.model.rt_comp.rt_comp import RTComp

# TODO: implement a class for a simple 1-zone symmetric 2D disk as a first example


class McfostDisk2D(RTComp):
    """
    A symmetric 2D MCFOST disk radiative transfer model.
    """

    def __init__(self, settings_dict: dict) -> None:
        pass

    def write_param(self):
        """
        Writes the MCFOST parameter input file.
        """
        pass


class McfostDiskZone:
    """
    An MCFOST disk zone
    """

    def __init__(self) -> None:
        pass


# class McfostStar:
#     """ """
