"""A module defining a universal abstract interface for interacting with RT codes.

Defines an abstract interface class for interacting with an RT code. Includes methods for setting and retrieving
necessary input variables (physical fitting parameters, resolution settings, etc.), the production of matching input
files, executing different runs (thermal structure run, SED run, image run, etc.), and retrieving the output files of
the RT code and placing them in appropriate objects (e.g. a list of Image objects for multiple images) for
comparison with observational data.

Warnings
--------
All concrete implementations of RT model interface classes must inherit from the abstract class defined here. This
is so the rest of the code can remain agnostic as to which RT code is used in the model.
"""

# TODO: provide an abstract interface defining the general functionalities of an RT model and the methods it must
# implement

from abc import ABC, abstractmethod
from distroi.data.sed import SED
from distroi.data.image import Image

# TODO: 
class RTComp(ABC):
    """Abstract class representing a radiative transfer model component.
    """

    @abstractmethod
    def calc_sed(self, wrange: tuple[float, float]) -> SED:
        """Calculate the SED within a specified wavelength interval.

        Parameters
        ----------
        wrange : tuple of float
            The wavelength range (min, max) in microns.

        Returns
        -------
        SED
            The calculated spectral energy distribution.
        """
        pass

    @abstractmethod
    def calc_image(self, wavelength: float) -> Image:
        """Calculate an image at the specified wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength at which to calculate the image in microns.

        Returns
        -------
        Image
            The calculated image.
        """
        pass

    @abstractmethod
    def get_params(self):
        """Retrieve a dictionary of parameters for this RT component.
        """
        pass
