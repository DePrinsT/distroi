"""
Defines an abstract interface class for interacting with an RT code. Includes methods for setting and retrieving
necessary input variables (physical fitting parameters, resolution settings, etc.), the production of matching input
files, executing different runs (thermal strucutre run, SED run, image run, etc.), and retrieving the output files of
the RT code and placing them in appropriate objects (e.g. a list of Image objects for multiple images) for
comparison with observational data.

All concrete implementations of RT model interface classes must inherit from the abstract class defined here.
"""

# TODO: provide an abstract interface defining the general functionalities of an RT model and the methods it must
# implement

from abc import ABC, abstractmethod
from distroi.data.sed import SED
from distroi.data.image import Image
from distroi.model import param

# TODO: 
class RTComp(ABC):
    """
    Abstract class representing a radiative transfer model component and the
    methods it should provide.

    :ivar dict[str, Param] params: Dictionary containing the names of parameters within the `RTComp` 
        scope and the corresponding parameter objects.
    """

    @abstractmethod
    def calc_sed(self, wrange: tuple[float, float]) -> SED:
        """
        Calculate the SED within a specified wavelength interval in unit micron.
        """
        pass

    @abstractmethod
    def calc_image(self, wavelength: float) -> Image:
        """
        Calculate an image at the specified wavelength in micron.
        """
        pass

    @abstractmethod
    def get_params(self) -> dict[str, param.Param]:
        """
        Retrieve a dictionary of parameters for this RT component, linking the name of the component within the 
        `RTComp` scope to the corresponding `Param` objects.
        """
        pass
