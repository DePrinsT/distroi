"""
Module defining the implementation of a model, from which observables matching observational data can be calculated.
"""

from distroi.model.rt_comp import rt_comp
from distroi.model.geom_comp import geom_comp
from distroi.model import param


class Model:
    """
    Defines a model consisting of a single radiative transfer component and possible additional geometric components.

    :ivar dict[str, RTComp | GeomComp] components: Dictionary containing the names of model components within the Model 
        scope and the corresponding component objects.
    :ivar dict[str, Param] params: Dictionary containing the names of model parameters within the `Model` 
        scope and the corresponding parameter objects.
    """

    def __init__(self) -> None:
        """
        Constructor method. See class docstring for information on initialzization parameters and instance properties.
        """
        pass

    def add_components(
        self,
        comp: rt_comp.RTComp | geom_comp.GeomComp | list[rt_comp.RTComp | geom_comp.GeomComp],
        names: list[str] = None,
    ) -> None:
        """
        Add a component or list of components to the model. The parameters of the component will be named accordingly in
        the `components` dict. If no names are specified, sensible defaults will be chosen.
        """
        pass
