"""A module defining the functionalities of a full model.

Defines the implementation of a model, from which observables for matching to observational data can be calculated.
"""

from distroi.model.rt_comp import rt_comp
from distroi.model.geom_comp import geom_comp


class Model:
    """Defines a model consisting of a single radiative transfer component and possible additional geometric components.

    Attributes
    ----------
    components : dict of str to RTComp or GeomComp
        Dictionary containing the names of model components within the Model scope and the corresponding component
        objects.
    params : dict of str to Param
        Dictionary containing the names of model parameters within the Model scope and the corresponding parameter
        objects.
    """

    def __init__(self):
        pass

    def add_components(
        self,
        comp: rt_comp.RTComp | geom_comp.GeomComp | list[rt_comp.RTComp | geom_comp.GeomComp],
        names: list[str] = None,
    ) -> None:
        """
        Add a component or list of components to the model. The parameters of the component will be named accordingly in
        the `components` dict. If no names are specified, sensible defaults will be chosen.

        Parameters
        ----------
        comp : RTComp or GeomComp or list of RTComp or GeomComp
            The component or list of components to be added to the model.
        names : list of str, optional
            The names to be assigned to the components. If not specified, sensible defaults will be chosen.
        """
        pass
