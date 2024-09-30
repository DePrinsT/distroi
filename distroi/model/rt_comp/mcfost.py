"""
Defines classes for interfacing with different models produced with the MCFOST RT code (Pinte et al. 2009:
https://ui.adsabs.harvard.edu/abs/2009A%26A...498..967P/abstract). Requires the user to have installed MCFOST on the
machine running DISTROI, and have it available in the command line interface (via e.g. '$ mcfost <parameter_file>').

All RT model class implementations here must inherit from the abstract class defined in the 'rt_model' module.
"""

# TODO: implement a class for a simple 1-zone symmetric 2D disk as a first example
