"""
Defines an abstract interface class for interacting with an RT code. Includes methods for setting and retrieving
necessary input variables (physical fitting parameters, resolution settings, etc.), the production of matching input
files, executing different runs (thermal strucutre run, SED run, image run, etc.), and retrieving the output files of
the RT code and placing them in appropriate objects (e.g. a list of ImageFFT objects for multiple images) for
comparison with observational data.

All concrete implementations of RT model interface classes must inherit from the abstract class defined here.
"""

# TODO: provide an abstract interface defining the general functionalities of an RT model and the methods it must
# implement
