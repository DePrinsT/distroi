"""
Subpackage containing auxiliary modules used to support the main parts of the code.
Some modules are originally written by A. Corporaal and J. Kluska.
"""

# Imports here will become available at the level of the subpackage.
# E.g. distroi.auxiliary.beam.oi_container_calc_gaussian_beam will be available as
# 'distroi.auxiliary.oi_container_calc_gaussian_beam'.

# IMPORTS FROM AUXILIARY SUBPACKAGE

# imports from beam module
from distroi.auxiliary.beam import Beam
from distroi.auxiliary.beam import oi_container_calc_gaussian_beam

# imports from time_base_oifits module
from distroi.auxiliary.time_base_oifits import oifits_time_window_plot
