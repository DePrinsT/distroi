"""
Contains constant values, unit conversions and settings to be used throughout the project.

:var str PROJECT_ROOT: Path to the root of the distroi project on the user's system.
:var float SPEED_OF_LIGHT: In SI units.
:var float DEG2RAD: Conversion of degree to radian.
:var float RAD2DEG: Conversion of radian to degree.
:var float MAS2RAD: Conversion of milli-arcsecond to radian.
:var float RAD2MAS: Conversion of radian to milli-arcsecond.
:var float MICRON2M: Conversion of meter to micrometer/micron.
:var float M2MICRON: Conversion of micrometer/micron to meter.
:var float WATT_PER_METER2_HZ_2JY: Flux desnity conversion of W m^-2 Hz^-1 to Jansky (Jy).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of the package
SPEED_OF_LIGHT = 299792458.  # speed of light in SI units

# unit conversions
DEG2RAD = np.pi / 180  # degree to radian
RAD2DEG = 1 / DEG2RAD  # radian to degree
MAS2RAD = 1e-3 / 3600 * DEG2RAD  # milli-arcsecond to radian
RAD2MAS = 1 / MAS2RAD  # radian to milli-arcsecond
MICRON2M = 1e-6  # micrometer to meter
M2MICRON = 1 / MICRON2M  # meter to micron
WATT_PER_METER2_HZ_2JY = 1e26  # conversion spectral flux density from SI W m^-2 Hz^-1 to Jansky


def set_matplotlib_params():
    """
    Function to set project-wide matplotlib parameters. To be used at the top of a module if plotting functionalities
    are included in it.

    :rtype: None
    """
    # setting some matplotlib parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['lines.linewidth'] = 1.0

    plt.rc('font', size=10)  # controls default text sizes
    plt.rc('axes', titlesize=11)  # fontsize of the axes title
    plt.rc('axes', labelsize=11)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=10)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=10)  # fontsize of the tick labels
    plt.rc('legend', fontsize=9)  # legend fontsize
    plt.rc('figure', titlesize=12)  # fontsize of the figure title
    return
