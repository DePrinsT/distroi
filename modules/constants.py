import os
import numpy as np

# constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of the package
SPEED_OF_LIGHT = 299792458  # speed of light in SI units

# unit conversions
DEG2RAD = np.pi / 180  # degree to radian
RAD2DEG = 1 / DEG2RAD  # radian to degree
MAS2RAD = 1e-3 / 3600 * DEG2RAD  # milli-arcsecond to radian
RAD2MAS = 1 / MAS2RAD  # radian to milli-arcsecond
MICRON2M = 1e-6  # micrometer to meter
M2MICRON = 1 / MICRON2M  # meter to micron
WATT_PER_METER2_HZ_2JY = 1e26  # conversion spectral flux density from SI W m^-2 Hz^-1 to Jansky
