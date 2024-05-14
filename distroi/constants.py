"""
Contains constants, unit conversions, utility functions and plotting settings to be uniformly used throughout the
DISTROI project.

:var str PROJECT_ROOT: Path to the root of the distroi project on the user's system.
:var float SPEED_OF_LIGHT: In SI units.
:var float DEG2RAD: Conversion of degree to radian.
:var float RAD2DEG: Conversion of radian to degree.
:var float MAS2RAD: Conversion of milli-arcsecond to radian.
:var float RAD2MAS: Conversion of radian to milli-arcsecond.
:var float MICRON2M: Conversion of meter to micrometer/micron.
:var float MICRON2AA: Conversion of micrometer/micron to Angstrom.
:var float AA2MICRON: Conversion of Angstrom to micrometer/micron.
:var float M2MICRON: Conversion of micrometer/micron to meter.
:var float WATT_PER_METER2_HZ_2JY: Flux density conversion of W m^-2 Hz^-1 to Jansky (Jy).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

# constants
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # root of the package
SPEED_OF_LIGHT = 299792458.  # speed of light in SI units (m s^-1)
K_BOLTZMANN = 1.380649e-23  # Boltzmann's constant in SI unis (J K^-1)
H_PLANCK = 6.62607015e-34  # Planck constant in SI units (J Hz^-1)
B_WIEN = 2.897771955e-3  # Wien's displacement constant in SI units (m K)
SIG_STEFAN_BOLTZMANN = 5.670374419e-8  # Stefan-Boltzmann constant in SI units (W m^-2 K^-4)

# unit conversions
DEG2RAD = np.pi / 180  # degree to radian
RAD2DEG = 1 / DEG2RAD  # radian to degree
MAS2RAD = 1e-3 / 3600 * DEG2RAD  # milli-arcsecond to radian
RAD2MAS = 1 / MAS2RAD  # radian to milli-arcsecond
MICRON2M = 1e-6  # micrometer to meter
M2MICRON = 1 / MICRON2M  # meter to micron
HZ2GHZ = 1e-9  # Hertz to gigaHertz
GHZ2Hz = 1 / HZ2GHZ  # gigaHertz to Hertz
MICRON2AA = 1e4  # micron to angstrom
AA2MICRON = 1 / MICRON2AA  # angstrom to micron
WATT_PER_M2_HZ_2JY = 1e26  # conversion spectral flux density (F_nu) from SI (W m^-2 Hz^-1) to Jansky
JY_2WATT_PER_M2_HZ = 1 / WATT_PER_M2_HZ_2JY  # conversion spectral flux density (F_nu) from Jansky to SI
ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M = 1e3  # conversion spectral flux density (F_lam) from erg s^-1 cm^-2 micron^-1
# to SI (W m^-2 m^-1)
WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON = 1 / ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # conversion spectral flux density


# (F_lam) from SI (W m^-2 m^-1) to erg s^-1 cm^-2 micron^-1


# plotting settings
def set_matplotlib_params():
    """
    Function to set project-wide matplotlib parameters. To be used at the top of a distroi module if plotting
    functionalities are included in it.

    :rtype: None
    """
    # setting some matplotlib parameters
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'dejavuserif'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['lines.markersize'] = 4
    plt.rcParams['lines.linewidth'] = 1.0

    plt.rc('font', size=12)  # controls default text sizes
    plt.rc('axes', titlesize=14)  # fontsize of the axes title
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    plt.rc('figure', titlesize=14)  # fontsize of the figure title
    return


# function to redden flux
def redden_flux(wavelength, flux, ebminv, reddening_law=PROJECT_ROOT + '/utils/ISM_reddening/'
                                                                       'ISMreddening_law_Cardelli1989.dat'):
    """
    Takes wavelength(s) and the associated flux values, and reddens them according to the specified E(B-V) value.
    Note that this function will not extrapolate outside the wavelength ranges of the reddening law. Instead, no
    reddening will be applied outside this range.

    :param Union[float, np.ndarray] wavelength: Wavelength(s) in micron.
    :param Union[float, np.ndarray] flux: The flux(es) to be reddened. Can be in either F_nu/F_lam or nu*F_nu/lam*F_lam
        format and in any units.
    :param float ebminv: E(B-V) reddening factor to be applied.
    :param str reddening_law: Path to the reddening law to be used. Defaults to the ISM reddening law by
        Cardelli (1989) in DISTROI's 'utils/ISM_reddening folder'. See this file for the expected formatting
        of your own reddening laws.
    :return flux_reddened: The reddened flux value(s).
    :rtype: Union[float, np.ndarray]
    """
    if ebminv == 0.0:
        return flux
    else:
        # read in the ISM reddening law wavelengths in Angström and A/E in magnitude
        df_law = pd.read_csv(reddening_law, header=2, names=['WAVE', 'A/E'],
                             sep=r'\s+', engine='python')
        # set wavelength to micrometer
        wave_law = np.array(df_law['WAVE']) * AA2MICRON
        ae_law = np.array(df_law['A/E'])

        # linearly interpolate A/E(B-V) values to used wavelengths
        f = interp1d(wave_law, ae_law, kind='linear', fill_value=0, bounds_error=False)
        ae = f(wavelength)
        flux_reddened = np.array(flux * 10 ** (-ae * ebminv / 2.5))

        return flux_reddened


# blackbody radiance functions
def bb_flam_at_wavelength(temp, wavelength):
    """
    Given a temperature and wavelength, returns the spectral radiance of a blackbody curve in B_lam format and SI units.

    :param float temp: Temperature of the blackbody in Kelvin.
    :param Union[np.ndarray, float] wavelength: Wavelength in micron.
    :return radiance: B_lam spectral radiance of the blackbody in SI units (W m^-2 m^-1 sterradian^-1).
    :rtype: float
    """
    wave = wavelength * MICRON2M  # wavelength in SI
    radiance = (2 * H_PLANCK * SPEED_OF_LIGHT ** 2 / wave ** 5) / (np.exp(H_PLANCK * SPEED_OF_LIGHT /
                                                                          (wave * K_BOLTZMANN * temp)) - 1)
    return radiance


def bb_flam_at_frequency(temp, frequency):
    """
    Given a temperature and frequency, returns the spectral radiance of a blackbody curve in B_lam format and SI units.

    :param float temp: Temperature of the blackbody in Kelvin.
    :param Union[np.ndarray, float] frequency: Frequency in Hertz.
    :return radiance: B_lam spectral radiance of the blackbody in SI units (W m^-2 m^-1 sterradian^-1).
    :rtype: float
    """
    wave = (SPEED_OF_LIGHT / frequency) * M2MICRON  # wavelength in micron
    radiance = bb_flam_at_wavelength(temp=temp, wavelength=wave)
    return radiance


def bb_fnu_at_frequency(temp, frequency):
    """
    Given a temperature and wavelength, returns the spectral radiance of a blackbody curve in B_nu format and SI units.

    :param float temp: Temperature of the blackbody in Kelvin.
    :param Union[np.ndarray, float] frequency: Frequency in Hertz.
    :return radiance: B_nu spectral radiance of the blackbody in SI units (W m^-2 Hz^-1 sterradian^-1).
    :rtype: float
    """
    radiance = (2 * H_PLANCK * frequency ** 3 / SPEED_OF_LIGHT ** 2) / (np.exp(H_PLANCK * frequency /
                                                                               (K_BOLTZMANN * temp)) - 1)
    return radiance


def bb_fnu_at_wavelength(temp, wavelength):
    """
    Given a temperature and wavelength, returns the spectral radiance of a blackbody curve in B_nu format and SI units.

    :param float temp: Temperature of the blackbody in Kelvin.
    :param Union[np.ndarray, float] wavelength: Wavelength in micron.
    :return radiance: B_nu spectral radiance of the blackbody in SI units (W m^-2 Hz^-1 sterradian^-1).
    :rtype: float
    """
    freq = SPEED_OF_LIGHT / (wavelength * MICRON2M)  # frequency in Hertz
    radiance = bb_fnu_at_frequency(temp=temp, frequency=freq)
    return radiance
