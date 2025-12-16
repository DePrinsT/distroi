"""A module to handle constants and other project-wide settings.

Contains constants, unit conversions, miscellaneous universal functions and plotting settings to be uniformly used
throughout the project.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

# settings
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
Root filepath of the project.
"""

FIG_OUTPUT_TYPE: str = "png"
"""
Output type of the figures (e.g. `'pdf'` or `'png'`).
"""

FIG_DPI: int = 200
"""
DPI of the output figures.
"""

IMG_CMAP = "hot"
"""
Matplotlib colour map used for general images.
"""

IMG_CMAP_DIVERGING = "bwr"
"""
Matplotlib colour map used for images needing a diverging colourmap.
"""

PLOT_CMAP = "rainbow"
"""
colour map used for other line/scatter plots.
"""

# constants
SPEED_OF_LIGHT: float = 299792458.0
"""
Speed of light in SI units (m s^-1).
"""

K_BOLTZMANN: float = 1.380649e-23
"""
Boltzmann's constant in SI unis (J K^-1).
"""

H_PLANCK: float = 6.62607015e-34
"""
Planck constant in SI units (J Hz^-1).
"""

B_WIEN: float = 2.897771955e-3
"""
Wien's displacement constant in SI units (m K).
"""

SIG_STEFAN_BOLTZMANN: float = 5.670374419e-8
"""
Stefan-Boltzmann constant in SI units (W m^-2 K^-4).
"""

# unit conversions
DEG2RAD: float = np.pi / 180
"""
Degree to radian conversion factor.
"""

RAD2DEG: float = 1 / DEG2RAD
"""
Radian to degree conversion factor.
"""

MAS2RAD: float = 1e-3 / 3600 * DEG2RAD
"""
Milli-arcsecond to radian conversion factor.
"""

RAD2MAS: float = 1 / MAS2RAD
"""
Radian to milli-arcsecond conversion factor.
"""

MICRON2M: float = 1e-6
"""
Micrometer to meter conversion factor.
"""

M2MICRON: float = 1 / MICRON2M
"""
Meter to micron conversion factor.
"""

AU2METER = 1.496e11
"""
Astronomical unit to meter conversion factor.
"""

METER2AU = 1 / AU2METER
"""
Meter to astronomical unit conversion factor.
"""

HZ2GHZ: float = 1e-9
"""
Hertz to gigaHertz conversion factor.
"""

GHZ2HZ: float = 1 / HZ2GHZ
"""
gigaHertz to Hertz conversion factor.
"""

MICRON2AA: float = 1e4
"""
Micron to Angstrom conversion factor.
"""

AA2MICRON: float = 1 / MICRON2AA
"""
Angstrom to Micron conversion factor.
"""

WATT_PER_M2_HZ_2JY: float = 1e26
"""
Spectral flux density (F_nu) from SI (W m^-2 Hz^-1) to Jansky conversion factor.
"""

JY_2WATT_PER_M2_HZ: float = 1 / WATT_PER_M2_HZ_2JY
"""
Spectral flux density (F_nu) from Jansky to SI (W m^-2 Hz^-1) conversion factor.
"""

ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M: float = 1e3
"""
Spectral flux density (F_lam) from erg s^-1 cm^-2 micron^-1 to SI (W m^-2 m^-1)
"""

WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON: float = 1 / ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M
"""
Spectral flux density (F_lam) from SI (W m^-2 m^-1) to erg s^-1 cm^-2 micron^-1
"""

LSOL2WATT: float = 3.828e26
"""
Flux solar luminosity to Watt conversion factor.
"""

WATT2LSOL: float = 1 / LSOL2WATT
"""
Flux Watt to solar luminosity conversion factor.
"""


# plotting settings
def set_matplotlib_params() -> None:
    """
    Function to set project-wide matplotlib parameters. To be used at the top of a distroi module if plotting
    functionalities are included in it.

    Returns
    -------
    None
    """
    # setting some matplotlib parameters
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["legend.edgecolor"] = "grey"
    plt.rcParams["legend.framealpha"] = 0.5
    plt.rcParams["lines.markersize"] = 6.0
    plt.rcParams["lines.linewidth"] = 2.0
    # plt.rcParams["image.interpolation"] = 'bicubic'  # set interpolation method for imshow

    plt.rc("font", size=16)  # controls default text sizes
    plt.rc("axes", titlesize=14)  # fontsize of the axes title
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)  # fontsize of the tick labels
    plt.rc("legend", fontsize=14)  # legend fontsize
    plt.rc("figure", titlesize=14)  # fontsize of the figure title
    return


# utility functions
def redden_flux(
    wavelength: np.ndarray | float,
    flux: np.ndarray | float,
    ebminv: float,
    reddening_law: str = PROJECT_ROOT + "/utils/ISM_reddening/ISMreddening_law_Cardelli1989.dat",
) -> np.ndarray:
    """Apply a reddening law to flux.

    Takes wavelength(s) and the associated flux values, and reddens them according to the specified E(B-V) law.
    Note that this function will not extrapolate outside the wavelength ranges of the reddening law. Instead, no
    reddening will be applied outside this range.

    Parameters
    ----------
    wavelength : float or np.ndarray
        Wavelength(s) in micron.
    flux : float or np.ndarray
        The flux(es) to be reddened. Can be in either F_nu/F_lam or nu*F_nu/lam*F_lam format and in any units.
    ebminv : float
        E(B-V) reddening factor to be applied.
    reddening_law : str, optional
        Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
        'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

    Returns
    -------
    np.ndarray
        The reddened flux value(s).
    """
    if ebminv == 0.0:
        return flux
    else:
        # read in the ISM reddening law wavelengths in Angström and A/E in magnitude
        df_law = pd.read_csv(reddening_law, header=2, names=["WAVE", "A/E"], sep=r"\s+", engine="python")
        # set wavelength to micrometer
        wave_law = np.array(df_law["WAVE"]) * AA2MICRON
        ae_law = np.array(df_law["A/E"])

        # linearly interpolate A/E(B-V) values to used wavelengths
        f = interp1d(wave_law, ae_law, kind="linear", fill_value=0, bounds_error=False)
        ae = f(wavelength)
        flux_reddened = np.array(flux * 10 ** (-ae * ebminv / 2.5))

        return flux_reddened


def bb_flam_at_wavelength(wavelength: np.ndarray | float, temp: float) -> np.ndarray | float:
    """Calculate spectral radiance of a blackbody curve in B_lam format and SI units for a given wavelength.

    Parameters
    ----------
    temp : float
        Temperature of the blackbody in Kelvin.
    wavelength : float or np.ndarray
        Wavelength in micron.

    Returns
    -------
    float or np.ndarray
        B_lam spectral radiance of the blackbody in SI units (W m^-2 m^-1 sterradian^-1).
    """
    wave = wavelength * MICRON2M  # wavelength in SI
    radiance = (2 * H_PLANCK * SPEED_OF_LIGHT**2 / wave**5) / (
        np.exp(H_PLANCK * SPEED_OF_LIGHT / (wave * K_BOLTZMANN * temp)) - 1
    )
    return radiance


def bb_flam_at_frequency(frequency: np.ndarray | float, temp: float) -> np.ndarray | float:
    """Calculate spectral radiance of a blackbody curve in B_lam format and SI units for a given frequency.

    Parameters
    ----------
    temp : float
        Temperature of the blackbody in Kelvin.
    frequency : float or np.ndarray
        Frequency in Hertz.

    Returns
    -------
    float or np.ndarray
        B_lam spectral radiance of the blackbody in SI units (W m^-2 m^-1 sterradian^-1).
    """
    wave = (SPEED_OF_LIGHT / frequency) * M2MICRON  # wavelength in micron
    radiance = bb_flam_at_wavelength(temp=temp, wavelength=wave)
    return radiance


def bb_fnu_at_frequency(frequency: np.ndarray | float, temp: float) -> np.ndarray | float:
    """Calculate spectral radiance of blackbody curve in B_nu format and SI units for a given frequency.

    Parameters
    ----------
    temp : float
        Temperature of the blackbody in Kelvin.
    frequency : float or np.ndarray
        Frequency in Hertz.

    Returns
    -------
    float or np.ndarray
        B_nu spectral radiance of the blackbody in SI units (W m^-2 Hz^-1 sterradian^-1).
    """
    radiance = (2 * H_PLANCK * frequency**3 / SPEED_OF_LIGHT**2) / (
        np.exp(H_PLANCK * frequency / (K_BOLTZMANN * temp)) - 1
    )
    return radiance


def bb_fnu_at_wavelength(wavelength: np.ndarray | float, temp: float) -> np.ndarray | float:
    """Calculate spectral radiance of blackbody curve in B_nu format and SI units for given wavelength.

    Parameters
    ----------
    temp : float
        Temperature of the blackbody in Kelvin.
    wavelength : float or np.ndarray
        Wavelength in micron.

    Returns
    -------
    float or np.ndarray
        B_nu spectral radiance of the blackbody in SI units (W m^-2 Hz^-1 sterradian^-1).
    """
    freq = SPEED_OF_LIGHT / (wavelength * MICRON2M)  # frequency in Hertz
    radiance = bb_fnu_at_frequency(temp=temp, frequency=freq)
    return radiance


def flam_cgs_per_mum_to_fnu_jansky(flam: np.ndarray | float, wavelength: np.ndarray | float) -> np.ndarray | float:
    """Convert spectral flux densities from F_lam in erg s^-1 cm^-2 micron^-1 to F_nu in Jansky.

    Parameters
    ----------
    flam : float or np.ndarray
        Spectral flux density in F_lam format and units of erg s^-1 cm^-2 micron^-1.
    wavelength : float or np.ndarray
        Associated wavelengths in micron.

    Returns
    -------
    float or np.ndarray
        Spectral flux density in F_nu format and Jy units.
    """

    wavelength_si = wavelength * MICRON2M  # wavelength in SI
    flam_si = flam * ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # flam in SI
    fnu = flam_si * wavelength_si**2 / SPEED_OF_LIGHT  # fnu in SI units
    fnu = fnu * WATT_PER_M2_HZ_2JY  # set fnu in Jy

    return fnu


def gaussian_2d_elliptical_ravel(
    points: tuple[np.ndarray, np.ndarray],
    amp: float = 1,
    x0: float = 0,
    y0: float = 0,
    sig_min: float = 1,
    sig_maj_min_sig_min: float = 0,
    pa: float = 0,
    offset: float = 0,
) -> np.ndarray:
    """Calculate value of elliptical 2D Gaussian as a ravelled array.

    Function for calculating the value of a 2D Elliptical Gaussian at a given xy-point. Defined by an amplitude,
    xy center, standard deviations along major/minor axis, a major axis position angle and an offset. Returns
    a raveled array.

    Parameters
    ----------
    points : tuple of np.ndarray
        2D tuples describing the (x, y) points to be inserted. Note that positive x is defined as leftward and
        positive y as upward (i.e. the East and North respectively in the OI convention).
    amp : float, optional
        Amplitude of the Gaussian.
    x0 : float, optional
        x-coordinate center of the Gaussian.
    y0 : float, optional
        y-coordinate center of the Gaussian.
    sig_min : float, optional
        Standard deviation in the minor axis direction.
    sig_maj_min_sig_min : float, optional
        How much the standard deviation in major ellipse axis direction is greater than that of minor axis direction.
        Defined so it can always be greater than or equal to sig_min when used in scipy.optimize.curve_fit.
    pa : float, optional
        Position angle of the Gaussian (i.e. the major axis direction) anti-clockwise, starting North (positive y).
    offset : float, optional
        Base level offset from 0.

    Returns
    -------
    np.ndarray
        A raveled 1D array containing the values of the Gaussian calculated at the points.
    """
    x, y = points  # unpack tuple point coordinates in OI definition
    theta = pa * DEG2RAD

    sig_maj = sig_min + sig_maj_min_sig_min  # calculate std in y direction

    # set multiplication factors for representing the rotation matrix
    # note the matrix assumes positive x is to the right, so we also add a minus to the
    a = (np.cos(theta) ** 2) / (2 * sig_min**2) + (np.sin(theta) ** 2) / (2 * sig_maj**2)
    b = -(np.sin(2 * theta)) / (4 * sig_min**2) + (np.sin(2 * theta)) / (4 * sig_maj**2)
    c = (np.sin(theta) ** 2) / (2 * sig_min**2) + (np.cos(theta) ** 2) / (2 * sig_maj**2)
    values = offset + amp * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    values = np.array(values).ravel()  # ravel to a 1D array, so it can be used in scipy curve fitting

    return values


def nan_filter_arrays(ref_array, *args, filter_ref=False):
    """
    Filter arrays based on NaN values of reference array.

    Parameters
    ----------
    ref_array : np.ndarray
        Reference array based on whose values NaN filtering will occur.
    *args : tuple(np.ndarray)
        Extra arrays which will be filtered. Must have same shape as ref_array.
    filter_ref : Bool
        Whether to filter `ref_array` itself. `ref_array` is not filtered in place
        but the filtered array is included in the returned tuple.

    Returns
    -------
    filtered_arrays : tuple(np.ndarray)
        Tuple of the filtered arrays in order, including the filtered reference array
        if `filter_ref` flag is used.
    """
    nan_mask = np.isnan(ref_array)
    non_nan_mask = ~nan_mask  # invert NaN mask

    filtered_arrays = []  # convert to tuple later

    if filter_ref:
        filtered_arrays.append(ref_array[non_nan_mask])  # add filtered reference array

    for target_array in args:
        filtered_arrays.append(target_array[non_nan_mask])  # filter other arrays based on mask

    filtered_arrays = tuple(filtered_arrays)  # cast to tuple

    return filtered_arrays
