"""
Defines the classes and methods needed for geometric components which can be addded to RT models in OI observable
calculations. A geometric component also optionally includes a spectral profile to describe its flux dependence
accross wavelength.
"""

from distroi import constants

import numpy as np
from scipy.special import j1 as bessel_j1  # import Bessel function of the first kind
from abc import ABC, abstractmethod


class SpecDep(ABC):
    """
    Abstract class representing a spectral dependence to be attached to a geometric model component. Note that these do
    not represent full-fledged spectra. These are not absolute-flux calibrated, and only represent the dependence of
    flux on wavelength/frequency. A flux at a reference wavelength/frequency (derived from e.g. geometrical modelling)
    must be passed along in order to get absolute values.
    """

    @abstractmethod
    def flux_from_ref(self, x: np.ndarray | float, x_ref: float, ref_flux: float, x_form: str = 'lam',
                      flux_form: str = 'flam') -> np.ndarray | float:
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.
        One

        :param float, np.ndarray x:
        :param float, x_ref:
        :param float ref_flux:
        :param str x_form:
        :param str flux_form:
        :rtype: np.ndarray
        """
        pass


class GeomComp(ABC):
    """
    Abstract class representing a geometric model component.

    :ivar SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    """

    @abstractmethod
    def calc_vis(self, u: np.ndarray | float, v: np.ndarray | float, wavelength: np.ndarray | float = None,
                 ref_wavelength: float = None, ref_corr_flux: float = None) -> np.ndarray | float:
        """
        Calculate the visibility of the geometric component at given spatial frequencies. Wavelengths corresponding to
        these spatial frequencies and a reference total flux (at reference wavelength) can also be passed along, in
        which case the returned visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray u: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as u.
        :param np.ndarray v: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as v
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """
        pass


class UniformDisk(GeomComp):
    """
    Class representing a uniform disk geometric component.
    
    :param float radius: The radius of the disk in milli-arcsecond.
    :param tuple(float) coords:  2D tuples with (x, y) coordinates of the disk center's coordinates.
        Note that positive x is defined as leftward and positive y as upward (i.e. the East and North repesctively
        in the OI convention). If not given, will default to (0, 0).
    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar float radius: See parameter description.
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(self, radius: float, coords: tuple[float, float] = None, spec_dep: SpecDep = None):
        self.radius = radius
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        self.spec_dep = spec_dep

    def calc_vis(self, u: np.ndarray | float, v: np.ndarray | float, wavelength: np.ndarray | float = None,
                 ref_wavelength: float = None, ref_corr_flux: float = None) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Wavelengths corresponding to these spatial frequencies a
        and a reference flux value (at reference wavelength) can also be passed along, in which case the returned
        visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray u: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as u.
        :param np.ndarray v: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as v
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """
        norm_comp_vis = (2 * bessel_j1(np.pi * self.radius * constants.MAS2RAD * np.sqrt(u ** 2 + v ** 2)) /
                         (np.pi * self.radius * constants.MAS2RAD * np.sqrt(u ** 2 + v ** 2)))
        vis = norm_comp_vis * np.exp(-2j * np.pi * (u * self.coords[0] + v * self.coords[1]))  # position phase term
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            return vis
        else:
            # todo: express in correlated flux
            return vis


class PointSource(GeomComp):
    """
    Class representing a point source geometric component.

    :param tuple(float) coords:  2D tuples with (x, y) coordinates of the point's coordinates.
        Note that positive x is defined as leftward and positive y as upward (i.e. the East and North repesctively
        in the OI convention). If not given, will default to (0, 0).
    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(self, coords: tuple[float, float] = None, spec_dep: SpecDep = None):
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        self.spec_dep = spec_dep

    def calc_vis(self, u: np.ndarray | float, v: np.ndarray | float, wavelength: np.ndarray | float = None,
                 ref_wavelength: float = None, ref_corr_flux: float = None) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Wavelengths corresponding to these spatial frequencies a
        and a reference flux value (at reference wavelength) can also be passed along, in which case the returned
        visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray u: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as u.
        :param np.ndarray v: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as v
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """
        # todo: express in correlated flux
        vis = np.exp(-2j * np.pi * (u * self.coords[0] + v * self.coords[1]))
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            return vis
        else:
            # todo: express in correlated flux
            return vis


class Overresolved(GeomComp):
    """
    Class representing a fully resolved, a.k.a. overresolved, geometric component.

    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(self, spec_dep=None):
        self.spec_dep = spec_dep

    def calc_vis(self, u: np.ndarray | float, v: np.ndarray | float, wavelength: np.ndarray | float = None,
                 ref_wavelength: float = None, ref_corr_flux: float = None) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Automatically returns an

        :param np.ndarray u: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as u.
        :param np.ndarray v: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as v
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """
        if isinstance(u, float):
            vis = 0
        else:
            vis = np.zeros_like(u)
        return vis


class BlackBodySpecDep(SpecDep):
    """
    Blackbody spectral flux dependency.

    :param float temp: The blackbody temperature in Kelvin.
    :ivar float temp: See parameter description.
    """

    def __init__(self, temp):
        self.temp = temp

    def flux_from_ref(self, x: np.ndarray | float, x_ref: float, ref_flux: float, x_form: str = 'lam',
                      flux_form: str = 'flam') -> np.ndarray | float:
        # todo implement
        return


class PowerLawSpecDep(SpecDep):
    """
    Power law flux dependency.

    :param float power: The power of the flux profile.
    :param str flux_form: The format of the flux. This flux will follow the specified power law dependency.
        Options are 'flam' (default) and 'lam_flam', as well as their frequency analogues 'fnu' and
        'nu_fnu'. The formats in wavelength specification ('flam' and 'lam_flam') assume the power law dependency to be
        in wavelength (i.e. flux1 / flux2 = (wavelength1 / wavelength2) ^ power), while the ones in frequency
        specification assume the power law to be in frequency (i.e. flux1 / flux2 = (frequency1 / frequency2) ^ power).
        Note that a power law of 'flam' of power 'd' in wavelength will result in a power law for 'fnu' in frequency of
        power '-d-2', i.e. the transformation between 'fnu' and 'flam' matters.
    :ivar float power: See parameter description.
    """

    def __init__(self, power, flux_form='flam'):
        self.power = power
        self.flux_form = flux_form

    def flux_from_ref(self, x: np.ndarray | float, x_ref: float, ref_flux: float, x_form: str = 'lam',
                      flux_form: str = 'flam') -> np.ndarray | float:
        # todo implement
        return


class FlatSpecDep(SpecDep):
    """
    Flat spectral dependence.

    :param str flux_form: The format of the flux which follows the flat dependency.
        Options are 'flam' (default) and 'lam_flam', as well as their frequency analogues 'fnu' and 'nu_fnu'.
        The formats in wavelength specification ('flam' and 'lam_flam') assume the power law dependency to be in
        wavelength (i.e. flux1 / flux2 = (wavelength1 / wavelength2) ^ power), while the ones in frequency specification
        assume the power law to be in frequency (i.e. flux1 / flux2 = (frequency1 / frequency2) ^ power). Note that a
        flat law in 'flam' in wavelength will result in a power law for 'fnu' in frequency of power '-2', i.e. the
        transformation between 'fnu' and 'flam' matters.
    """

    def __init__(self, flux_form='flam'):
        self.flux_form = flux_form

    def flux_from_ref(self, x: np.ndarray | float, x_ref: float, ref_flux: float, x_form: str = 'lam',
                      flux_form: str = 'flam') -> np.ndarray | float:
        return


class ThinAccDiskSpecDep(SpecDep):
    """
    Spectral dependency of a thin, multi-blackbody accretion disk, as specified in De Prins et al. 2024. The disk
    has a blackbody temperature gradient derived assuming a certain accretion rate and radiative efficiency, the
    latter denoting the fraction of released gravitational power which is converted to radiation.

    :param float acc_rate: Accretion rate at the inner disk rim in units of M_sun yr^-1.
    :param float star_mass: Mass of the star at the centre of the disk in M_sun.
    :param float r_in: Inner disk rim radius in Solar radii.
    :param float r_out: Outer disk rim radius in Solar radii.
    :param float eta_rad: Radiative efficiency, expressed as a fraction between 0 and 1
    """

    def __init__(self, acc_rate: float, star_mass: float, r_in: float, r_out: float, eta_rad: float):
        self.acc_rate = acc_rate
        self.star_mass = star_mass
        self.r_in = r_in
        self.r_out = r_out
        self.eta_rad = eta_rad

    def flux_from_ref(self, x: np.ndarray | float, x_ref: float, ref_flux: float, x_form: str = 'lam',
                      flux_form: str = 'flam') -> np.ndarray | float:
        return
