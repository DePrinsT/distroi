"""A module to define spectral dependencies of model components.

Spectral dependencies can be attached to model components in the calculation of observables.

Notes
-----
Support is provided for spectral dependencies in F_lam, lam x F_lam, F_nu and nu x F_nu flux formats.
"""

from distroi.auxiliary import constants

import numpy as np
from abc import ABC, abstractmethod


# TODO: finnish cleaning up docstrings
class SpecDep(ABC):
    """Abstract representation of a total flux spectral dependency.

    Abstract class representing a spectral dependence to be attached to a geometric model component. Note that these do
    not represent full-fledged spectra. These are not absolute-flux calibrated, and only represent the dependence of
    flux on wavelength/frequency. A flux at a reference wavelength/frequency (derived from e.g. geometrical modelling)
    must be passed along in order to get absolute values.
    """

    ## TODO: actually implement the get_params method
    # @abstractmethod
    # def get_params(self) -> dict[str, param.Param]:
    #     """
    #     Retrieve a dictionary of parameters for this SpecDep, linking the name of the component within the
    #     SpecDep scope to the corresponding `Param` objects.
    #     """
    #     pass

    @abstractmethod
    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """Retrieve flux at wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        x_ref : np.ndarray or float
            Reference wavelength/frequency (in micron/Hz) at which to calculate the flux. In case ``flux_form = 'flam'``
            or `'lam_flam'`, `x_ref` is assumed to be a wavelength, while in case of `'fnu'` and `'nu_fnu'`, `x_ref`
            is assumed to be a frequency.
        ref_flux : float
            Reference flux from which to calculate the flux, in the specified `flux_form` format.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or `'lam_flam'`, `x` is assumed
            to be wavelengths, while in case of `'fnu'` and `'nu_fnu'`, `x` is assumed to be frequencies.

        Returns
        -------
        np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. Note that the
            units of both input and output will correspond to those of `x_ref` and `ref_flux`.
        """
        pass


# TODO: make this spectral dependence work when attached to an Image. This seems to cause an overflow at the moment.
class BlackBodySpecDep(SpecDep):
    """Blackbody spectral flux dependency.

    Parameters
    ----------
    temp : float
        The blackbody temperature in Kelvin.
    """

    def __init__(self, temp):
        self.temp = temp

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """Retrieve flux at wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        x_ref : np.ndarray or float
            Reference wavelength/frequency (in micron/Hz) at which to calculate the flux. In case ``flux_form = 'flam'``
            or `'lam_flam'`, `x_ref` is assumed to be a wavelength, while in case of `'fnu'` and `'nu_fnu'`, `x_ref`
            is assumed to be a frequency.
        ref_flux : float
            Reference flux from which to calculate the flux, in the specified `flux_form` format.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or `'lam_flam'`, `x` is assumed
            to be wavelengths, while in case of `'fnu'` and `'nu_fnu'`, `x` is assumed to be frequencies.

        Returns
        -------
        np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. Note that the
            units of both input and output will correspond to those of `x_ref` and `ref_flux`.
        """
        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            raise ValueError("Flux format 'flux_form' not recognized.")

        # different cases for requested flux format and power law flux format
        if flux_form == "flam":
            flux = ref_flux * (
                constants.bb_flam_at_wavelength(x, temp=self.temp)
                / constants.bb_flam_at_wavelength(x_ref, temp=self.temp)
            )
        if flux_form == "fnu":
            flux = ref_flux * (
                constants.bb_fnu_at_frequency(x, temp=self.temp) / constants.bb_fnu_at_frequency(x_ref, temp=self.temp)
            )
        if flux_form == "lam_flam":
            flux = (
                ref_flux
                * (x / x_ref)
                * (
                    constants.bb_flam_at_wavelength(x, temp=self.temp)
                    / constants.bb_flam_at_wavelength(x_ref, temp=self.temp)
                )
            )
        if flux_form == "nu_fnu":
            flux = (
                ref_flux
                * (x / x_ref)
                * (
                    constants.bb_fnu_at_frequency(x, temp=self.temp)
                    / constants.bb_fnu_at_frequency(x_ref, temp=self.temp)
                )
            )
        return flux


class PowerLawSpecDep(SpecDep):
    """Power law flux dependency.

    Parameters
    ----------
    power : float
        The power of the flux profile.
    flux_form : str, optional
        The format of the flux to be calculated. This flux will follow the specified power law dependency. Options are
        `'flam'` (default) and `'lam_flam'`, as well as their frequency analogues `'fnu'` and `'nu_fnu'`. The formats in
        wavelength specification (`'flam'` and `'lam_flam'`) assume the power law dependency to be in wavelength (i.e.
        ``flux1 / flux2 = (wavelength1 / wavelength2) ** power``), while the ones in frequency specification assume the
        power law to be in frequency (i.e. ``flux1 / flux2 = (frequency1 / frequency2)**power``).
        Note that a power law of 'flam' of power 'd' in wavelength will result in a power law for 'fnu' in frequency of
        power '-d-2', i.e. the transformation between 'fnu' and 'flam' matters.
    """

    def __init__(self, power, flux_form="flam"):
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            raise ValueError("Flux format 'flux_form' not recognized.")
        self.power = power
        self.flux_form = flux_form

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """Retrieve flux at wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        x_ref : np.ndarray or float
            Reference wavelength/frequency (in micron/Hz) at which to calculate the flux. In case ``flux_form = 'flam'``
            or `'lam_flam'`, `x_ref` is assumed to be a wavelength, while in case of `'fnu'` and `'nu_fnu'`, `x_ref`
            is assumed to be a frequency.
        ref_flux : float
            Reference flux from which to calculate the flux, in the specified `flux_form` format.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or `'lam_flam'`, `x` is assumed
            to be wavelengths, while in case of `'fnu'` and `'nu_fnu'`, `x` is assumed to be frequencies.

        Returns
        -------
        np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. Note that the
            units of both input and output will correspond to those of `x_ref` and `ref_flux`.
        """

        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            raise ValueError("Flux format 'flux_form' not recognized")

        # different cases for requested flux format and power law flux format
        if flux_form == "flam" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** self.power
        elif flux_form == "flam" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** (self.power - 1)
        elif flux_form == "flam" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** (-self.power - 2)
        elif flux_form == "flam" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** (-self.power - 1)
        elif flux_form == "fnu" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** (-self.power - 2)
        elif flux_form == "fnu" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** (-self.power - 1)
        elif flux_form == "fnu" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** self.power
        elif flux_form == "fnu" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** (self.power - 1)
        elif flux_form == "lam_flam" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** (self.power + 1)
        elif flux_form == "lam_flam" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** self.power
        elif flux_form == "lam_flam" and flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** (-self.power - 1)
        elif flux_form == "lam_flam" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** -self.power
        elif flux_form == "nu_fnu" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** (-self.power - 1)
        elif flux_form == "nu_fnu" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** -self.power
        elif flux_form == "nu_fnu" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** (self.power + 1)
        elif flux_form == "nu_fnu" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** self.power
        return flux


class FlatSpecDep(SpecDep):
    """Flat spectral dependency.

    Parameters
    ----------
    flux_form : str, optional
        The format of the flux which follows the flat dependency. Options are `'flam'` (default) and `'lam_flam'`,
        as well as their frequency analogues `'fnu'` and `'nu_fnu'`. The formats in wavelength specification
        (`'flam'` and `'lam_flam'`) assume the power law dependency to be in wavelength (i.e. ``flux1 / flux2 = 
        (wavelength1 / wavelength2) ^ power``), while the ones in frequency specification assume the power law to be
        in frequency (i.e. ``flux1 / flux2 = (frequency1 / frequency2) ^ power``). Note that a flat law in 'flam' in 
        wavelength will result in a power law for 'fnu' in frequency of power '-2', i.e. the transformation between
        'fnu' and 'flam' matters.
    """

    def __init__(self, flux_form="flam"):
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            raise ValueError("Flux format 'flux_form' not recognized.")
        self.flux_form = flux_form

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """Retrieve flux at wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        x_ref : np.ndarray or float
            Reference wavelength/frequency (in micron/Hz) at which to calculate the flux. In case ``flux_form = 'flam'``
            or `'lam_flam'`, `x_ref` is assumed to be a wavelength, while in case of `'fnu'` and `'nu_fnu'`, `x_ref`
            is assumed to be a frequency.
        ref_flux : float
            Reference flux from which to calculate the flux, in the specified `flux_form` format.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or `'lam_flam'`, `x` is assumed
            to be wavelengths, while in case of `'fnu'` and `'nu_fnu'`, `x` is assumed to be frequencies.

        Returns
        -------
        np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. Note that the
            units of both input and output will correspond to those of `x_ref` and `ref_flux`.
        """

        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            raise ValueError("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")

        # different cases for requested flux format and power law flux format
        if flux_form == "flam" and self.flux_form == "flam":
            flux = ref_flux
        elif flux_form == "flam" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "flam" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** -2
        elif flux_form == "flam" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "fnu" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** -2
        elif flux_form == "fnu" and self.flux_form == "lam_flam":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "fnu" and self.flux_form == "fnu":
            flux = ref_flux
        elif flux_form == "fnu" and self.flux_form == "nu_fnu":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "lam_flam" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref)
        elif flux_form == "lam_flam" and self.flux_form == "lam_flam":
            flux = ref_flux
        elif flux_form == "lam_flam" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "lam_flam" and self.flux_form == "nu_fnu":
            flux = ref_flux
        elif flux_form == "nu_fnu" and self.flux_form == "flam":
            flux = ref_flux * (x / x_ref) ** -1
        elif flux_form == "nu_fnu" and self.flux_form == "lam_flam":
            flux = ref_flux
        elif flux_form == "nu_fnu" and self.flux_form == "fnu":
            flux = ref_flux * (x / x_ref)
        elif flux_form == "nu_fnu" and self.flux_form == "nu_fnu":
            flux = ref_flux
        return flux


class ThinAccDiskSpecDep(SpecDep):
    """Spectral dependency of a thin, multi-blackbody accretion disk.

    Spectral dependency of a thin, multi-blackbody accretion disk, as specified in De Prins et al. 2024. The disk
    has a blackbody temperature gradient derived assuming a certain accretion rate and radiative efficiency, the
    latter denoting the fraction of released gravitational power which is converted to radiation.

    Parameters
    ----------
    acc_rate : float
        Accretion rate at the inner disk rim in units of M_sun yr^-1.
    star_mass : float
        Mass of the star at the centre of the disk in M_sun.
    r_in : float
        Inner disk rim radius in Solar radii.
    r_out : float
        Outer disk rim radius in Solar radii.
    eta_rad : float
        Radiative efficiency, expressed as a fraction between 0 and 1.
    """

    def __init__(
        self,
        acc_rate: float,
        star_mass: float,
        r_in: float,
        r_out: float,
        eta_rad: float,
    ):
        self.acc_rate = acc_rate
        self.star_mass = star_mass
        self.r_in = r_in
        self.r_out = r_out
        self.eta_rad = eta_rad

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """Retrieve flux at wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        x_ref : np.ndarray or float
            Reference wavelength/frequency (in micron/Hz) at which to calculate the flux. In case ``flux_form = 'flam'``
            or `'lam_flam'`, `x_ref` is assumed to be a wavelength, while in case of `'fnu'` and `'nu_fnu'`, `x_ref`
            is assumed to be a frequency.
        ref_flux : float
            Reference flux from which to calculate the flux, in the specified `flux_form` format.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or `'lam_flam'`, `x` is assumed
            to be wavelengths, while in case of `'fnu'` and `'nu_fnu'`, `x` is assumed to be frequencies.

        Returns
        -------
        np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. Note that the
            units of both input and output will correspond to those of `x_ref` and `ref_flux`.
        """
        # TODO: implement!
        return
