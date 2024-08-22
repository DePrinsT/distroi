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
    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        :param np.ndarray | float x: Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        :param np.ndarray | float x_ref: Reference wavelength/frequency (in micron/Hz) at which to calculate the flux.
            In case flux_form = 'flam' or 'lam_flam', x_ref is assumed to be a wavelength, while in case of 'fnu' and
            'nu_fnu', x_ref is assumed to be a frequency.
        :param float ref_flux: Reference flux from which to calculate the flux, in the specified 'flux_form' format.
        :param str flux_form: The format of the flux to be calculated. Options are 'flam' (default) and 'lam_flam',
            as well as their frequency analogues 'fnu' and 'nu_fnu'. In case flux_form = 'flam' or 'lam_flam', x is 
            assumed to be wavelengths, while in case of 'fnu' and 'nu_fnu', x is assumed to be frequencies.
        :return flux: The flux calculated at x using the reference wavelength/frequency and reference flux value.
            Note that the units of both input and output will correspond to those of x_ref and ref_flux.
        :rtype: np.ndarray | float
        """
        pass


class GeomComp(ABC):
    """
    Abstract class representing a geometric model component.

    :ivar SpecDep | None spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will
        be assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    """

    @abstractmethod
    def calc_vis(
        self,
        uf: np.ndarray | float,
        vf: np.ndarray | float,
        wavelength: np.ndarray | float | None = None,
        ref_wavelength: float | None = None,
        ref_corr_flux: float | None = None,
    ) -> np.ndarray | float:
        """
        Calculate the visibility of the geometric component at given spatial frequencies. Wavelengths corresponding to
        these spatial frequencies and a reference total flux (at reference wavelength) can also be passed along, in
        which case the returned visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray | float uf: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as uf.
        :param np.ndarray | float vf: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as vf.
        :param np.ndarray | float wavelength: 1D array with wavelength values in micron.
        :param float | None ref_wavelength: Reference wavelength in micron.
        :param float | None ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray | float
        """
        pass


class UniformDisk(GeomComp):
    """
    Class representing a uniform disk geometric component.

    :param float diameter: The radius of the disk in milli-arcsecond.
    :param tuple(float) coords:  2D tuples with (x, y) coordinates of the disk center's coordinates (in mas).
        Note that positive x is defined as leftward and positive y as upward (i.e. the East and North repesctively
        in the OI convention). If not given, will default to (0, 0).
    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar float radius: See parameter description.
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(
        self,
        diameter: float,
        coords: tuple[float, float] | None = None,
        spec_dep: SpecDep | None = None,
    ):
        """
        Initializes a UniformDisk object.
        """
        self.diameter = diameter
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if self.spec_dep is not None:
            self.spec_dep = spec_dep  # set spectral dependence if given
        else:
            self.spec_dep = FlatSpecDep(flux_form="fnu")  # otherwise, assume a flat spectrum in correlated flux

    def calc_vis(
        self,
        uf: np.ndarray | float,
        vf: np.ndarray | float,
        wavelength: np.ndarray | float | None = None,
        ref_wavelength: float | None = None,
        ref_corr_flux: float | None = None,
    ) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Wavelengths corresponding to these spatial frequencies a
        and a reference flux value (at reference wavelength) can also be passed along, in which case the returned
        visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray uf: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as uf.
        :param np.ndarray vf: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as vf.
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """

        norm_comp_vis = (
            2
            * bessel_j1(np.pi * self.diameter * constants.MAS2RAD * np.sqrt(uf**2 + vf**2))
            / (np.pi * self.diameter * constants.MAS2RAD * np.sqrt(uf**2 + vf**2))
        )
        # add position phase term
        norm_comp_vis_phase = norm_comp_vis * np.exp(
            -2j * np.pi * (uf * self.coords[0] * constants.MAS2RAD + vf * self.coords[1] * constants.MAS2RAD)
        )
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            vis = norm_comp_vis_phase
            return vis
        else:
            frequency = constants.SPEED_OF_LIGHT / (wavelength * constants.MICRON2M)
            ref_frequency = constants.SPEED_OF_LIGHT / (ref_wavelength * constants.MICRON2M)
            corr_flux = self.spec_dep.flux_from_ref(
                x=frequency,
                x_ref=ref_frequency,
                ref_flux=ref_corr_flux,
                flux_form="fnu",
            )
            vis = corr_flux * norm_comp_vis_phase
            return vis


class Gaussian(GeomComp):
    """
    Class representing a Gaussian geomteric component:

    :param float fwhm: Full-width-half-maximum of the Gaussian in the image plane (in mas units).
    :param tuple(float) coords:  2D tuples with (x, y) coordinates of the point's coordinates (in mas).
        Note that positive x is defined as leftward and positive y as upward (i.e. the East and North repesctively
        in the OI convention). If not given, will default to (0, 0).
    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar float fwhm: See parameter description.
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(self, fwhm: float, coords: tuple[float, float] | None = None, spec_dep: SpecDep | None = None):
        """
        Initializes a Gaussian object.
        """
        self.fwhm = fwhm
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if self.spec_dep is not None:
            self.spec_dep = spec_dep  # set spectral dependence if given
        else:
            self.spec_dep = FlatSpecDep(flux_form="fnu")  # otherwise, assume a flat spectrum in correlated flux

    def calc_vis(
        self,
        uf: np.ndarray | float,
        vf: np.ndarray | float,
        wavelength: np.ndarray | float | None = None,
        ref_wavelength: float | None = None,
        ref_corr_flux: float | None = None,
    ) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Wavelengths corresponding to these spatial frequencies a
        and a reference flux value (at reference wavelength) can also be passed along, in which case the returned
        visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray uf: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as uf.
        :param np.ndarray vf: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as vf.
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """
        # TODO: check if correct, lots of sources online disagree
        norm_comp_vis = np.exp(-1 * np.pi**2 * (self.fwhm * constants.MAS2RAD) ** 2 * (uf**2 + vf**2) / (4 * np.log(2)))
        # add position phase term
        norm_comp_vis_phase = norm_comp_vis * np.exp(
            -2j * np.pi * (uf * self.coords[0] * constants.MAS2RAD + vf * self.coords[1] * constants.MAS2RAD)
        )
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            vis = norm_comp_vis_phase
            return vis
        else:
            frequency = constants.SPEED_OF_LIGHT / (wavelength * constants.MICRON2M)
            ref_frequency = constants.SPEED_OF_LIGHT / (ref_wavelength * constants.MICRON2M)
            corr_flux = self.spec_dep.flux_from_ref(
                x=frequency,
                x_ref=ref_frequency,
                ref_flux=ref_corr_flux,
                flux_form="fnu",
            )
            vis = corr_flux * norm_comp_vis_phase
            return vis


class PointSource(GeomComp):
    """
    Class representing a point source geometric component.

    :param tuple(float) coords:  2D tuples with (x, y) coordinates of the point's coordinates (in mas).
        Note that positive x is defined as leftward and positive y as upward (i.e. the East and North repesctively
        in the OI convention). If not given, will default to (0, 0).
    :param SpecDep spec_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in correlated flux accross wavelength (note that flatness in correlated flux means a spectral
        dependency ~ wavelength ^ -2 for F_lam).
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spec_dep: See parameter description.
    """

    def __init__(self, coords: tuple[float, float] | None = None, spec_dep: SpecDep | None = None):
        """
        Initializes a PointSource object.
        """
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if self.spec_dep is not None:
            self.spec_dep = spec_dep  # set spectral dependence if given
        else:
            self.spec_dep = FlatSpecDep(flux_form="fnu")  # otherwise, assume a flat spectrum in correlated flux

    def calc_vis(
        self,
        uf: np.ndarray | float,
        vf: np.ndarray | float,
        wavelength: np.ndarray | float | None = None,
        ref_wavelength: float | None = None,
        ref_corr_flux: float | None = None,
    ) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Wavelengths corresponding to these spatial frequencies a
        and a reference flux value (at reference wavelength) can also be passed along, in which case the returned
        visibilities will be in correlated flux (Jy) instead of normalized.

        :param np.ndarray uf: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as uf.
        :param np.ndarray vf: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as vf.
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """

        norm_comp_vis = np.exp(
            -2j * np.pi * (uf * self.coords[0] * constants.MAS2RAD + vf * self.coords[1] * constants.MAS2RAD)
        )
        print("stuff going on")
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            vis = norm_comp_vis
            return vis
        else:
            frequency = constants.SPEED_OF_LIGHT / (wavelength * constants.MICRON2M)
            ref_frequency = constants.SPEED_OF_LIGHT / (ref_wavelength * constants.MICRON2M)
            corr_flux = self.spec_dep.flux_from_ref(
                x=frequency,
                x_ref=ref_frequency,
                ref_flux=ref_corr_flux,
                flux_form="fnu",
            )
            vis = corr_flux * norm_comp_vis
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
        """
        Initializes an Overresolved object.
        """
        if self.spec_dep is not None:
            self.spec_dep = spec_dep  # set spectral dependence if given
        else:
            self.spec_dep = FlatSpecDep(flux_form="fnu")  # otherwise, assume a flat spectrum in correlated flux

    def calc_vis(
        self,
        uf: np.ndarray | float,
        vf: np.ndarray | float,
        wavelength: np.ndarray | float | None = None,
        ref_wavelength: float | None = None,
        ref_corr_flux: float | None = None,
    ) -> np.ndarray | float:
        """
        Calculate the visibility at given spatial frequencies. Automatically returns an

        :param np.ndarray uf: 1D array with spatial x-axis frequencies in 1/radian. Must be the same size as uf.
        :param np.ndarray vf: 1D array with spatial y-axis frequencies in 1/radian. Must be the same size as vf.
        :param np.ndarray wavelength: 1D array with wavelength values in micron.
        :param float ref_wavelength: Reference wavelength in micron.
        :param float ref_corr_flux: Reference correlated flux in Jy corresponding to ref_wavelength. If provided
            together with ref_corr_flux, then the returned visibilities are in correlated flux.
        :return vis: 1D array with the calculated visibilities (normalized or in corrleated flux, depending on the
            optional arguments)
        :rtype: np.ndarray
        """

        if isinstance(uf, float):
            vis = 0
        else:
            vis = np.zeros_like(uf)
        return vis

# TODO: make this spectral dependence work when attached to an ImageFFT. This seems not to work at the moment.
class BlackBodySpecDep(SpecDep):
    """
    Blackbody spectral flux dependency.

    :param float temp: The blackbody temperature in Kelvin.
    :ivar float temp: See parameter description.
    """

    def __init__(self, temp):
        """
        Initializes a BlackBodySpecDep object.
        """
        self.temp = temp

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        :param np.ndarray | float x: Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        :param np.ndarray | float x_ref: Reference wavelength/frequency (in micron/Hz) at which to calculate the flux.
            In case flux_form = 'flam' or 'lam_flam', x_ref is assumed to be a wavelength, while in case of 'fnu' and
            'nu_fnu', x_ref is assumed to be a frequency.
        :param float ref_flux: Reference flux from which to calculate the flux, in the specified 'flux_form' format.
        :param str flux_form: The format of the flux to be calculated. Options are 'flam' (default) and 'lam_flam',
            as well as their frequency analogues 'fnu' and 'nu_fnu'. In case flux_form = 'flam' or 'lam_flam', x is
            assumed to be wavelengths, while in case of 'fnu' and 'nu_fnu', x is assumed to be frequencies.
        :return flux: The flux calculated at x using the reference wavelength/frequency and reference flux value. Note
            that the units of both input and output will correspond to those of x_ref and ref_flux.
        :rtype: np.ndarray | float
        """
        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            print("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")
            flux_form = "flam"

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
    """
    Power law flux dependency.

    :param float power: The power of the flux profile.
    :param str flux_form: The format of the flux to be calculated. This flux will follow the specified power law
        dependency. Options are 'flam' (default) and 'lam_flam', as well as their frequency analogues 'fnu' and
        'nu_fnu'. The formats in wavelength specification ('flam' and 'lam_flam') assume the power law dependency to be
        in wavelength (i.e. flux1 / flux2 = (wavelength1 / wavelength2) ^ power), while the ones in frequency
        specification assume the power law to be in frequency (i.e. flux1 / flux2 = (frequency1 / frequency2) ^ power).
        Note that a power law of 'flam' of power 'd' in wavelength will result in a power law for 'fnu' in frequency of
        power '-d-2', i.e. the transformation between 'fnu' and 'flam' matters.
    :ivar float power: See parameter description.
    """

    def __init__(self, power, flux_form="flam"):
        """
        Initializes a PowerLawSpecDep object.
        """
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            print("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")
            self.flux_form = "flam"
        self.power = power
        self.flux_form = flux_form

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        :param np.ndarray | float x: Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        :param np.ndarray | float x_ref: Reference wavelength/frequency (in micron/Hz) at which to calculate the flux.
            In case flux_form = 'flam' or 'lam_flam', x_ref is assumed to be a wavelength, while in case of 'fnu' and
            'nu_fnu', x_ref is assumed to be a frequency.
        :param float ref_flux: Reference flux from which to calculate the flux, in the specified 'flux_form' format.
        :param str flux_form: The format of the flux to be calculated. Options are 'flam' (default) and 'lam_flam',
            as well as their frequency analogues 'fnu' and 'nu_fnu'. In case flux_form = 'flam' or 'lam_flam', x is
            assumed to be wavelengths, while in case of 'fnu' and 'nu_fnu', x is assumed to be frequencies.
        :return flux: The flux calculated at x using the reference wavelength/frequency and reference flux value. Note
            that the units of both input and output will correspond to those of x_ref and ref_flux.
        :rtype: np.ndarray | float
        """

        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            print("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")
            flux_form = "flam"

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

    def __init__(self, flux_form="flam"):
        """
        Initializes a FlatSpecDep object.
        """
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            print("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")
            self.flux_form = "flam"
        self.flux_form = flux_form

    def flux_from_ref(
        self,
        x: np.ndarray | float,
        x_ref: float,
        ref_flux: float,
        flux_form: str = "flam",
    ) -> np.ndarray | float:
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        :param np.ndarray | float x: Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        :param np.ndarray | float x_ref: Reference wavelength/frequency (in micron/Hz) at which to calculate the flux.
            In case flux_form = 'flam' or 'lam_flam', x_ref is assumed to be a wavelength, while in case of 'fnu' and
            'nu_fnu', x_ref is assumed to be a frequency.
        :param float ref_flux: Reference flux from which to calculate the flux, in the specified 'flux_form' format.
        :param str flux_form: The format of the flux to be calculated. Options are 'flam' (default) and 'lam_flam',
            as well as their frequency analogues 'fnu' and 'nu_fnu'. In case flux_form = 'flam' or 'lam_flam', x is
            assumed to be wavelengths, while in case of 'fnu' and 'nu_fnu', x is assumed to be frequencies.
        :return flux: The flux calculated at x using the reference wavelength/frequency and reference flux value. Note
            that the units of both input and output will correspond to those of x_ref and ref_flux.
        :rtype: np.ndarray | float
        """

        # check requested flux format
        if flux_form not in ("flam", "lam_flam", "fnu", "nu_fnu"):
            print("Flux format 'flux_form' not recognized, defaulting to 'flam' instead.")
            flux_form = "flam"

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

    def __init__(
        self,
        acc_rate: float,
        star_mass: float,
        r_in: float,
        r_out: float,
        eta_rad: float,
    ):
        """
        Initializes a ThinAccDiskSpecDep object.
        """
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
        """
        Retrieve the flux at certain wavelengths/frequencies when given a reference flux value and wavelength/frequency.

        :param np.ndarray | float x: Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        :param np.ndarray | float x_ref: Reference wavelength/frequency (in micron/Hz) at which to calculate the flux.
            In case flux_form = 'flam' or 'lam_flam', x_ref is assumed to be a wavelength, while in case of 'fnu' and
            'nu_fnu', x_ref is assumed to be a frequency.
        :param float ref_flux: Reference flux from which to calculate the flux, in the specified 'flux_form' format.
        :param str flux_form: The format of the flux to be calculated. Options are 'flam' (default) and 'lam_flam',
            as well as their frequency analogues 'fnu' and 'nu_fnu'. In case flux_form = 'flam' or 'lam_flam', x is
            assumed to be wavelengths, while in case of 'fnu' and 'nu_fnu', x is assumed to be frequencies.
        :return flux: The flux calculated at x using the reference wavelength/frequency and reference flux value. Note
            that the units of both input and output will correspond to those of x_ref and ref_flux.
        :rtype: np.ndarray | float
        """
        # TODO: implement!
        return


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     # TEST THAT WORKS
#     constants.set_matplotlib_params()  # set project matplotlib parameters
#     # temp = 3000
#     # spec_dep = BlackBodySpecDep(temp=temp)
#     spec_dep = PowerLawSpecDep(power=-4, flux_form='flam')
#     # disk = UniformDisk(radius=(1 / (100 * 1e6) * constants.RAD2MAS), coords=(0, 0), spec_dep=spec_dep)
#     disk = Gaussian(fwhm=(0.4 / (100 * 1e6) * constants.RAD2MAS), coords=(0, 0), spec_dep=spec_dep)
#     n_points = 100
#     n_wave = 6
#     uf = np.array(list(np.linspace(1e6, 100 * 1e6, n_points)) * n_wave)
#     vf = np.array(list(np.linspace(1e6, 100 * 1e6, n_points)) * n_wave)
#     wave = np.repeat(np.linspace(0.5, 1.5, n_wave), n_points)
#     vis = disk.calc_vis(uf, vf, wavelength=wave, ref_wavelength=1, ref_corr_flux=1)
#
#     fig, ax = plt.subplots()
#     scat = ax.scatter(np.sqrt(uf ** 2 + vf ** 2), abs(vis) / np.max(abs(vis)), s=5, c=wave, cmap='inferno')
#     ax.set_xlabel(r'Baseline ($\lambda$)')
#     ax.set_ylabel(r'$F_{corr} (Jy)$')
#     plt.colorbar(scat)
#     print(np.min(1 / uf) * constants.RAD2MAS)
#
#     fig, ax = plt.subplots()
#     scat = ax.scatter(np.sqrt(uf ** 2 + vf ** 2), np.angle(vis, deg=True), s=5, c=wave, cmap='inferno')
#     ax.set_xlabel(r'Baseline ($\lambda$)')
#     ax.set_ylabel(r'$\phi_{CP}$ ($^\circ$)')
#     plt.colorbar(scat)
#     plt.show()
