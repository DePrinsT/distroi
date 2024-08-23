"""
Defines the classes and methods for representing geometric components and calculating their complex visibilities.

These geometric components can be used in conjunction to RT models in the calculation of interferometric observables.
A geometric component also includes a spectral profile to describe its flux dependence accross wavelength.
By default, the spectrum is assumed to be flat in F_lam.

NOTE: A flat spectrum in F_lam implies a dependency of F_nu ~ nu^-2 ~ lambda^2. Hence, the correlated flux will not be
flat. Take this in mind when calculating or comparing to observations in correlated flux.
"""

from distroi import constants
from distroi import spec_dep

import numpy as np
from scipy.special import j1 as bessel_j1  # import Bessel function of the first kind
from abc import ABC, abstractmethod


class GeomComp(ABC):
    """
    Abstract class representing a geometric model component.

    :ivar spec_dep.SpecDep | None spc_dep: Optional spectral dependence of the component. If None, the spectral
        dependency will be assumed flat in F_lam flux accross wavelength (note that flatness in F_lam means a spectral
        dependency ~ wavelength^2 ~ frequency^-2 for F_nu, and thus for the correlated flux).
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
    :param SpecDep spc_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in F_lam flux accross wavelength (note that flatness in F_lam means a spectral
        dependency ~ wavelength^2 ~ frequency^-2 for F_nu, and thus for the correlated flux).
    :ivar float radius: See parameter description.
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spc_dep: See parameter description.
    """

    def __init__(
        self,
        diameter: float,
        coords: tuple[float, float] | None = None,
        spc_dep: spec_dep.SpecDep | None = None,
    ):
        """
        Initializes a UniformDisk object.
        """
        self.diameter = diameter
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if spc_dep is not None:
            self.spc_dep = spc_dep  # set spectral dependence if given
        else:
            self.spc_dep = spec_dep.FlatSpecDep(flux_form="flam")  # otherwise, assume flat spectrum in F_lam

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
            corr_flux = self.spc_dep.flux_from_ref(
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
    :param SpecDep spc_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in F_lam flux accross wavelength (note that flatness in F_lam means a spectral
        dependency ~ wavelength^2 ~ frequency^-2 for F_nu, and thus for the correlated flux).
    :ivar float fwhm: See parameter description.
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spc_dep: See parameter description.
    """

    def __init__(self, fwhm: float, coords: tuple[float, float] | None = None, spc_dep: spec_dep.SpecDep | None = None):
        """
        Initializes a Gaussian object.
        """
        self.fwhm = fwhm
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if spc_dep is not None:
            self.spc_dep = spc_dep  # set spectral dependence if given
        else:
            self.spc_dep = spec_dep.FlatSpecDep(flux_form="flam")  # otherwise, assume a flat spectrum in F_lam

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
            corr_flux = self.spc_dep.flux_from_ref(
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
    :param SpecDep spc_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in F_lam flux accross wavelength (note that flatness in F_lam means a spectral
        dependency ~ wavelength^2 ~ frequency^-2 for F_nu, and thus for the correlated flux).
    :ivar tuple(float) coords: See parameter description.
    :ivar SpecDep spc_dep: See parameter description.
    """

    def __init__(self, coords: tuple[float, float] | None = None, spc_dep: spec_dep.SpecDep | None = None):
        """
        Initializes a PointSource object.
        """
        if coords is None:
            self.coords = (0, 0)
        else:
            self.coords = coords
        if spc_dep is not None:
            self.spc_dep = spc_dep  # set spectral dependence if given
        else:
            self.spc_dep = spec_dep.FlatSpecDep(flux_form="flam")  # otherwise, assume a flat spectrum in F_lam

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
        if wavelength is None or ref_wavelength is None or ref_corr_flux is None:
            vis = norm_comp_vis
            return vis
        else:
            frequency = constants.SPEED_OF_LIGHT / (wavelength * constants.MICRON2M)
            ref_frequency = constants.SPEED_OF_LIGHT / (ref_wavelength * constants.MICRON2M)
            corr_flux = self.spc_dep.flux_from_ref(
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

    :param SpecDep spc_dep: Optional spectral dependence of the component. If None, the spectral dependency will be
        assumed flat in F_lam flux accross wavelength (note that flatness in F_lam means a spectral
        dependency ~ wavelength^2 ~ frequency^-2 for F_nu, and thus for the correlated flux).
    :ivar SpecDep spc_dep: See parameter description.
    """

    def __init__(self, spc_dep=None):
        """
        Initializes an Overresolved object.
        """
        if spc_dep is not None:
            self.spc_dep = spc_dep  # set spectral dependence if given
        else:
            self.spc_dep = spc_dep.FlatSpecDep(flux_form="flam")  # otherwise, assume a flat spectrum in F_lam

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
