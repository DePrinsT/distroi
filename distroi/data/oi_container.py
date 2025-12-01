"""
Contains a class to store optical interferometric (OI) observables and additional functions to take radiative
transfer (RT) model images and convert them to interferometric observables at the spatial frequencies of data
stored in the OIFITS format. Currently supports the following combinations of observables: Squared visibilities -
Closure phases; Visibilities - Closure phases; Correlated fluxes (formally stored as visibilities) - Closure phases.
"""

from distroi.auxiliary import constants, nan_filter_arrays
from distroi.data import image
from distroi.data import sed
from distroi.model.geom_comp import geom_comp
from distroi.auxiliary import select_data_oifits

import os

import numpy as np

from typing import Literal

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


# TODO: add support for bispectrum amplitude
class OIContainer:
    """
    Class to contain optical interferometry observables, where each observable is stored in the form of raveled 1D
    numpy arrays. Based on the Data class in ReadOIFITS.py, where the observables are stored per OIFITS table,
    but the raveled form makes calculations easier and less prone to error. Currently supports visibilities (both in
    normalized and correlated flux form), squared visibilities and closure phases. In this format this class and the
    methods below can be expanded to accommodate for different kinds of observables (i.e. addition of differential
    visibilities). Can contain observables either related to astronomical observations or model RT images.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys and values representing all instance variables described below (excluding fcorr).
    fcorr : bool, optional
        Set to True if the visibilities are to be stored as correlated fluxes in Jy.

    Attributes
    ----------
    vis_in_fcorr : bool
        Whether visibilities are stored in correlated flux or not.
    v_uf : np.ndarray
        u-axis spatial freqs in 1/rad for visibility data.
    v_vf : np.ndarray
        v-axis spatial freqs in 1/rad for visibility data.
    v_wave : np.ndarray
        Wavelengths in micron for visibility data.
    v : np.ndarray
        Visibilities, either normalized visibilities or correlated flux in Jansky.
    v_err : np.ndarray
        Error on visibilities.
    v_base : np.ndarray
        Baseline length in MegaLambda for visibility data.
    v2_uf : np.ndarray
        u-axis spatial freqs in 1/rad for squared visibility data.
    v2_vf : np.ndarray
        v-axis spatial freqs in 1/rad for squared visibility data.
    v2_wave : np.ndarray
        Wavelengths in micron for squared visibility data.
    v2 : np.ndarray
        Squared visibilities.
    v2_err : np.ndarray
        Error on squared visibilities.
    v2_base : np.ndarray
        Baseline length in MegaLambda for squared visibility data.
    t3_uf1 : np.ndarray
        u-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    t3_vf1 : np.ndarray
        v-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    t3_uf2 : np.ndarray
        u-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    t3_vf2 : np.ndarray
        v-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    t3_uf3 : np.ndarray
        u-axis spatial freqs in 1/rad for the 3rd projected baseline along the closure triangle.
    t3_vf3 : np.ndarray
        v-axis spatial freqs in 1/rad for the 3rd projected baseline along the closure triangle.
    t3_wave : np.ndarray
        Wavelengths in micron for closure phase data.
    t3phi : np.ndarray
        Closure phases in degrees.
    t3phi_err : np.ndarray
        Error on closure phases.
    t3_bmax : np.ndarray
        Maximum baseline length along the closure triangle in units of MegaLambda.
    """

    def __init__(self, dictionary: dict[str, np.ndarray], fcorr: bool = False):
        self.vis_in_fcorr = fcorr  # set if visibilities are in correlated flux

        # Properties for visibility (V) data
        self.v_uf = None  # u-axis spatial freqs in 1/rad
        self.v_vf = None  # v-axis spatial freqs in 1/rad
        self.v_wave = None  # wavelengths in micron
        self.v = None  # visibilities (either normalized visibilities or correlated flux in Janksy)
        self.v_err = None  # error on visibilities
        self.v_base = None  # baseline length in MegaLambda (i.e. 1e6*v_wave)

        # Properties for squared visibility (V2) data
        self.v2_uf = None  # u-axis spatial freqs in 1/rad
        self.v2_vf = None  # v-axis spatial freqs in 1/rad
        self.v2_wave = None  # wavelengths in micron
        self.v2 = None  # squared visibilities
        self.v2_err = None  # errors on squared visibilities
        self.v2_base = None  # baseline length in MegaLambda (i.e. 1e6*v_wave)

        # Properties for closure phase (T3_PHI) data
        self.t3_uf1 = None  # u-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3_vf1 = None  # v-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3_uf2 = None
        self.t3_vf2 = None
        self.t3_uf3 = None
        self.t3_vf3 = None
        self.t3_wave = None  # wavelenghts in micron
        self.t3_phi = None  # closure phases in degrees
        self.t3_phierr = None  # errors on closure phases
        self.t3_bmax = None  # maximum baseline lengths along the closure triangles in units of MegaLambda

        # Read in from the dictionary
        if dictionary is not None:
            self.v_uf = dictionary["v_uf"]
            self.v_vf = dictionary["v_vf"]
            self.v_wave = dictionary["v_wave"]
            self.v = dictionary["v"]
            self.v_err = dictionary["v_err"]
            self.v_base = dictionary["v_base"]

            self.v2_uf = dictionary["v2_uf"]
            self.v2_vf = dictionary["v2_vf"]
            self.v2_wave = dictionary["v2_wave"]
            self.v2 = dictionary["v2"]
            self.v2_err = dictionary["v2_err"]
            self.v2_base = dictionary["v2_base"]

            self.t3_uf1 = dictionary["t3_uf1"]
            self.t3_vf1 = dictionary["t3_vf1"]
            self.t3_uf2 = dictionary["t3_uf2"]
            self.t3_vf2 = dictionary["t3_vf2"]
            self.t3_uf3 = dictionary["t3_uf3"]
            self.t3_vf3 = dictionary["t3_vf3"]
            self.t3_wave = dictionary["t3_wave"]
            self.t3_phi = dictionary["t3phi"]
            self.t3_phierr = dictionary["t3phi_err"]
            self.t3_bmax = dictionary["t3_bmax"]

    def plot_data(
        self,
        fig_dir: str = None,
        log_plotv: bool = False,
        plot_vistype: Literal["vis2", "vis", "fcorr"] = "vis2",
        show_plots: bool = True,
        data_figsize: tuple = (10, 5),
    ) -> None:
        """
        Plots the data included in the OIContainer instance. Currently, plots uv coverage, a (squared) visibility curve
        and closure phases.

        Parameters
        ----------
        fig_dir : str, optional
            Directory to store plots in.
        log_plotv : bool, optional
            Set to True for a logarithmic y-scale in the (squared) visibility plot.
        plot_vistype : {'vis2', 'vis', 'fcorr'}, optional
            Sets the type of visibility to be plotted. 'vis2' for squared visibilities or 'vis' for visibilities
            (either normalized or correlated flux in Jy, as implied by the OIContainer objects).
        show_plots : bool, optional
            Set to False if you do not want the plots to be shown during your python instance. Note that if True,
            this freezes further code execution until the plot windows are closed.

        Returns
        -------
        None
        """
        # check if valid plot_vistype passed along
        valid_vistypes = ["vis2", "vis", "fcorr"]
        if plot_vistype not in valid_vistypes:
            raise ValueError(f"Warning: Invalid plot_vistype '{plot_vistype}'. Valid options are: {valid_vistypes}. ")

        # create plotting directory if it doesn't exist yet
        if fig_dir is not None:
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)

        # set spatial frequencies, visibilities and plotting label based on specified option
        if plot_vistype == "vis2":
            uf, vf = self.v2_uf, self.v2_vf
            vis, viserr = self.v2, self.v2_err
            wave, base, vislabel = self.v2_wave, self.v2_base, "$V^2$"
        elif plot_vistype == "vis" or "fcorr":
            uf, vf = self.v_uf, self.v_vf
            vis, viserr = self.v, self.v_err
            wave, base = self.v_wave, self.v_base
            if not self.vis_in_fcorr:
                vislabel = "$V$"
            else:
                vislabel = r"$F_{corr}$ (Jy)"

        # plot uv coverage
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.subplots_adjust(right=0.8, bottom=0.2)
        cax = fig.add_axes([0.82, 0.20, 0.02, 0.68])
        ax.set_aspect("equal", adjustable="datalim")  # make plot axes have same scale
        ax.scatter(uf / 1e6, vf / 1e6, c=wave, s=1, cmap=constants.PLOT_CMAP)
        sc = ax.scatter(
            -uf / 1e6,
            -vf / 1e6,
            c=wave,
            s=1,
            cmap=constants.PLOT_CMAP,
        )
        clb = fig.colorbar(sc, cax=cax)
        clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_title("uv coverage")
        ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
        if fig_dir is not None:
            plt.savefig(
                os.path.join(fig_dir, f"uv_plane.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )

        # plot (squared) visibilities
        fig, ax = plt.subplots(1, 1, figsize=data_figsize)
        sc = ax.scatter(base, vis, c=wave, s=2, cmap=constants.PLOT_CMAP)
        ax.errorbar(
            base,
            vis,
            viserr,
            ecolor="grey",
            marker="",
            capsize=0,
            zorder=0,
            markersize=2,
            ls="",
            alpha=0.5,
            elinewidth=0.5,
        )
        clb = fig.colorbar(sc, pad=0.01, aspect=40)
        clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)
        ax.set_ylabel(vislabel)
        ax.set_title("Visibilities")

        if log_plotv:
            ax.set_ylim(0.5 * np.min(vis), 1.1 * np.max(vis))
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 1.1 * np.max(vis))

        ax.set_xlim(0, np.max(base) * 1.05)
        ax.axhline(y=1, c="k", ls="--", lw=1, zorder=0)
        ax.set_xlabel(r"$B$ ($\mathrm{M \lambda}$)")
        ax.set_ylabel(vislabel)
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(
                os.path.join(fig_dir, f"visibilities.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )

        # plot phi_closure
        fig, ax = plt.subplots(1, 1, figsize=data_figsize)
        sc = ax.scatter(
            self.t3_bmax,
            self.t3_phi,
            c=self.t3_wave,
            s=2,
            cmap=constants.PLOT_CMAP,
        )
        ax.errorbar(
            self.t3_bmax,
            self.t3_phi,
            self.t3_phierr,
            ecolor="grey",
            marker="",
            capsize=0,
            zorder=0,
            markersize=2,
            ls="",
            alpha=0.5,
            elinewidth=0.5,
        )
        clb = fig.colorbar(sc, pad=0.01, aspect=40)
        clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

        ax.set_ylabel(r"$\phi_{CP}$ ($^\circ$)")
        ax.set_title("Closure Phases")
        ax.set_ylim(np.min(self.t3_phi - self.t3_phierr), np.max(self.t3_phi + self.t3_phierr))
        ax.set_xlim(0, np.max(self.t3_bmax) * 1.05)
        ax.axhline(y=0, c="k", ls="--", lw=1, zorder=0)
        ax.set_xlabel(r"$B_{max}$ ($\mathrm{M \lambda}$)")
        plt.tight_layout()
        if fig_dir is not None:
            plt.savefig(
                os.path.join(fig_dir, f"closure_phases.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()

        return None


def read_oi_container_from_oifits(
    data_dir: str,
    data_file: str,
    wave_lims: tuple[float, float] | None = None,
    v2lim: float = None,
    fcorr: bool = False,
) -> OIContainer:
    """
    Retrieve data from (multiple) OIFITS files and return in an OIContainer class instance.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the files are stored.
    data_file : str
        Data filename, including wildcards if needed to read in multiple files at once.
    wave_lims : tuple[float, float], optional
        The lower and upper wavelength limits in micron used when reading in data.
    v2lim : float, optional
        Upper limit on the squared visibility used when reading in data.
    fcorr : bool, optional
        Set to True if visibility data is to be interpreted as correlated fluxes in Jy units.

    Returns
    -------
    OIContainer
        OIContainer with the observables from the OIFITS file.
    """
    # if condition because * constants.MICRON2M fails if wavelimits None
    if wave_lims is not None:
        oidata = select_data_oifits.SelectData(
            data_dir=data_dir,
            data_file=data_file,
            wave_1=wave_lims[0] * constants.MICRON2M,
            wave_2=wave_lims[1] * constants.MICRON2M,
            lim_V2=v2lim,
        )
    else:
        oidata = select_data_oifits.SelectData(data_dir=data_dir, data_file=data_file, lim_V2=v2lim)

    # dictionary to construct OIContainer instance
    dictionary = {}

    # arrays to store all necessary variables in a 1d array
    vufdat, vvfdat, vwavedat, vdat, verr = [], [], [], [], []  # for visibility tables
    v2ufdat, v2vfdat, v2wavedat, v2dat, v2err = (
        [],
        [],
        [],
        [],
        [],
    )  # for squared visibility tables
    uf1dat, vf1dat, uf2dat, vf2dat, t3wavedat, t3phidat, t3phierr = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )  # for closure phase
    # tables

    # get V data
    if oidata.vis != []:
        for vistable in oidata.vis:
            vufdat.extend(np.ravel(vistable.uf))  # unravel the arrays to make them 1d
            vvfdat.extend(np.ravel(vistable.vf))
            vwavedat.extend(np.ravel(vistable.effwave))
            vdat.extend(np.ravel(vistable.visamp))
            verr.extend(np.ravel(vistable.visamperr))
    # get V2 data
    if oidata.vis2 != []:
        for vis2table in oidata.vis2:
            v2ufdat.extend(np.ravel(vis2table.uf))  # unravel the arrays to make them 1d
            v2vfdat.extend(np.ravel(vis2table.vf))
            v2wavedat.extend(np.ravel(vis2table.effwave))
            v2dat.extend(np.ravel(vis2table.vis2data))
            v2err.extend(np.ravel(vis2table.vis2err))
    # get phi_closure data
    if oidata.t3 != []:
        for t3table in oidata.t3:
            uf1dat.extend(t3table.uf1)
            vf1dat.extend(t3table.vf1)
            uf2dat.extend(np.ravel(t3table.uf2))
            vf2dat.extend(np.ravel(t3table.vf2))
            t3wavedat.extend(np.ravel(t3table.effwave))
            t3phidat.extend(np.ravel(t3table.t3phi))
            t3phierr.extend(np.ravel(t3table.t3phierr))

    vufdat = np.array(vufdat)  # transfer into numpy arrays
    vvfdat = np.array(vvfdat)
    vwavedat = np.array(vwavedat)
    vdat = np.array(vdat)
    verr = np.array(verr)
    v_base = np.sqrt(vufdat**2 + vvfdat**2) / 1e6  # uv baseline length in MegaLambda

    v2ufdat = np.array(v2ufdat)  # transfer into numpy arrays
    v2vfdat = np.array(v2vfdat)
    v2wavedat = np.array(v2wavedat)
    v2dat = np.array(v2dat)
    v2err = np.array(v2err)
    v2_base = np.sqrt(v2ufdat**2 + v2vfdat**2) / 1e6  # uv baseline length in MegaLambda

    uf1dat = np.array(uf1dat)
    vf1dat = np.array(vf1dat)
    uf2dat = np.array(uf2dat)
    vf2dat = np.array(vf2dat)
    t3wavedat = np.array(t3wavedat)
    t3phidat = np.array(t3phidat)
    t3phierr = np.array(t3phierr)

    uf3dat = uf1dat + uf2dat  # 3d baseline (frequency) and max baseline of closure triangle (in MegaLambda)
    vf3dat = vf1dat + vf2dat
    t3_bmax = (
        np.maximum(
            np.sqrt(uf3dat**2 + vf3dat**2),
            np.maximum(np.sqrt(uf1dat**2 + vf1dat**2), np.sqrt(uf2dat**2 + vf2dat**2)),
        )
        / 1e6
    )

    # Filter 1D arrays further based on NaN values
    vdat, verr, vufdat, vvfdat, vwavedat, v_base = nan_filter_arrays(
        vdat, verr, vufdat, vvfdat, vwavedat, v_base, filter_ref=True
    )
    v2dat, v2err, v2ufdat, v2vfdat, v2wavedat, v2_base = nan_filter_arrays(
        v2dat, v2err, v2ufdat, v2vfdat, v2wavedat, v2_base, filter_ref=True
    )
    t3phidat, t3phierr, uf1dat, vf1dat, uf2dat, vf2dat, uf3dat, vf3dat, t3wavedat, t3_bmax = nan_filter_arrays(
        t3phidat, t3phierr, uf1dat, vf1dat, uf2dat, vf2dat, uf3dat, vf3dat, t3wavedat, t3_bmax, filter_ref=True
    )

    # NOTE: wavelengths are read in in units of meter, but OIContainer objects work with wavelengths in units micron,
    # so do not forget to convert wavelengths to unit micron before initializing the OIContainer

    # fill in data observables dictionary
    dictionary["v_uf"] = vufdat  # spatial freqs in 1/rad
    dictionary["v_vf"] = vvfdat
    dictionary["v_wave"] = vwavedat * constants.M2MICRON  # wavelengths in micron
    dictionary["v"] = vdat
    dictionary["v_err"] = verr  # baseline length in MegaLambda
    dictionary["v_base"] = v_base

    dictionary["v2_uf"] = v2ufdat  # spatial freqs in 1/rad
    dictionary["v2_vf"] = v2vfdat
    dictionary["v2_wave"] = v2wavedat * constants.M2MICRON  # wavelengths in micron
    dictionary["v2"] = v2dat
    dictionary["v2_err"] = v2err  # baseline length in MegaLambda
    dictionary["v2_base"] = v2_base

    dictionary["t3_uf1"] = uf1dat  # spatial freqs in 1/rad
    dictionary["t3_vf1"] = vf1dat
    dictionary["t3_uf2"] = uf2dat
    dictionary["t3_vf2"] = vf2dat
    dictionary["t3_uf3"] = uf3dat
    dictionary["t3_vf3"] = vf3dat
    dictionary["t3_wave"] = t3wavedat * constants.M2MICRON  # wavelengths in micron
    dictionary["t3phi"] = t3phidat  # closure phases in degrees
    dictionary["t3phi_err"] = t3phierr
    dictionary["t3_bmax"] = t3_bmax  # max baseline lengths in MegaLambda

    # Return an OIContainer object
    container = OIContainer(dictionary=dictionary, fcorr=fcorr)
    return container


# TODO: add option to add geometric component flux to the SED (say only in the wavelength range of the OI container?)
def oi_container_calc_image_fft_observables(
    container_data: OIContainer,
    img_ffts: list[image.Image],
    img_sed: sed.SED | None = None,
    geom_comps: list[geom_comp.GeomComp] | None = None,
    geom_comp_flux_fracs: list[float] | None = None,
    ref_wavelength: float | None = None,
    interp_method: str = "linear",
) -> OIContainer:
    """
    Loads in OI observables from an OIContainer, and calculates corresponding model image observables. The model images
    are passed along as a list of Image objects.

    If this list contains one object, the normalized visibilities will not be interpolated in the wavelength dimension,
    i.e. the emission morphology is 'monochromatic'. If the list contains multiple Image objects, interpolation in
    wavelength is performed. In this case the wavelength coverage of the Image objects needs to exceed that of the
    OIContainer!

    NOTE: This method expects that every Image object in the list has the same pixelscale and amount of pixels
    (in both x- and y-direction).

    Parameters
    ----------
    container_data : OIContainer
        OIContainer at whose spatial frequencies we calculate model observables.
    img_ffts : list of image.Image
        List of Image objects representing the RT model images at different wavelengths. If containing only one object,
        no interpolation of the normalized visibilities in the wavelength dimension will be performed.
    img_sed : sed.SED, optional
        Optional SED to be passed along defining the total flux wavelength dependence of the model Image(s). By default,
        this is None, and the total flux wavelength dependence will be taken from either the SpecDep property of the
        Image, in case img_ffts contains a single Image object, or from a linear interpolation between the total fluxes
        in case img_ffts contains multiple Image objects.
    geom_comps : list of geom_comp.GeomComp, optional
        Optional list of GeomComp objects, representing geometric components to be added to the complex visibility
        calculation.
    geom_comp_flux_fracs : list of float, optional
        Flux fractions of the geometric components. Should add up to less than one. The remainder from the difference
        with one will be the flux fraction attributed to the model image(s).
    ref_wavelength : float, optional
        Reference wavelength for the geometric component flux fractions in micron. In case image_ffts contains more
        than one model image. This wavelength must lie within the wavelength range spanned by the SED (if passed along),
        or by the model images.
    interp_method : str, optional
        Interpolation method used by scipy to perform interpolations. Can support 'linear', 'nearest', 'slinear', or
        'cubic'.

    Returns
    -------
    OIContainer
        OIContainer for model image observables.
    """
    # TODO: add functionality for including geometric components
    # TODO: check if closure phase calculations are correct, because they don't seem to be

    # check if geometric component flux fractions do not exceed 1
    if geom_comps is not None and sum(geom_comp_flux_fracs) >= 1.0:
        raise ValueError("The sum of geometric component flux fractions cannot exceed 1.")

    # retrieve interpolator for normalized complex visibilities from model image(s)
    vcomp_norm_img_interpolator = image.image_fft_comp_vis_interpolator(
        img_ffts, normalised=True, interp_method=interp_method
    )

    # wrapper function to calculate absolute complex visibilities and total fluxes (in Jansky)
    # for given uv frequencies and wavelengths from the model image(s) with the interpolators
    def image_fft_get_vcomp_abs_and_ftot(
        uf: np.ndarray, vf: np.ndarray, wavelengths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(img_ffts) == 1:
            img_fft = img_ffts[0]  # extract the single Image object
            vcomp_norm_img = vcomp_norm_img_interpolator((vf, uf))  # image normalized complex visibility

            # get total model flux
            if img_sed is None:  # in case no SED to describe image wavelength dependence
                frequencies = constants.SPEED_OF_LIGHT / (wavelengths * constants.MICRON2M)  # freqs to calc at

                # calculate model image's total F_nu flux in Jansky
                freq_img = constants.SPEED_OF_LIGHT / (img_fft.wavelength * constants.MICRON2M)  # image freq
                ftot_img = img_fft.sp_dep.flux_from_ref(
                    x=frequencies,
                    x_ref=freq_img,
                    ref_flux=img_fft.ftot,
                    flux_form="fnu",
                )
            else:  # in case a model SED is passed along
                frequencies = constants.SPEED_OF_LIGHT / (wavelengths * constants.MICRON2M)  # freqs to calc at
                # calculate total F_nu flux in Jansky form the SED
                ftot_img = img_sed.get_flux(x=frequencies, flux_form="fnu")

        else:  # case for mutliple model images
            vcomp_norm_img = vcomp_norm_img_interpolator((wavelengths, vf, uf))  # image normalized complex visibility

            # get total model flux
            if img_sed is None:
                ftot_img_interpolator = image.image_fft_ftot_interpolator(
                    img_ffts=img_ffts, interp_method=interp_method
                )
                ftot_img = ftot_img_interpolator(wavelengths)
            else:
                frequencies = constants.SPEED_OF_LIGHT / (wavelengths * constants.MICRON2M)  # freqs to calc at
                # calculate total F_nu flux in Jansky form the SED
                ftot_img = img_sed.get_flux(x=frequencies, flux_form="fnu")

        vcomp_abs_img = vcomp_norm_img * ftot_img  # absolute complex visibility for model image

        return (vcomp_abs_img, ftot_img)

    # wrapper function to add the effect of geometric components on absolute complex visibilities on top of that
    # of the model image(s)
    def vcomp_abs_add_geom_comps(
        uf: np.ndarray, vf: np.ndarray, wavelengths: np.ndarray, vcomp_abs: np.ndarray, ftot: np.ndarray
    ) -> None:
        frequencies = constants.SPEED_OF_LIGHT / (wavelengths * constants.MICRON2M)  # frequencies to calculate at
        ref_frequency = constants.SPEED_OF_LIGHT / (ref_wavelength * constants.MICRON2M)  # ref frequency in Hz

        # calculate total flux of model image at reference wavelength/frequency
        if img_sed is not None:
            # case where an SED is passed
            ftot_img_ref = img_sed.get_flux(x=ref_frequency, flux_form="fnu")  # total model image(s) flux at ref freq
        elif len(img_ffts) == 1:
            # case for a single model image
            freq_img = constants.SPEED_OF_LIGHT / (img_ffts[0].wavelength * constants.MICRON2M)  # image freq
            ftot_img_ref = img_ffts[0].sp_dep.flux_from_ref(
                x=ref_frequency,
                x_ref=freq_img,
                ref_flux=img_ffts[0].ftot,
                flux_form="fnu",
            )
        else:
            # case for multiple model images
            ftot_img_interpolator = image.image_fft_ftot_interpolator(img_ffts=img_ffts, interp_method=interp_method)
            ftot_img_ref = ftot_img_interpolator(ref_wavelength)

        # loop over geometric components to add their effects to the total model complex visibility and flux
        for index, component in enumerate(geom_comps):
            # total flux of component at reference wavelength/frequency
            component_ftot_ref = (geom_comp_flux_fracs[index] * ftot_img_ref) / (1 - sum(geom_comp_flux_fracs))
            # component total flux at argument wavelengths
            component_ftot = component.sp_dep.flux_from_ref(
                x=frequencies,
                x_ref=ref_frequency,
                ref_flux=component_ftot_ref,
                flux_form="fnu",
            )
            # component absolute complex visibility at argument wavelengths
            component_vcomp_abs = component.calc_vis(
                uf, vf, wavelength=wavelengths, ref_wavelength=ref_wavelength, ref_corr_flux=component_ftot_ref
            )

            # add contribution to the total flux
            ftot += component_ftot
            vcomp_abs += component_vcomp_abs

        return

    # calculate normalized visibilities
    # =================================

    # get model image(s) complex visibility and total flux
    vcomp_abs, ftot = image_fft_get_vcomp_abs_and_ftot(container_data.v_uf, container_data.v_vf, container_data.v_wave)

    # add effect of geometric components
    if geom_comps is not None and len(geom_comps) > 0:
        vcomp_abs_add_geom_comps(container_data.v_uf, container_data.v_vf, container_data.v_wave, vcomp_abs, ftot)

    # final calculation
    if not container_data.vis_in_fcorr:
        vmod = abs(vcomp_abs / ftot)  # normalized
    else:
        vmod = abs(vcomp_abs)  # in correlated flux

    # calculate squared visibilities
    # ==============================

    # get model image(s) complex visibility and total flux
    vcomp_abs, ftot = image_fft_get_vcomp_abs_and_ftot(
        container_data.v2_uf, container_data.v2_vf, container_data.v2_wave
    )

    # add effect of geometric components
    if geom_comps is not None and len(geom_comps) > 0:
        vcomp_abs_add_geom_comps(container_data.v2_uf, container_data.v2_vf, container_data.v2_wave, vcomp_abs, ftot)

    # final calculation
    v2mod = abs(vcomp_abs / ftot) ** 2

    # calculate closure phases
    # ========================

    # get model image(s) complex visibility and total flux
    vcomp1_abs, ftot1 = image_fft_get_vcomp_abs_and_ftot(
        container_data.t3_uf1, container_data.t3_vf1, container_data.t3_wave
    )
    vcomp2_abs, ftot2 = image_fft_get_vcomp_abs_and_ftot(
        container_data.t3_uf2, container_data.t3_vf2, container_data.t3_wave
    )
    vcomp3_abs, ftot3 = image_fft_get_vcomp_abs_and_ftot(
        container_data.t3_uf3, container_data.t3_vf3, container_data.t3_wave
    )

    # add effect of geometric components
    if geom_comps is not None and len(geom_comps) > 0:
        vcomp_abs_add_geom_comps(
            container_data.t3_uf1, container_data.t3_vf1, container_data.t3_wave, vcomp1_abs, ftot1
        )
        vcomp_abs_add_geom_comps(
            container_data.t3_uf2, container_data.t3_vf2, container_data.t3_wave, vcomp2_abs, ftot2
        )
        vcomp_abs_add_geom_comps(
            container_data.t3_uf3, container_data.t3_vf3, container_data.t3_wave, vcomp2_abs, ftot3
        )

    # final calculation

    # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA.
    # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
    # so we take the complex conjugate there.
    t3phimod = np.angle(vcomp1_abs * vcomp2_abs * np.conjugate(vcomp3_abs), deg=True)

    # initialize dictionary to construct OIContainer for model observables
    observables_mod = {
        "v_uf": container_data.v_uf,
        "v_vf": container_data.v_vf,
        "v_wave": container_data.v_wave,
        "v": vmod,
        "v_err": np.zeros_like(container_data.v_err),
        "v_base": container_data.v_base,
        "v2_uf": container_data.v2_uf,
        "v2_vf": container_data.v2_vf,
        "v2_wave": container_data.v2_wave,
        "v2": v2mod,
        "v2_err": np.zeros_like(container_data.v2_err),
        "v2_base": container_data.v2_base,
        "t3_uf1": container_data.t3_uf1,
        "t3_vf1": container_data.t3_vf1,
        "t3_uf2": container_data.t3_uf2,
        "t3_vf2": container_data.t3_vf2,
        "t3_uf3": container_data.t3_uf3,
        "t3_vf3": container_data.t3_vf3,
        "t3_wave": container_data.t3_wave,
        "t3phi": t3phimod,
        "t3phi_err": np.zeros_like(container_data.t3_phierr),
        "t3_bmax": container_data.t3_bmax,
    }

    # return an OIContainer object
    container_mod = OIContainer(dictionary=observables_mod, fcorr=container_data.vis_in_fcorr)
    return container_mod


def oi_container_plot_data_vs_model(
    container_data: OIContainer,
    container_mod: OIContainer,
    fig_dir: str = None,
    log_plotv: bool = False,
    plot_vistype: Literal["vis2", "vis", "fcorr"] = "vis2",
    show_plots: bool = True,
) -> None:
    """
    Plots the data against the model OI observables. Currently, plots uv coverage, a (squared) visibility curve and
    closure phases. Note that this function shares a name with a similar function in the sed module. Take care with
    your namespace if you use both functions in the same script.

    Parameters
    ----------
    container_data : OIContainer
        Container with data observables.
    container_mod : OIContainer
        Container with model observables.
    fig_dir : str, optional
        Directory to store plots in.
    log_plotv : bool, optional
        Set to True for a logarithmic y-scale in the (squared) visibility plot.
    plot_vistype : {'vis2', 'vis', 'fcorr'}, optional
        Sets the type of visibility to be plotted. 'vis2' for squared visibilities, 'vis' for visibilities or 'fcorr'
        for correlated flux in Jy.
    show_plots : bool, optional
        Set to False if you do not want the plots to be shown during your python instance. Note that if True, this
        freezes further code execution until the plot windows are closed.

    Returns
    -------
    None
    """
    valid_vistypes = ["vis2", "vis", "fcorr"]
    if plot_vistype not in valid_vistypes:
        raise ValueError(f"Warning: Invalid plot_vistype '{plot_vistype}'. Valid options are: {valid_vistypes}.")

    # create plotting directory if it doesn't exist yet
    if fig_dir is not None:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == "vis2":
        ufdata = container_data.v2_uf
        vfdata = container_data.v2_vf
        vismod = container_mod.v2
        visdata = container_data.v2
        viserrdata = container_data.v2_err
        wavedata = container_data.v2_wave
        basedata = container_data.v2_base
        vislabel = "$V^2$"
    elif plot_vistype == "vis":
        ufdata = container_data.v_uf
        vfdata = container_data.v_vf
        vismod = container_mod.v
        wavedata = container_data.v_wave
        visdata = container_data.v
        viserrdata = container_data.v_err
        basedata = container_data.v_base
        if (not container_data.vis_in_fcorr) and (not container_mod.vis_in_fcorr):
            vislabel = "$V$"
        elif container_data.vis_in_fcorr and container_mod.vis_in_fcorr:
            vislabel = r"$F_{corr}$ (Jy)"
        else:
            raise Exception("container_data and container_mod do not have the same value for vis_in_fcorr")
        return

    # plot uv coverage
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    ax.set_aspect("equal", adjustable="datalim")  # make plot axes have the same scale
    ax.scatter(
        ufdata / 1e6,
        vfdata / 1e6,
        c=wavedata,
        s=1,
        cmap=constants.PLOT_CMAP,
    )
    sc = ax.scatter(
        -ufdata / 1e6,
        -vfdata / 1e6,
        c=wavedata,
        s=1,
        cmap=constants.PLOT_CMAP,
    )
    clb = fig.colorbar(sc, cax=cax)
    clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

    ax.set_xlim(ax.get_xlim()[::-1])  # switch x-axis direction
    ax.set_title("uv coverage")
    ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
    ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"uv_plane.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )

    # plot (squared) visibilities
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(
        basedata,
        visdata,
        viserrdata,
        label="data",
        mec="royalblue",
        marker="o",
        capsize=0,
        zorder=0,
        markersize=2,
        ls="",
        alpha=0.8,
        elinewidth=0.5,
    )
    ax[0].scatter(
        basedata,
        vismod,
        label="model",
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )
    ax[1].scatter(
        basedata,
        (vismod - visdata) / viserrdata,
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )

    ax[0].set_ylabel(vislabel)
    ax[0].legend()
    ax[0].set_title("Visibilities")
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    if log_plotv:
        ax[0].set_ylim(0.5 * np.min(visdata), 1.1 * np.max(np.maximum(visdata, vismod)))
        ax[0].set_yscale("log")
    else:
        ax[0].set_ylim(0, 1.1 * np.max(np.maximum(visdata, vismod)))

    ax[1].set_xlim(0, np.max(basedata) * 1.05)
    ax[1].axhline(y=0, c="k", ls="--", lw=1, zorder=0)
    ax[1].set_xlabel(r"$B$ ($\mathrm{M \lambda}$)")
    ax[1].set_ylabel(r"error $(\sigma)$")
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"visibilities.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )

    # plot phi_closure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(
        container_data.t3_bmax,
        container_data.t3_phi,
        container_data.t3_phierr,
        label="data",
        mec="royalblue",
        marker="o",
        capsize=0,
        zorder=0,
        markersize=2,
        ls="",
        alpha=0.8,
        elinewidth=0.5,
    )
    ax[0].scatter(
        container_data.t3_bmax,
        container_mod.t3_phi,
        label="model",
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )
    ax[1].scatter(
        container_data.t3_bmax,
        (container_mod.t3_phi - container_data.t3_phi) / container_data.t3_phierr,
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )

    ax[0].set_ylabel(r"$\phi_{CP}$ ($^\circ$)")
    ax[0].legend()
    ax[0].set_title("Closure Phases")
    ax[0].tick_params(axis="x", direction="in", pad=-15)
    ax[0].set_ylim(
        min(
            np.min(container_data.t3_phi - container_data.t3_phierr),
            np.min(container_mod.t3_phi),
        ),
        max(
            np.max(container_data.t3_phi + container_data.t3_phierr),
            np.max(container_mod.t3_phi),
        ),
    )

    ax[1].set_xlim(0, np.max(container_data.t3_bmax) * 1.05)
    ax[1].axhline(y=0, c="k", ls="--", lw=1, zorder=0)
    ax[1].set_xlabel(r"$B_{max}$ ($\mathrm{M \lambda}$)")
    ax[1].set_ylabel(r"error $(\sigma_{\phi_{CP}})$")
    if fig_dir is not None:
        plt.savefig(
            os.path.join(fig_dir, f"closure_phases.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()

    return


if __name__ == "__main__":
    from distroi.auxiliary.beam import oi_container_calc_gaussian_beam

    # object_id = "IW_Car"
    # epoch_id = "img_ep_nov2019-mar2020"
    # data_dir, data_file = (
    #     f"/home/toond/Documents/phd/data/{object_id}/inspiring/PIONIER/{epoch_id}/",
    #     "*.fits",
    # )

    object_id = "EN_TrA"
    epoch_id = "img_ep_jan2021-mar2021"
    data_dir, data_file = (
        f"/home/toond/Documents/phd/data/{object_id}/inspiring/PIONIER/{epoch_id}/",
        "*.fits",
    )

    # object_id = "IRAS15469-5311"
    # epoch_id = "img_ep_jan2021-mar2021"
    # data_dir, data_file = (
    #     f"/home/toond/Documents/phd/data/{object_id}/inspiring/PIONIER/{epoch_id}/",
    #     "*.fits",
    # )

    # object_id = "IRAS08544-4431"
    # data_dir, data_file = (
    #     f"/home/toond/Documents/phd/data/IRAS08544-4431/imaging_campaign_hillen_et_al2016/PIONIER",
    #     "*.fits",
    # )

    container_data = read_oi_container_from_oifits(data_dir, data_file)
    fig_dir = f"{data_dir}/figures/"
    container_data.plot_data(fig_dir=fig_dir)

    u = container_data.v2_uf
    v = container_data.v2_vf
    max_uv_dist = np.max(np.sqrt(u**2 + v**2))  # max distance in 1/rad from origin, sets pixelscale for image space
    pix_res = (0.5 / max_uv_dist) * constants.RAD2MAS  # smallest resolution element (at Nyquist sampling)
    print(f"resolution element is: {pix_res:.4E}")

    beam = oi_container_calc_gaussian_beam(
        container_data,
        vistype="vis2",
        make_plots=True,
        show_plots=True,
        fig_dir=fig_dir,
        num_res=3,
        pix_per_res=32,
    )
