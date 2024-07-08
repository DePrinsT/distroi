"""
Contains a class to store optical inteferometric (OI) observables and additional functions to take radiative
transfer (RT) model images and convert them to interferometric observables at the spatial frequencies of data
stored in the OIFITS format. Currently supports the following combinations of obsevrables: Squared visibilities -
Closure phases ; Visibilities - Closure phases ; Correlated fluxes (formaly stored as visibilities) - Closure phases.
"""

from distroi import constants
from distroi import image_fft
from distroi.auxiliary import SelectData

import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class OIContainer:
    """
    Class to contain optical interferometry observables, where each observable is stored in the form of raveled 1D
    numpy arrays. Based on the Data class in ReadOIFITS.py, where the observables are stored per OIFITS table,
    but the raveled form makes calculations easier and less prone to error. Currenlty supports visibilities (both in
    normalized and correlated flux form), squared visibilities and closure phases. In this format this class and the
    methods below can be expanded to accomodate for different kinds of observables (i.e. addition of differential
    visibilities). Can contain observables either related to astronomical observations or model RT images.

    :param dict dictionary: Dictionary containing keys and values representing all instance variables described below
        (excluding fcorr).
    :param bool fcorr: Set to True if the visibilities are to be stored as correlated fluxes in Jy.
    :ivar bool vis_in_fcorr: Whether visibilities are stored in correlated flux or not.
    :ivar np.ndarray vuf: u-axis spatial freqs in 1/rad for visibility data.
    :ivar np.ndarray vvf: v-axis spatial freqs in 1/rad for visibility data.
    :ivar np.ndarray vwave: Wavelengths in meter for visibility data.
    :ivar np.ndarray v: Visibilities, either normalized visibilities or correlated flux in Janksy.
    :ivar np.ndarray verr: Error on visibilities.
    :ivar np.ndarray vbase: Baseline length in MegaLambda for visibility data.
    :ivar np.ndarray v2uf: u-axis spatial freqs in 1/rad for squared visibility data.
    :ivar np.ndarray v2vf: v-axis spatial freqs in 1/rad for squared visibility data.
    :ivar np.ndarray v2wave: Wavelengths in meter for squared visibility data.
    :ivar np.ndarray v2: Squared visibilities.
    :ivar np.ndarray v2err: Error on squared visibilities.
    :ivar np.ndarray v2base: Baseline length in MegaLambda for squared visibility data.
    :ivar np.ndarray t3uf1: u-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    :ivar np.ndarray t3vf1: v-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    :ivar np.ndarray t3uf2: u-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    :ivar np.ndarray t3vf2: v-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    :ivar np.ndarray t3uf3: u-axis spatial freqs in 1/rad for the 3d projected baseline along the closure triangle.
    :ivar np.ndarray t3vf3: v-axis spatial freqs in 1/rad for the 3d projected baseline along the closure triangle.
    :ivar np.ndarray t3wave: Wavelengths in meter for closure phase data.
    :ivar np.ndarray t3phi: Closure phases in degrees.
    :ivar np.ndarray t3phierr: Error on closure phases.
    :ivar np.ndarray t3bmax: Maximum baseline length along the closure triangle in units of MegaLambda.
    """

    def __init__(self, dictionary: dict[str, np.ndarray], fcorr: bool = False):
        """
        Constructor method. See class docstring for information on instance properties.
        """
        self.vis_in_fcorr = fcorr  # set if visibilities are in correlated flux

        # Properties for visibility (v) data
        self.vuf = None  # u-axis spatial freqs in 1/rad
        self.vvf = None  # v-axis spatial freqs in 1/rad
        self.vwave = None  # wavelengths in meter
        self.v = None  # visibilities (either normalized visibilities or correlated flux in Janksy)
        self.verr = None  # error on visibilities
        self.vbase = None  # baseline length in MegaLambda (i.e. 1e6*vwave)

        # Properties for squared visibility (v2) data
        self.v2uf = None  # u-axis spatial freqs in 1/rad
        self.v2vf = None  # v-axis spatial freqs in 1/rad
        self.v2wave = None  # wavelengths in meter
        self.v2 = None  # squared visibilities
        self.v2err = None  # errors on squared visibilities
        self.v2base = None  # baseline length in MegaLambda (i.e. 1e6*vwave)

        # Properties for closure phase (t3) data
        self.t3uf1 = None  # u-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3vf1 = None  # v-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3uf2 = None
        self.t3vf2 = None
        self.t3uf3 = None
        self.t3vf3 = None
        self.t3wave = None  # wavelenghts in meter
        self.t3phi = None  # closure phases in degrees
        self.t3phierr = None  # errors on closure phases
        self.t3bmax = None  # maximum baseline lengths along the closure triangles in units of MegaLambda

        # Read in from the dictionary
        if dictionary is not None:
            self.vuf = dictionary["vuf"]
            self.vvf = dictionary["vvf"]
            self.vwave = dictionary["vwave"]
            self.v = dictionary["v"]
            self.verr = dictionary["verr"]
            self.vbase = dictionary["vbase"]

            self.v2uf = dictionary["v2uf"]
            self.v2vf = dictionary["v2vf"]
            self.v2wave = dictionary["v2wave"]
            self.v2 = dictionary["v2"]
            self.v2err = dictionary["v2err"]
            self.v2base = dictionary["v2base"]

            self.t3uf1 = dictionary["t3uf1"]
            self.t3vf1 = dictionary["t3vf1"]
            self.t3uf2 = dictionary["t3uf2"]
            self.t3vf2 = dictionary["t3vf2"]
            self.t3uf3 = dictionary["t3uf3"]
            self.t3vf3 = dictionary["t3vf3"]
            self.t3wave = dictionary["t3wave"]
            self.t3phi = dictionary["t3phi"]
            self.t3phierr = dictionary["t3phierr"]
            self.t3bmax = dictionary["t3bmax"]

    def plot_data(
        self,
        fig_dir: str = None,
        log_plotv: bool = False,
        plot_vistype: str = "vis2",
        show_plots: bool = True,
    ) -> None:
        """
        Plots the data included in the OIContainer instance. Currently, plots uv coverage, a (squared) visibility curve and
        closure phases.

        :param str fig_dir: Directory to store plots in.
        :param bool log_plotv: Set to True for a logarithmic y-scale in the (squared) visibility plot.
        :param str plot_vistype: Sets the type of visibility to be plotted. 'vis2' for squared visibilities or 'vis'
            for visibilities (either normalized or correlated flux in Jy, as implied by the OIContainer objects).
        :param bool show_plots: Set to False if you do not want the plots to be shown during your python instance.
            Note that if True, this freazes further code execution until the plot windows are closed.
        :rtype: None
        """
        # create plotting directory if it doesn't exist yet
        if fig_dir is not None:
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)

        # set spatial frequencies, visibilities and plotting label based on specified option
        if plot_vistype == "vis2":
            uf, vf = self.v2uf, self.v2vf
            vis, viserr = self.v2, self.v2err
            wave, base, vislabel = self.v2wave, self.v2base, "$V^2$"
        elif plot_vistype == "vis":
            uf, vf = self.vuf, self.vvf
            vis, viserr = self.v, self.verr
            wave, base = self.vwave, self.vbase
            if not self.vis_in_fcorr:
                vislabel = "$V$"
            else:
                vislabel = r"$F_{corr}$ (Jy)"
        else:
            print("parameter plot_vistype is not recognized, will return None!")
            return

        # plot uv coverage
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
        ax.set_aspect("equal", adjustable="datalim")  # make plot axes have same scale
        ax.scatter(
            uf / 1e6, vf / 1e6, c=wave * constants.M2MICRON, s=1, cmap="gist_rainbow_r"
        )
        sc = ax.scatter(
            -uf / 1e6,
            -vf / 1e6,
            c=wave * constants.M2MICRON,
            s=1,
            cmap="gist_rainbow_r",
        )
        clb = fig.colorbar(sc, cax=cax)
        clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_title("uv coverage")
        ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/uv_plane.png", dpi=300, bbox_inches="tight")

        # plot (squared) visibilities
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sc = ax.scatter(
            base, vis, c=wave * constants.M2MICRON, s=2, cmap="gist_rainbow_r"
        )
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
        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/visibilities.png", dpi=300, bbox_inches="tight")

        # plot phi_closure
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sc = ax.scatter(
            self.t3bmax,
            self.t3phi,
            c=self.t3wave * constants.M2MICRON,
            s=2,
            cmap="gist_rainbow_r",
        )
        ax.errorbar(
            self.t3bmax,
            self.t3phi,
            self.t3phierr,
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
        ax.set_ylim(
            np.min(self.t3phi - self.t3phierr), np.max(self.t3phi + self.t3phierr)
        )
        ax.set_xlim(0, np.max(self.t3bmax) * 1.05)
        ax.axhline(y=0, c="k", ls="--", lw=1, zorder=0)
        ax.set_xlabel(r"$B_{max}$ ($\mathrm{M \lambda}$)")
        ax.set_ylabel(r"error $(\sigma_{\phi_{CP}})$")
        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/closure_phases.png", dpi=300, bbox_inches="tight")
        if show_plots:
            plt.show()

        return None


def read_oicontainer_oifits(
    data_dir: str,
    data_file: str,
    wave_lims: tuple[float, float] = None,
    v2lim: float = None,
    fcorr: bool = False,
) -> OIContainer:
    """
    Retrieve data from (multiple) OIFITS files and return in an OIConatiner class instance.

    :param str data_dir: Path to the directory where the files are stored.
    :param str data_file: Data filename, including wildcards if needed to read in multiple files at once.
    :param tuple(float) wave_lims: The lower and upper wavelength limits in micron used when reading in data.
    :param float v2lim: Upper limit on the squared visibility used when reading in data.
    :param bool fcorr: Set to True if visibility data is to be interpreted as correlated fluxes in Jy units.
    :return container: OIContainer with the observables from the OIFITS file.
    :rtype: OIContainer
    """
    # if condition because * constants.MICRON2M fails if wavelimits None
    if wave_lims is not None:
        oidata = SelectData.SelectData(
            data_dir=data_dir,
            data_file=data_file,
            wave_1=wave_lims[0] * constants.MICRON2M,
            wave_2=wave_lims[1] * constants.MICRON2M,
            lim_V2=v2lim,
        )
    else:
        oidata = SelectData.SelectData(
            data_dir=data_dir, data_file=data_file, lim_V2=v2lim
        )

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
    vbase = np.sqrt(vufdat**2 + vvfdat**2) / 1e6  # uv baseline length in MegaLambda

    v2ufdat = np.array(v2ufdat)  # transfer into numpy arrays
    v2vfdat = np.array(v2vfdat)
    v2wavedat = np.array(v2wavedat)
    v2dat = np.array(v2dat)
    v2err = np.array(v2err)
    v2base = np.sqrt(v2ufdat**2 + v2vfdat**2) / 1e6  # uv baseline length in MegaLambda

    uf1dat = np.array(uf1dat)
    vf1dat = np.array(vf1dat)
    uf2dat = np.array(uf2dat)
    vf2dat = np.array(vf2dat)
    t3wavedat = np.array(t3wavedat)
    t3phidat = np.array(t3phidat)
    t3phierr = np.array(t3phierr)

    uf3dat = (
        uf1dat + uf2dat
    )  # 3d baseline (frequency) and max baseline of closure triangle (in MegaLambda)
    vf3dat = vf1dat + vf2dat
    t3bmax = (
        np.maximum(
            np.sqrt(uf3dat**2 + vf3dat**2),
            np.maximum(np.sqrt(uf1dat**2 + vf1dat**2), np.sqrt(uf2dat**2 + vf2dat**2)),
        )
        / 1e6
    )

    # fill in data observables dictionary
    dictionary["vuf"] = vufdat  # spatial freqs in 1/rad
    dictionary["vvf"] = vvfdat
    dictionary["vwave"] = vwavedat  # wavelengths in meter
    dictionary["v"] = vdat
    dictionary["verr"] = verr  # baseline length in MegaLambda
    dictionary["vbase"] = vbase

    dictionary["v2uf"] = v2ufdat  # spatial freqs in 1/rad
    dictionary["v2vf"] = v2vfdat
    dictionary["v2wave"] = v2wavedat  # wavelengths in meter
    dictionary["v2"] = v2dat
    dictionary["v2err"] = v2err  # baseline length in MegaLambda
    dictionary["v2base"] = v2base

    dictionary["t3uf1"] = uf1dat  # spatial freqs in 1/rad
    dictionary["t3vf1"] = vf1dat
    dictionary["t3uf2"] = uf2dat
    dictionary["t3vf2"] = vf2dat
    dictionary["t3uf3"] = uf3dat
    dictionary["t3vf3"] = vf3dat
    dictionary["t3wave"] = t3wavedat  # wavelengths in meter
    dictionary["t3phi"] = t3phidat  # closure phases in degrees
    dictionary["t3phierr"] = t3phierr
    dictionary["t3bmax"] = t3bmax  # max baseline lengths in MegaLambda

    # Return an OIContainer object
    container = OIContainer(dictionary=dictionary, fcorr=fcorr)
    return container


def calc_mod_observables(
    container_data: OIContainer, img_fft_list: list[image_fft.ImageFFT]
) -> OIContainer:
    """
    Loads in OI observables from an OIContainer, typically containing observational data, and calculates model image
    observables at the same uv coverage. The model images are passed along as a list of ImageFFT objects. If the
    length of this list is 1, no interpolation in the wavelength dimension is performed, i.e. the emission model is
    'monochromatic'. If the list contains multiple ImageFFT objects, interpolation in wavelength is performed. In
    this case the wavelength coverage of the ImageFFT objects needs to exceed that of the OIContainer! Expects that
    every ImageFFT object in the list has the same pixelscale and amount of pixels (in both x- and y-direction).

    :param OIContainer container_data: OIContainer at whose spatial frequencies we calculate model observables.
    :param img_fft_list: List of ImageFFT objects representing the RT model images. If of length one, no interpolation
        in the wavelength dimension will be performed.
    :return container_mod: OIContainer for model image observables.
    :rtype: OIContainer
    """

    if len(img_fft_list) == 1:  # monochromatic case for a single image
        # create interpolator for the normalized complex FFT
        interp_norm = mod_comp_vis_interpolator(img_fft_list)

        # Calculate visibilities (requires separate interpolator for correlated fluxes if needed).
        if container_data.vis_in_fcorr:
            interp_fcorr = mod_comp_vis_interpolator(img_fft_list, fcorr=True)
            vmod = abs(interp_fcorr((container_data.vvf, container_data.vuf)))
        else:
            vmod = abs(interp_norm((container_data.vvf, container_data.vuf)))

        # Calculate squared visibilities.
        v2mod = abs(interp_norm((container_data.v2vf, container_data.v2uf))) ** 2
        # Calculate closure phases. We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,
        # v3) = AC, not CA This causes a minus sign shift for 3rd baseline when calculating closure phase (for real
        # images), so we take the complex conjugate there.
        t3phimod = np.angle(
            interp_norm((container_data.t3vf1, container_data.t3uf1))
            * interp_norm((container_data.t3vf2, container_data.t3uf2))
            * np.conjugate(interp_norm((container_data.t3vf3, container_data.t3uf3))),
            deg=True,
        )

    else:  # case for multiple images simultaneously
        # create interpolator for the normalized complex FFT
        interp_norm = mod_comp_vis_interpolator(img_fft_list)

        # Calculate visibilities (requires separate interpolator for correlated fluxes if needed).
        if container_data.vis_in_fcorr:
            interp_fcorr = mod_comp_vis_interpolator(img_fft_list, fcorr=True)
            vmod = abs(
                interp_fcorr(
                    (container_data.vwave, container_data.vvf, container_data.vuf)
                )
            )
        else:
            vmod = abs(
                interp_norm(
                    (container_data.vwave, container_data.vvf, container_data.vuf)
                )
            )

        # Calculate squared visibilities.
        v2mod = (
            abs(
                interp_norm(
                    (container_data.v2wave, container_data.v2vf, container_data.v2uf)
                )
            )
            ** 2
        )

        # Calculate closure phases. We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,
        # v3) = AC, not CA This causes a minus sign shift for 3rd baseline when calculating closure phase (for real
        # images), so we take the complex conjugate there.
        t3phimod = np.angle(
            interp_norm(
                (container_data.t3wave, container_data.t3vf1, container_data.t3uf1)
            )
            * interp_norm(
                (container_data.t3wave, container_data.t3vf2, container_data.t3uf2)
            )
            * np.conjugate(
                interp_norm(
                    (container_data.t3wave, container_data.t3vf3, container_data.t3uf3)
                )
            ),
            deg=True,
        )

    # initialize dictionary to construct OIContainer for model observables
    observables_mod = {
        "vuf": container_data.vuf,
        "vvf": container_data.vvf,
        "vwave": container_data.vwave,
        "v": vmod,
        "verr": container_data.verr,
        "vbase": container_data.vbase,
        "v2uf": container_data.v2uf,
        "v2vf": container_data.v2vf,
        "v2wave": container_data.v2wave,
        "v2": v2mod,
        "v2err": container_data.v2err,
        "v2base": container_data.v2base,
        "t3uf1": container_data.t3uf1,
        "t3vf1": container_data.t3vf1,
        "t3uf2": container_data.t3uf2,
        "t3vf2": container_data.t3vf2,
        "t3uf3": container_data.t3uf3,
        "t3vf3": container_data.t3vf3,
        "t3wave": container_data.t3wave,
        "t3phi": t3phimod,
        "t3phierr": container_data.t3phierr,
        "t3bmax": container_data.t3bmax,
    }

    # return an OIContainer object
    container_mod = OIContainer(
        dictionary=observables_mod, fcorr=container_data.vis_in_fcorr
    )
    return container_mod


def mod_comp_vis_interpolator(
    img_fft_list: list[image_fft.ImageFFT], fcorr: bool = False
) -> RegularGridInterpolator:
    """
    Creates a scipy RegularGridInterpolator from model ImageFFT objects, which can be used to interpolate the complex
    visibility to different spatial frequencies than those returned by the FFT algorithm and, optionally,
    different wavelengths than those of the RT model images themselves. Note: The interpolator will throw errors if
    arguments outside their bounds are supplied! Note: Expects, in case of multiple model images, that every image
    included has the same pixelscale and amount of pixels (in both x- and y-direction).

    :param list img_fft_list: List of ImageFFT objects to create an interpolator from. If the list has length one,
        i.e. a monochromatic model for the emission, the returned interpolator can only take the 2 spatial frequencies
        as arguments. If the list contains multiple objects, i.e. a chromatic model for the emission, the interpolator
        will also be able to take wavelength as an argument and will be able to interpolate along the wavelength
        dimension.
    :param bool fcorr: Set to True if you want the returned interpolator to produce absolute, non-normalized
        complex visibilities, i.e. complex correlated fluxes (units Jy). By default, i.e. fcorr=False,
        the visibilities produced by the interpolator are normalized (e.g. for calculating squared visibilities).
    :return interpolator: Interpolator for the model image FFTs. If len(img_fft_list) == 1, only takes the uv spatial
        frequencies (units 1/rad) as arguments as follows: interpolator(v, u).  If len(img_fft_list) > 1, then it also
        can interpolate between wavelengths (units meter) as follows: interpolator(wavelength, v, u).
    :rtype: scipy.interpolate.RegularGridInterpolator
    """

    if len(img_fft_list) == 1:  # single image -> monochromatic emission model
        img = img_fft_list[0]
        wavelength, ftot, fft, uf, vf = (
            img.wavelength,
            img.ftot,
            img.fft,
            img.uf,
            img.vf,
        )
        if (
            fcorr
        ):  # create interpolator and normalize FFT to complex visibilities if needed
            interpolator = RegularGridInterpolator(
                (vf, uf), fft, method="linear"
            )  # make interpol absolute FFT
        else:
            interpolator = RegularGridInterpolator(
                (vf, uf), fft / ftot, method="linear"
            )  # same normalized

    else:  # multiple images -> chromatic emission model
        mod_wavelengths = []  # list of model image wavelengths in meter
        fft_chromatic = []  # 3d 'array' list to store the different model image FFTs accros wavelength

        for img in img_fft_list:
            wavelength, ftot, fft, uf, vf = (
                img.wavelength,
                img.ftot,
                img.fft,
                img.uf,
                img.vf,
            )
            if fcorr:  # attach FFTs to chromatic list and normalize FFTs to complex visibilities if needed
                fft_chromatic.append(fft)  # store image's FFT in chromatic list
            else:
                fft_chromatic.append(fft / ftot)
            mod_wavelengths.append(
                wavelength * constants.MICRON2M
            )  # store image wavelength in meter

        # sort lists according to ascending wavelength just to be sure (required for making the interpolator)
        mod_wavelengths, fft_chromatic = list(
            zip(*sorted(zip(mod_wavelengths, fft_chromatic)))
        )

        # make interpolator from multiple FFTs, note this assumes all images have the same pixelscale
        # and amount of pixels (in both x and y directions)
        fft_chromatic = np.array(fft_chromatic)
        interpolator = RegularGridInterpolator((mod_wavelengths, vf, uf), fft_chromatic)
    return interpolator


def plot_data_vs_model(
    container_data: OIContainer,
    container_mod: OIContainer,
    fig_dir: str = None,
    log_plotv: bool = False,
    plot_vistype: str = "vis2",
    show_plots: bool = True,
) -> None:
    """
    Plots the data against the model OI observables. Currently, plots uv coverage, a (squared) visibility curve and
    closure phases. Note that this function shares a name with a similar function in the sed module. Take care with
    your namespace if you use both functions in the same script.

    :param OIContainer container_data: Container with data observables.
    :param OIContainer container_mod: Container with model observables.
    :param str fig_dir: Directory to store plots in.
    :param bool log_plotv: Set to True for a logarithmic y-scale in the (squared) visibility plot.
    :param str plot_vistype: Sets the type of visibility to be plotted. 'vis2' for squared visibilities or 'vis'
        for visibilities (either normalized or correlated flux in Jy, as implied by the OIContainer objects).
    :param bool show_plots: Set to False if you do not want the plots to be shown during your python instance.
        Note that if True, this freazes further code execution until the plot windows are closed.
    :rtype: None
    """
    # create plotting directory if it doesn't exist yet
    if fig_dir is not None:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == "vis2":
        ufdata = container_data.v2uf
        vfdata = container_data.v2vf
        vismod = container_mod.v2
        visdata = container_data.v2
        viserrdata = container_data.v2err
        wavedata = container_data.v2wave
        basedata = container_data.v2base
        vislabel = "$V^2$"
    elif plot_vistype == "vis":
        ufdata = container_data.vuf
        vfdata = container_data.vvf
        vismod = container_mod.v
        wavedata = container_data.vwave
        visdata = container_data.v
        viserrdata = container_data.verr
        basedata = container_data.vbase
        if (not container_data.vis_in_fcorr) and (not container_mod.vis_in_fcorr):
            vislabel = "$V$"
        elif container_data.vis_in_fcorr and container_mod.vis_in_fcorr:
            vislabel = r"$F_{corr}$ (Jy)"
        else:
            print(
                "container_data and container_mod do not have the same value for vis_in_fcorr, will return None!"
            )
            return
    else:
        print("parameter plot_vistype is not recognized, will return None!")
        return

    # plot uv coverage
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    ax.set_aspect("equal", adjustable="datalim")  # make plot axes have the same scale
    ax.scatter(
        ufdata / 1e6,
        vfdata / 1e6,
        c=wavedata * constants.M2MICRON,
        s=1,
        cmap="gist_rainbow_r",
    )
    sc = ax.scatter(
        -ufdata / 1e6,
        -vfdata / 1e6,
        c=wavedata * constants.M2MICRON,
        s=1,
        cmap="gist_rainbow_r",
    )
    clb = fig.colorbar(sc, cax=cax)
    clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

    ax.set_xlim(ax.get_xlim()[::-1])  # switch x-axis direction
    ax.set_title("uv coverage")
    ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
    ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/uv_plane.png", dpi=300, bbox_inches="tight")

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
        plt.savefig(f"{fig_dir}/visibilities.png", dpi=300, bbox_inches="tight")

    # plot phi_closure
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(
        container_data.t3bmax,
        container_data.t3phi,
        container_data.t3phierr,
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
        container_data.t3bmax,
        container_mod.t3phi,
        label="model",
        marker="o",
        facecolor="white",
        edgecolor="r",
        s=4,
        alpha=0.6,
    )
    ax[1].scatter(
        container_data.t3bmax,
        (container_mod.t3phi - container_data.t3phi) / container_data.t3phierr,
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
            np.min(container_data.t3phi - container_data.t3phierr),
            np.min(container_mod.t3phi),
        ),
        max(
            np.max(container_data.t3phi + container_data.t3phierr),
            np.max(container_mod.t3phi),
        ),
    )

    ax[1].set_xlim(0, np.max(container_data.t3bmax) * 1.05)
    ax[1].axhline(y=0, c="k", ls="--", lw=1, zorder=0)
    ax[1].set_xlabel(r"$B_{max}$ ($\mathrm{M \lambda}$)")
    ax[1].set_ylabel(r"error $(\sigma_{\phi_{CP}})$")
    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/closure_phases.png", dpi=300, bbox_inches="tight")
    if show_plots:
        plt.show()

    return


if __name__ == "__main__":
    from distroi.auxiliary.beam import calc_gaussian_beam

    # object_id_list = ['AI_Sco', 'EN_TrA', 'HD93662', 'HD95767', 'HD108015', 'HR4049', 'IRAS08544-4431', 'IRAS15469-5311',
    #                   'IW_Car', 'PS_Gem', 'U_Mon']
    # for object_id in object_id_list:
    #     data_dir, data_file = (f'/home/toond/Documents/phd/data/{object_id}/inspiring/PIONIER/all_data/',
    #                            '*.fits')
    #     container_data = read_oicontainer_oifits(data_dir, data_file)
    #     fig_dir = f'{data_dir}/figures/'
    #     container_data.plot_data(fig_dir=fig_dir)
    #     beam = calc_gaussian_beam(container_data, vistype='vis2', make_plots=True, show_plots=True, fig_dir=fig_dir,
    #                               num_res=2, pix_per_res=48)

    object_id = "IRAS15469-5311"
    data_dir, data_file = (
        f"/home/toond/Documents/phd/data/{object_id}/inspiring/PIONIER/img_ep_jan2021-mar2021/",
        "*.fits",
    )
    container_data = read_oicontainer_oifits(data_dir, data_file)
    fig_dir = f"{data_dir}/figures/"
    container_data.plot_data(fig_dir=fig_dir)
    beam = calc_gaussian_beam(
        container_data,
        vistype="vis2",
        make_plots=True,
        show_plots=True,
        fig_dir=fig_dir,
        num_res=2,
        pix_per_res=64,
    )
