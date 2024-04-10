"""
Contains a class to store optical inteferometric (OI) observables and additional functions to take radiative
transfer (RT) model images and convert them to interferometric observables at the spatial frequencies of data
stored in the OIFITS format. Currently supports the following combinations of obsevrables: Squared visibilities -
Closure phases ; Visibilities - Closure phases ; Correlated fluxes (formaly stored as visibilities) - Closure phases.
"""

from distroi import constants
from distroi import image
from distroi.auxiliary import SelectData

import os
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
from distroi.constants import set_matplotlib_params

set_matplotlib_params()  # set project matplotlib parameters


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
    :ivar numpy.ndarray vuf: u-axis spatial freqs in 1/rad for visibility data.
    :ivar numpy.ndarray vvf: v-axis spatial freqs in 1/rad for visibility data.
    :ivar numpy.ndarray vwave: Wavelengths in meter for visibility data.
    :ivar numpy.ndarray v: Visibilities, either normalized visibilities or correlated flux in Janksy.
    :ivar numpy.ndarray verr: Error on visibilities.
    :ivar numpy.ndarray vbase: Baseline length in MegaLambda for visibility data.
    :ivar numpy.ndarray v2uf: u-axis spatial freqs in 1/rad for squared visibility data.
    :ivar numpy.ndarray v2vf: v-axis spatial freqs in 1/rad for squared visibility data.
    :ivar numpy.ndarray v2wave: Wavelengths in meter for squared visibility data.
    :ivar numpy.ndarray v2: Squared visibilities.
    :ivar numpy.ndarray v2err: Error on squared visibilities.
    :ivar numpy.ndarray v2base: Baseline length in MegaLambda for squared visibility data.
    :ivar numpy.ndarray t3uf1: u-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    :ivar numpy.ndarray t3vf1: v-axis spatial freqs in 1/rad for the 1st projected baseline along the closure triangle.
    :ivar numpy.ndarray t3uf3: u-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    :ivar numpy.ndarray t3vf2: v-axis spatial freqs in 1/rad for the 2nd projected baseline along the closure triangle.
    :ivar numpy.ndarray t3uf3: u-axis spatial freqs in 1/rad for the 3d projected baseline along the closure triangle.
    :ivar numpy.ndarray t3vf3: v-axis spatial freqs in 1/rad for the 3d projected baseline along the closure triangle.
    :ivar numpy.ndarray t3wave: Wavelengths in meter for closure phase data.
    :ivar numpy.ndarray t3phi: Closure phases in degrees.
    :ivar numpy.ndarray t3phierr: Error on closure phases.
    :ivar numpy.ndarray t3bmax: Maximum baseline length along the closure triangle in units of MegaLambda.
    """

    def __init__(self, dictionary, fcorr=False):
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
            self.vuf = dictionary['vuf']
            self.vvf = dictionary['vvf']
            self.vwave = dictionary['vwave']
            self.v = dictionary['v']
            self.verr = dictionary['verr']
            self.vbase = dictionary['vbase']

            self.v2uf = dictionary['v2uf']
            self.v2vf = dictionary['v2vf']
            self.v2wave = dictionary['v2wave']
            self.v2 = dictionary['v2']
            self.v2err = dictionary['v2err']
            self.v2base = dictionary['v2base']

            self.t3uf1 = dictionary['t3uf1']
            self.t3vf1 = dictionary['t3vf1']
            self.t3uf2 = dictionary['t3uf2']
            self.t3vf2 = dictionary['t3vf2']
            self.t3uf3 = dictionary['t3uf3']
            self.t3vf3 = dictionary['t3vf3']
            self.t3wave = dictionary['t3wave']
            self.t3phi = dictionary['t3phi']
            self.t3phierr = dictionary['t3phierr']
            self.t3bmax = dictionary['t3bmax']


def read_oicontainer_oifits(data_dir, data_file, wave_lims=None, v2lim=None, fcorr=False):
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
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file,
                                       wave_1=wave_lims[0] * constants.MICRON2M,
                                       wave_2=wave_lims[1] * constants.MICRON2M, lim_V2=v2lim)
    else:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file, lim_V2=v2lim)

    # dictionary to construct OIContainer instance
    dictionary = {}

    # arrays to store all necessary variables in a 1d array
    vufdat, vvfdat, vwavedat, vdat, verr = [], [], [], [], []  # for visibility tables
    v2ufdat, v2vfdat, v2wavedat, v2dat, v2err = [], [], [], [], []  # for squared visibility tables
    uf1dat, vf1dat, uf2dat, vf2dat, t3wavedat, t3phidat, t3phierr = [], [], [], [], [], [], []  # for closure phase
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
    vbase = np.sqrt(vufdat ** 2 + vvfdat ** 2) / 1e6  # uv baseline length in MegaLambda

    v2ufdat = np.array(v2ufdat)  # transfer into numpy arrays
    v2vfdat = np.array(v2vfdat)
    v2wavedat = np.array(v2wavedat)
    v2dat = np.array(v2dat)
    v2err = np.array(v2err)
    v2base = np.sqrt(v2ufdat ** 2 + v2vfdat ** 2) / 1e6  # uv baseline length in MegaLambda

    uf1dat = np.array(uf1dat)
    vf1dat = np.array(vf1dat)
    uf2dat = np.array(uf2dat)
    vf2dat = np.array(vf2dat)
    t3wavedat = np.array(t3wavedat)
    t3phidat = np.array(t3phidat)
    t3phierr = np.array(t3phierr)

    uf3dat = uf1dat + uf2dat  # 3d baseline (frequency) and max baseline of closure triangle (in MegaLambda)
    vf3dat = vf1dat + vf2dat
    t3bmax = np.maximum(np.sqrt(uf3dat ** 2 + vf3dat ** 2),
                        np.maximum(np.sqrt(uf1dat ** 2 + vf1dat ** 2),
                                   np.sqrt(uf2dat ** 2 + vf2dat ** 2))) / 1e6

    # fill in data observables dictionary
    dictionary['vuf'] = vufdat  # spatial freqs in 1/rad
    dictionary['vvf'] = vvfdat
    dictionary['vwave'] = vwavedat  # wavelengths in meter
    dictionary['v'] = vdat
    dictionary['verr'] = verr  # baseline length in MegaLambda
    dictionary['vbase'] = vbase

    dictionary['v2uf'] = v2ufdat  # spatial freqs in 1/rad
    dictionary['v2vf'] = v2vfdat
    dictionary['v2wave'] = v2wavedat  # wavelengths in meter
    dictionary['v2'] = v2dat
    dictionary['v2err'] = v2err  # baseline length in MegaLambda
    dictionary['v2base'] = v2base

    dictionary['t3uf1'] = uf1dat  # spatial freqs in 1/rad
    dictionary['t3vf1'] = vf1dat
    dictionary['t3uf2'] = uf2dat
    dictionary['t3vf2'] = vf2dat
    dictionary['t3uf3'] = uf3dat
    dictionary['t3vf3'] = vf3dat
    dictionary['t3wave'] = t3wavedat  # wavelengths in meter
    dictionary['t3phi'] = t3phidat  # closure phases in degrees
    dictionary['t3phierr'] = t3phierr
    dictionary['t3bmax'] = t3bmax  # max baseline lengths in MegaLambda

    # Return an OIContainer object
    container = OIContainer(dictionary=dictionary, fcorr=fcorr)
    return container


def calc_model_observables(container_data, mod_dir, img_dir, monochr=False, ebminv=0.0,
                           read_method='mcfost'):
    """
    Loads in OI observables from an OIContainer, typically containing observational data, and calculates model image
    observables at the same uv coverage. In case monochr=False, expects the same amount of pixels and same pixelscale
    for every model image included. TODO: Only supports MCFOST image file naming convention, make more general.

    :param OIContainer container_data: OIContainer at whose spatial frequencies we calculate model observables.
    :param str mod_dir: Parent directory of the RT model of interest.
    :param str img_dir: Subdirectory containing RT model images. If 'monochr' is True, uses the first image file found
        in 'mod_dir+img_dir'. If False, then all image files in the subdirectories of 'mod_dir+img_dir' are used.
    :param bool monochr: Set True if you want to use a monochromatic model, i.e. no interpolation is performed
        in the wavelength dimension. If False, multiple images are loaded in and interpolation of the model image FFTs
        along wavelength is performed. In this case the model images used must collectively  have a wider wavelength
        coverage than 'container_data'.
    :param float ebminv: E(B-V) of additional reddening to be applied to the model images. Only useful if
        the visibilities in 'container_data' are expressed in correlated flux.
    :param str read_method: Type of method used to read in RT model images when creating Image class instances.
        Currently only supports 'mcfost'.
    :return container_mod: OIContainer for model image observables.
    :rtype: OIContainer
    """

    if monochr:
        # load in and perform FFT on a single image
        if read_method == 'mcfost':
            transform = image.read_image_mcfost(f'{mod_dir}{img_dir}/RT.fits.gz')
        transform.redden(ebminv=ebminv)  # redden the image
        wavelength, ftot, img_fft, uf, vf = (transform.wavelength, transform.ftot, transform.img_fft,
                                             transform.uf, transform.vf)

        #  make interpolator from the absolute image (in Jansky) and calculate model observables from that
        interpolator = RegularGridInterpolator((vf, uf), img_fft)
        if not container_data.vis_in_fcorr:
            vmod = abs(interpolator((container_data.vvf, container_data.vuf)) / ftot)  # visibilities normalized
        else:
            vmod = abs(interpolator((container_data.vvf, container_data.vuf)))  # visibilities in correlated flux
        v2mod = abs(interpolator((container_data.v2vf, container_data.v2uf)) / ftot) ** 2  # squared visibilities
        # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images),
        # so we take the complex conjugate there.
        t3phimod = np.angle(interpolator((container_data.t3vf1, container_data.t3uf1)) *
                            interpolator((container_data.t3vf2, container_data.t3uf2)) *
                            np.conjugate(interpolator((container_data.t3vf3, container_data.t3uf3))), deg=True)

    else:
        # list of MCFOST image subdirectories to use
        mod_img_subdirs = sorted(glob.glob(f'{img_dir}/*data_*', root_dir=mod_dir),
                                 key=lambda x: float(x.split("_")[-1]))

        mod_wavelengths = []  # model image wavelengths in meter
        img_fft_norm_chromatic = []  # 3d array to store FFT normalized by total flux at diferent wavelengths
        img_fft_chromatic = []  # 3d array to store absolute FFT (in Jansky)

        for img_subdir in mod_img_subdirs:
            # load in and perform FFT on a single image
            if read_method == 'mcfost':
                transform = image.read_image_mcfost(f'{mod_dir}{img_subdir}/RT.fits.gz')
            transform.redden(ebminv=ebminv)  # redden the image
            wavelength, ftot, img_fft, uf, vf = (transform.wavelength, transform.ftot, transform.img_fft,
                                                 transform.uf, transform.vf)

            mod_wavelengths.append(wavelength * constants.MICRON2M)
            img_fft_norm_chromatic.append(img_fft / ftot)
            img_fft_chromatic.append(img_fft)

        img_fft_norm_chromatic = np.array(img_fft_norm_chromatic)
        img_fft_chromatic = np.array(img_fft_chromatic)

        #  make interpolators (one for normalized and one for non-normalized FFT) and calculate model observables
        #  the non-normalized one is only necessary if the data's visibilities in correlated flux
        interpolator_norm = RegularGridInterpolator((mod_wavelengths, vf, uf), img_fft_norm_chromatic)
        if container_data.vis_in_fcorr:
            interpolator = RegularGridInterpolator((mod_wavelengths, vf, uf), img_fft_chromatic)
            vmod = abs(interpolator((container_data.vwave, container_data.vvf, container_data.vuf)))
        else:
            vmod = abs(interpolator_norm((container_data.vwave, container_data.vvf, container_data.vuf)))

        v2mod = abs(interpolator_norm((container_data.v2wave, container_data.v2vf, container_data.v2uf))) ** 2
        # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images),
        # so we take the complex conjugate there.
        t3phimod = np.angle(interpolator_norm((container_data.t3wave, container_data.t3vf1, container_data.t3uf1)) *
                            interpolator_norm((container_data.t3wave, container_data.t3vf2, container_data.t3uf2)) *
                            np.conjugate(interpolator_norm((container_data.t3wave, container_data.t3vf3,
                                                            container_data.t3uf3))), deg=True)

    # initialize dictionary to construct OIContainer for model observables
    observables_mod = {'vuf': container_data.vuf, 'vvf': container_data.vvf, 'vwave': container_data.vwave, 'v': vmod,
                       'verr': container_data.verr, 'vbase': container_data.vbase, 'v2uf': container_data.v2uf,
                       'v2vf': container_data.v2vf, 'v2wave': container_data.v2wave, 'v2': v2mod,
                       'v2err': container_data.v2err, 'v2base': container_data.v2base, 't3uf1': container_data.t3uf1,
                       't3vf1': container_data.t3vf1, 't3uf2': container_data.t3uf2, 't3vf2': container_data.t3vf2,
                       't3uf3': container_data.t3uf3, 't3vf3': container_data.t3vf3, 't3wave': container_data.t3wave,
                       't3phi': t3phimod, 't3phierr': container_data.t3phierr, 't3bmax': container_data.t3bmax}

    container_mod = OIContainer(dictionary=observables_mod, fcorr=container_data.vis_in_fcorr)

    return container_mod


# def get_complex_visibility_interpolator(mod_dir, img_dir, monochr=False, ebminv=0.0, fcorr=False,
#                                         read_method='mcfost'):
#     """
#     Creates a scipy.interpolate.RegularGridInterpolator of RT model image(s) FFT(s), which can be used to interpolate
#     the complex visibility to different spatial frequencies than those returned by the FFT algorithm and, optionally,
#     different wavelengths than those of the RT model images themselves. Note that the interpolator will throw
#     errors if arguments outside their bounds are supplied!
#
#     :param str mod_dir: Parent directory of the RT model of interest.
#     :param str img_dir: Subdirectory containing RT model images. If 'monochr' is True, uses the first image file found
#         in 'mod_dir+img_dir'. If False, then all image files in the subdirectories of 'mod_dir+img_dir' are used.
#     :param bool monochr: Set True if you want to use a monochromatic imterpolator, i.e. the interpolator will only take
#         the spatial frequencies as arguments and will not interpolate in wavelength. If False, the interpolator will
#         take wavelength as a third argument as well and will be able to interpolate in wavelength.
#     :param float ebminv: E(B-V) of additional reddening to be applied to the model images. Only useful if 'fcorr' is set
#         to True.
#     :param bool fcorr: Set to True if you want the returned interpolator to produce absolute, non-normalized
#         complex visibilities, i.e. complex correlated fluxes, in units of Jy. By default, i.e. fcorr=False,
#         the visibilities produced by the interpolator are normalized (i.e. for calculating squared visibilities).
#     :param str read_method: Type of method used to read in RT model images when creating Image class instances.
#         Currently only supports 'mcfost'.
#     """
#     return


def plot_data_vs_model(container_data, container_mod, fig_dir=None, log_plotv=False, plot_vistype='vis2',
                       show_plots=False):
    """
    Plots the data against the model OI observables. Currently, plots uv-coverage, a (squared) visibility curve and
    closure phases.

    :param OIContainer container_data: Container with data observables.
    :param OIContainer container_mod: Container with model observables.
    :param str fig_dir: Directory to store plots in.
    :param bool log_plotv: Set to True for a logarithmic y-scale in the (squared) visibility plot.
    :param str plot_vistype: Sets the type of visibility to be plotted. 'vis2' for squared visibilities or 'vis'
        for visibilities (either normalized or correlated flux in Jy, as implied by the OIContainer objects).
    :param bool show_plots: Set to True if you want the plots to be shown during your python instance. Note that this
        freazes further code execution until the plot windows are closed.
    :rtype: None
    """
    # create plotting directory if it doesn't exist yet
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == 'vis2':
        ufdata = container_data.v2uf
        vfdata = container_data.v2vf
        vismod = container_mod.v2
        visdata = container_data.v2
        viserrdata = container_data.v2err
        wavedata = container_data.v2wave
        basedata = container_data.v2base
        vislabel = '$V^2$'
    elif plot_vistype == 'vis':
        ufdata = container_data.vuf
        vfdata = container_data.vvf
        vismod = container_mod.v
        wavedata = container_data.vwave
        visdata = container_data.v
        viserrdata = container_data.verr
        basedata = container_data.vbase
        if (not container_data.vis_in_fcorr) and (not container_mod.vis_in_fcorr):
            vislabel = '$V$'
        elif container_data.vis_in_fcorr and container_mod.vis_in_fcorr:
            vislabel = r'$F_{corr}$ (Jy)'
        else:
            print("container_data and container_mod do not have the same value for vis_in_fcorr, will return None!")
            return
    else:
        print("parameter plot_vistype is not recognized, will return None!")
        return

    # plot uv coverage
    color_map = 'gist_rainbow_r'  # color map for wavelengths

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.82, 0.15, 0.02, 0.7])

    ax.scatter(ufdata / 1e6, vfdata / 1e6,
               c=wavedata * constants.M2MICRON, s=1,
               cmap=color_map)
    sc = ax.scatter(-ufdata / 1e6, -vfdata / 1e6,
                    c=wavedata * constants.M2MICRON, s=1,
                    cmap=color_map)
    clb = fig.colorbar(sc, cax=cax)
    clb.set_label(r'$\lambda$ ($\mu$m)', rotation=270, labelpad=15)

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_title(f'uv coverage')
    ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
    ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

    plt.savefig(f"{fig_dir}/uv_plane.png", dpi=300, bbox_inches='tight')

    # plot (squared) visibilities
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(basedata, visdata, viserrdata, label='data', mec='royalblue',
                   marker='o', capsize=0, zorder=0, markersize=2, ls='', alpha=0.8, elinewidth=0.5)
    ax[0].scatter(basedata, vismod, label=f'model', marker='o',
                  facecolor='white', edgecolor='r', s=4, alpha=0.6)
    ax[1].scatter(basedata, (vismod - visdata) / viserrdata,
                  marker='o', facecolor='white', edgecolor='r', s=4, alpha=0.6)

    ax[0].set_ylabel(vislabel)
    ax[0].legend()
    ax[0].set_title(f'Visibilities')
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    if log_plotv:
        ax[0].set_ylim(0.5 * np.min(visdata), 1.1 * np.max(np.maximum(visdata, vismod)))
        ax[0].set_yscale('log')
    else:
        ax[0].set_ylim(0, 1.1 * np.max(np.maximum(visdata, vismod)))

    ax[1].set_xlim(0, np.max(basedata) * 1.05)
    ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
    ax[1].set_xlabel(r'$B$ ($\mathrm{M \lambda}$)')
    ax[1].set_ylabel(r'error $(\sigma)$')
    plt.savefig(f"{fig_dir}/visibilities.png", dpi=300, bbox_inches='tight')

    # plot phi_closure
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
    ax = gs.subplots(sharex=True)

    ax[0].errorbar(container_data.t3bmax, container_data.t3phi, container_data.t3phierr, label='data', mec='royalblue',
                   marker='o', capsize=0, zorder=0, markersize=2, ls='', alpha=0.8, elinewidth=0.5)
    ax[0].scatter(container_data.t3bmax, container_mod.t3phi, label=f'model', marker='o',
                  facecolor='white', edgecolor='r', s=4, alpha=0.6)
    ax[1].scatter(container_data.t3bmax, (container_mod.t3phi - container_data.t3phi) / container_data.t3phierr,
                  marker='o', facecolor='white', edgecolor='r', s=4, alpha=0.6)

    ax[0].set_ylabel(r'$\phi_{CP}$ ($^\circ$)')
    ax[0].legend()
    ax[0].set_title(f'Closure Phases')
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    ax[1].set_xlim(0, np.max(container_data.t3bmax) * 1.05)
    ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
    ax[1].set_xlabel(r'$B_{max}$ ($\mathrm{M \lambda}$)')
    ax[1].set_ylabel(r'error $(\sigma_{\phi_{CP}})$')

    if fig_dir is not None:
        plt.savefig(f"{fig_dir}/closure_phases.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()

    return
