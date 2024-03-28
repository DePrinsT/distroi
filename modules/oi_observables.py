"""
File: oi_observables.py
Author: Toon De Prins
Description: Contains a class to store inteferometric observables and additional functions to take model disk images
and convert them to interferometric observables at the spatial frequencies of selected observational data (stored in
the OIFITS format). Currently supports the following combinations of obsevrables: Squared visibilities - Closure phases
; Visibilities - Closure phases ; Correlated fluxes (formaly stored as visibilities) - Closure phases.
"""

from modules import constants
from modules import image_fft
from modules.external import SelectData

import os
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import modules.matplotlib_settings as matplotlib_settings

matplotlib_settings.set_matplotlib_params()  # set project matplotlib parameters


class OIContainer:
    """
    Class to contain optical interferometry observables, where each observable is stored in the form of raveled 1D
    numpy arrays. Heavily based on the 'Data' class in ReadOIFITS.py, where the observables are stored per table,
    but the raveled form makes calculations between models and data much easier and less prone to error. Currenlty
    supports visibilities (both in normalized and correlated flux form), squared visibilities and closure phases.
    In this format this class and the methods below can be expanded to accomodate for different kinds of observables
    (i.e. addition of differntial visibilities).
    """

    def __init__(self, observables):
        """
        Creates an OIobservables object, assigning all either necessary properties by reading them in from a dictionary.

        Parameters:
            observables (dict): dictionary containing arrays describing all observables
        Properties initialized:
        """
        # Properties for visibility (v) data
        self.vuf = np.array([])  # u-axis spatial freqs in 1/rad
        self.vvf = np.array([])  # v-axis spatial freqs in 1/rad
        self.vwave = np.array([])  # wavelengths in meter
        self.v = np.array([])  # visibilities (either normalized visibilities or correlated flux in Janksy)
        self.verr = np.array([])  # error on visibilities
        self.vbase = np.array([])  # baseline length in MegaLambda (i.e. 1e6*vwave)

        # Properties for squared visibility (v2) data
        self.v2uf = np.array([])  # u-axis spatial freqs in 1/rad
        self.v2vf = np.array([])  # v-axis spatial freqs in 1/rad
        self.v2wave = np.array([])  # wavelengths in meter
        self.v2 = np.array([])  # squared visibilities
        self.v2err = np.array([])  # errors on squared visibilities
        self.v2base = np.array([])  # baseline length in MegaLambda (i.e. 1e6*vwave)

        # Properties for closure phase (t3) data
        self.t3uf1 = np.array([])  # u-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3vf1 = np.array([])  # v-axis spatial freqs in 1/rad for the 3 projected baselines
        self.t3uf2 = np.array([])
        self.t3vf2 = np.array([])
        self.t3uf3 = np.array([])
        self.t3vf3 = np.array([])
        self.t3wave = np.array([])  # wavelenghts in meter
        self.t3phi = np.array([])  # closure phases in degrees
        self.t3phierr = np.array([])  # errors on closure phases
        self.t3bmax = np.array([])  # maximum baseline lengths along the closure triangles in units of MegaLambda

        # Read in from the dictionary
        if observables is not None:
            self.vuf = observables['vuf']
            self.vvf = observables['vvf']
            self.vwave = observables['vwave']
            self.v = observables['v']
            self.verr = observables['verr']
            self.vbase = observables['vbase']

            self.v2uf = observables['v2uf']
            self.v2vf = observables['v2vf']
            self.v2wave = observables['v2wave']
            self.v2 = observables['v2']
            self.v2err = observables['v2err']
            self.v2base = observables['v2base']

            self.t3uf1 = observables['t3uf1']
            self.t3vf1 = observables['t3vf1']
            self.t3uf2 = observables['t3uf2']
            self.t3vf2 = observables['t3vf2']
            self.t3uf3 = observables['t3uf3']
            self.t3vf3 = observables['t3vf3']
            self.t3wave = observables['t3wave']
            self.t3phi = observables['t3phi']
            self.t3phierr = observables['t3phierr']
            self.t3bmax = observables['t3bmax']


def container_from_oifits(data_dir, data_file, wave_lims=None, v2lim=None):
    """
    Function to retrieve observation data from OIFITS files and return it as raveled numpy arrays in a dictionary.
    This is basically a wrapper around ReadOIFITS.py and SelectData.py, but the raveled numpy arrays make things
    easier to calculate/interpolate using numpy/scipy. wave_1 and 2 are wavelength limits in meter to be applied.
    """
    # if condition because * constants.MICRON2M fails if wavelimits are False
    if wave_lims[0] is not None and wave_lims[1] is not None:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file,
                                       wave_1=wave_lims[0] * constants.MICRON2M,
                                       wave_2=wave_lims[1] * constants.MICRON2M, lim_V2=v2lim)
    else:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file, lim_V2=v2lim)

    # dictionary to construct
    observables = {}

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
    observables['vuf'] = vufdat  # spatial freqs in 1/rad
    observables['vvf'] = vvfdat
    observables['vwave'] = vwavedat  # wavelengths in meter
    observables['v'] = vdat
    observables['verr'] = verr  # baseline length in MegaLambda
    observables['vbase'] = vbase

    observables['v2uf'] = v2ufdat  # spatial freqs in 1/rad
    observables['v2vf'] = v2vfdat
    observables['v2wave'] = v2wavedat  # wavelengths in meter
    observables['v2'] = v2dat
    observables['v2err'] = v2err  # baseline length in MegaLambda
    observables['v2base'] = v2base

    observables['t3uf1'] = uf1dat  # spatial freqs in 1/rad
    observables['t3vf1'] = vf1dat
    observables['t3uf2'] = uf2dat
    observables['t3vf2'] = vf2dat
    observables['t3uf3'] = uf3dat
    observables['t3vf3'] = vf3dat
    observables['t3wave'] = t3wavedat  # wavelengths in meter
    observables['t3phi'] = t3phidat  # closure phases in degrees
    observables['t3phierr'] = t3phierr
    observables['t3bmax'] = t3bmax  # max baseline lengths in MegaLambda
    
    # Return an OIContainer object
    container = OIContainer(observables)
    return container


def calc_model_observables(container_data, mod_dir, img_dir, monochr=False, fcorr=False, ebminv=0.0,
                           read_method='mcfost', disk_only=False):
    """
    Function that loads in the OIFITS observations and calculates MCFOST model interferometry observables at the same
    spatial frequencies. The monochromatism 'monochr' argument can be used to use only observation data between the
    wavelength interval (wavelim_lower, wavelim_upper). In this case the 'img_dir' argument needs to be specified,
    as only the specific MCFOST .fits.gz output image file in 'mod_dir'+'img_dir' will be used for calculating the
    FFT. No interpolation in the wavelength dimension will be performed.

    If monochr=True, all image subdirectories in the directory 'mod_dir'+'img_dir' will instead be used to
    interpolate between wavelengths. In this case the wavelength coverage of the MCFOST images needs to be wider than
    that of (wavelim_lower, wavelim_upper), otherwise an error is thrown. Spatial frequencies, wavelengths and
    observables are returned in dictionaries. I.e. in this case 'mod_dir'+'img_dir' denotes the superdirectory which
    contains subdirectories for each wavelength (each of these containing their own .fits.gz output image file). (
    wavelim_lower, wavelim_upper) can still be used if monochr=True to restrict the loaded data (e.g. if the data at
    wavelength edges is bad). Note: assumes all images have the same amount of pixels and pixelscale (in angular units).

    If fcorr = True, the model visibilities are calculated and stored as correlated fluxes
    """

    if monochr:
        # perform FFT on single image
        transform = image_fft.ImageFFT(f'{mod_dir}{img_dir}/RT.fits.gz', read_method=read_method,
                                       disk_only=disk_only)
        transform.redden(ebminv=ebminv)  # redden the image 
        wave, ftot, img_fft, uf, vf = transform.wave, transform.ftot, transform.img_fft, transform.uf, transform.vf

        #  make interpolator on absolute image (in Jansky) and calculate model observables from that
        interpolator = RegularGridInterpolator((vf, uf), img_fft)
        if not fcorr:
            vmod = abs(interpolator((container_data.vvf, container_data.vuf)) / ftot)  # visibilities normalized
        else:
            vmod = abs(interpolator((container_data.vvf, container_data.vuf)))  # visibilities in correlated flux
        v2mod = abs(interpolator((container_data.v2vf, container_data.v2uf)) / ftot) ** 2  # squared visibilities
        phi1mod = np.angle(interpolator((container_data.t3vf1, container_data.t3uf1)), deg=True)
        phi2mod = np.angle(interpolator((container_data.t3vf2, container_data.t3uf2)), deg=True)
        phi3mod = np.angle(interpolator((container_data.t3vf3, container_data.t3uf3)), deg=True)
        # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
        t3phimod = phi1mod + phi2mod - phi3mod  # closure phases

    else:
        # list of MCFOST image subdirectories to use
        mod_img_subdirs = sorted(glob.glob(f'{img_dir}/data_*', root_dir=mod_dir),
                                 key=lambda x: float(x.split("_")[-1]))

        mod_waves = []  # model image wavelengths in meter
        img_fft_norm_chromatic = []  # 3d array to store FFT normalized by total flux at diferent wavelengths
        img_fft_chromatic = []  # 3d array to store absolute FFT (in Jansky)

        for img_subdir in mod_img_subdirs:
            # perform FFT on single image
            transform = image_fft.ImageFFT(f'{mod_dir}{img_subdir}/RT.fits.gz', read_method=read_method,
                                           disk_only=disk_only)
            transform.redden(ebminv=ebminv)  # redden the image
            wave, ftot, img_fft, uf, vf = transform.wave, transform.ftot, transform.img_fft, transform.uf, transform.vf

            mod_waves.append(wave * constants.MICRON2M)
            img_fft_norm_chromatic.append(img_fft / ftot)
            img_fft_chromatic.append(img_fft)

        img_fft_norm_chromatic = np.array(img_fft_norm_chromatic)
        img_fft_chromatic = np.array(img_fft_chromatic)

        #  make interpolators (one for normalized and one for non-normalized FFT) and calculate model observables
        #  the non-normalized one is only necessary if we calculate visibilities in correlated flux
        interpolator_norm = RegularGridInterpolator((mod_waves, vf, uf), img_fft_norm_chromatic)
        if fcorr:
            interpolator = RegularGridInterpolator((mod_waves, vf, uf), img_fft_chromatic)
            vmod = abs(interpolator((container_data.vwave, container_data.vvf, container_data.vuf)))
        else:
            vmod = abs(interpolator_norm((container_data.vwave, container_data.vvf, container_data.vuf)))

        v2mod = abs(interpolator_norm((container_data.v2wave, container_data.v2vf, container_data.v2uf))) ** 2
        phi1mod = np.angle(interpolator_norm((container_data.t3wave, container_data.t3vf1, container_data.t3uf1)),
                           deg=True)
        phi2mod = np.angle(interpolator_norm((container_data.t3wave, container_data.t3vf2, container_data.t3uf2)),
                           deg=True)
        phi3mod = np.angle(interpolator_norm((container_data.t3wave, container_data.t3vf3, container_data.t3uf3)),
                           deg=True)
        # We use the convention such that if we have triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
        t3phimod = phi1mod + phi2mod - phi3mod

    # initialize dictionary to construct OIContainer for model observables
    observables_mod = {'vuf': container_data.vuf, 'vvf': container_data.vvf, 'vwave': container_data.vwave, 'v': vmod,
                       'verr': container_data.verr, 'vbase': container_data.vbase, 'v2uf': container_data.v2uf,
                       'v2vf': container_data.v2vf, 'v2wave': container_data.v2wave, 'v2': v2mod,
                       'v2err': container_data.v2err, 'v2base': container_data.v2base, 't3uf1': container_data.t3uf1,
                       't3vf1': container_data.t3vf1, 't3uf2': container_data.t3uf2, 't3vf2': container_data.t3vf2,
                       't3uf3': container_data.t3uf3, 't3vf3': container_data.t3vf3, 't3wave': container_data.t3wave,
                       't3phi': t3phimod, 't3phierr': container_data.t3phierr, 't3bmax': container_data.t3bmax}

    container_model = OIContainer(observables_mod)
    
    return container_model


def plot_data_vs_model(container_data, container_model, fig_dir=None, log_plotv=False, plot_vistype='vis2'):
    # create plotting directory if it doesn't exist yet
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # close previous figures
    plt.close()

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == 'vis2':
        ufdata = container_data.v2uf
        vfdata = container_data.v2vf
        vismod = container_model.v2
        visdata = container_data.v2
        viserrdata = container_data.v2err
        wavedata = container_data.v2wave
        basedata = container_data.v2base
        vislabel = '$V^2$'
    elif plot_vistype == 'vis' or plot_vistype == 'fcorr':
        ufdata = container_data.vuf
        vfdata = container_data.vvf
        vismod = container_model.v
        wavedata = container_data.vwave
        visdata = container_data.v
        viserrdata = container_data.verr
        basedata = container_data.vbase
        if plot_vistype == 'vis':
            vislabel = '$V$'
        else:
            vislabel = r'$F_{corr}$ (Jy)'

    # plot uv coverage
    color_map = 'gist_rainbow_r'  # color map for wavelengths

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

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
    ax[0].scatter(basedata, vismod, label=f'MCFOST model', marker='o',
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
    ax[0].scatter(container_data.t3bmax, container_model.t3phi, label=f'MCFOST model', marker='o',
                  facecolor='white', edgecolor='r', s=4, alpha=0.6)
    ax[1].scatter(container_data.t3bmax, (container_model.t3phi - container_data.t3phi) / container_data.t3phierr,
                  marker='o', facecolor='white', edgecolor='r', s=4, alpha=0.6)

    ax[0].set_ylabel(r'$\phi_{CP}$ ($^\circ$)')
    ax[0].legend()
    ax[0].set_title(f'Closure Phases')
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    ax[1].set_xlim(0, np.max(container_data.t3bmax) * 1.05)
    ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
    ax[1].set_xlabel(r'$B_{max}$ ($\mathrm{M \lambda}$)')
    ax[1].set_ylabel(r'error $(\sigma_{\phi_{CP}})$')

    plt.savefig(f"{fig_dir}/closure_phases.png", dpi=300, bbox_inches='tight')
    return

# # PIONIER tests
# # ------------------------
# print('PIONIER TESTS')
# data_dir = '/home/toond/Documents/phd/data/IRAS0844-4431/PIONIER/'
# data_file = '*.fits'
# mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
# img_dir = 'PIONIER/data_1.65/'
# fig_dir = '/home/toond/Downloads/figs/PIONIER/'
#
# # FFT test
# wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_dir, plotting=True, freq_info=True,
#                                           disk_only=True, fig_dir=fig_dir, plot_vistype='vis2', ebminv=0.0)
#
# # Monochromatic model observables test
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
#                                                    monochr=True, wavelim_lower=1.63, wavelim_upper=1.65,
#                                                    plotting=True, fig_dir=fig_dir, plot_vistype='vis2',
#                                                    ebminv=0.0)
#
# # Chromatic model observables test
# img_dir = 'PIONIER'
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
#                                                    wavelim_lower=None, wavelim_upper=None, plotting=True,
#                                                    fig_dir=fig_dir, plot_vistype='vis2', ebminv=0.0)
# # ------------------------

# GRAVITY tests
# ------------------------
# print('GRAVITY TESTS')
# data_dir = '/home/toond/Documents/phd/data/IRAS0844-4431/GRAVITY/'
# data_file = '*1.fits'
# mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
# img_dir = 'GRAVITY/data_2.2/'
# fig_dir = '/home/toond/Downloads/figs/GRAVITY/'
# #
# # FFT test
# wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_dir, plotting=True, freq_info=True,
#                                           disk_only=False, fig_dir=fig_dir, ebminv=0.0, plot_vistype='vis2')
#
# # Monochromatic model observables test
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
#                                                    monochr=True, wavelim_lower=2.1, wavelim_upper=2.3,
#                                                    plotting=True, fig_dir=fig_dir, ebminv=0.0,
#                                                    plot_vistype='vis2')
#
# # Chromatic model observables test
# img_dir = 'GRAVITY'
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
#                                                    wavelim_lower=None, wavelim_upper=None, plotting=True,
#                                                    fig_dir=fig_dir, ebminv=0.0, plot_vistype='vis2')
# # ------------------------

# # MATISSE L-BAND tests
# # ------------------------
# print('MATISSE L-BAND')
# data_dir = '/home/toond/Documents/phd/data/IRAS0844-4431/MATISSE_L/'
# data_file = '*.fits'
# mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
# img_dir = 'MATISSE_L/data_3.5/'
# fig_dir = '/home/toond/Downloads/figs/MATISSE_L/'
#
# # FFT test
# wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_dir, plotting=True, freq_info=True,
#                                           disk_only=False, fig_dir=fig_dir, log_plotv=True, plot_vistype='vis2',
#                                           ebminv=0.0)
#
# # Monochromatic model observables test
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
#                                                    monochr=True, wavelim_lower=3.48, wavelim_upper=3.55,
#                                                    plotting=True, fig_dir=fig_dir, log_plotv=True,
#                                                    ebminv=0.0, plot_vistype='vis2')
# Chromatic model observables test
# img_dir = 'MATISSE_L'
# # cut off the wavelength range edges because data is bad there
# obsvbs_dat, container_model = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
#                                                    wavelim_lower=2.95, wavelim_upper=3.95, v2lim=1e-8,
#                                                    plotting=True, log_plotv=True, fig_dir=fig_dir,
#                                                    plot_vistype='vis2', ebminv=0.0)
# # ------------------------
