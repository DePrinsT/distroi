"""
File: oi_observables.py
Author: Toon De Prins
Description: Contains functions to take MCFOST model images and convert them to interferometric observables at the
spatial frequencies of selected observational data stored in the  OIFITS format. NOTE: Assumes MCFOST images are
calculated under a single inclination/azimuthal viewing angle. Currently supports the following combinations of
obsevrables: Squared visibilities - Closure phases ; Correlated fluxes (formaly stored as visibilities) -
Closure phases.
"""
#TODO: make this work with the new Image class in image_fft.py
import constants
from modules.external import SelectData

import os
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
import matplotlib_settings

matplotlib_settings.set_matplotlib_params()  # set project matplotlib parameters


def get_observation_data(data_dir, data_file, wavelim_lower=None, wavelim_upper=None, v2lim=None):
    """
    Function to retrieve observation data from OIFITS files and return it as raveled numpy arrays in a dictionary.
    This is basically a wrapper around ReadOIFITS.py and SelectData.py, but the raveled numpy arrays make things
    easier to calculate/interpolate using numpy/scipy. wave_1 and 2 are wavelength limits in meter to be applied.
    """
    # if condition because * constants.MICRON2M fails if wavelimits are False
    if wavelim_lower is not None and wavelim_upper is not None:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file,
                                       wave_1=wavelim_lower * constants.MICRON2M,
                                       wave_2=wavelim_upper * constants.MICRON2M, lim_V2=v2lim)
    else:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file, lim_V2=v2lim)

    obsvbs_dat = {}

    # arrays to store all necessary variables in a 1d array
    vufdat = []  # for visibility tables
    vvfdat = []
    vwavedat = []
    vdat = []
    verr = []

    v2ufdat = []  # for squared visibility tables
    v2vfdat = []
    v2wavedat = []
    v2dat = []
    v2err = []

    uf1dat = []  # for closure phase tables
    vf1dat = []
    uf2dat = []
    vf2dat = []
    t3wavedat = []
    t3phidat = []
    t3phierr = []

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
    obsvbs_dat['vuf'] = vufdat  # spatial freqs in 1/rad
    obsvbs_dat['vvf'] = vvfdat
    obsvbs_dat['vwave'] = vwavedat  # wavelengths in meter
    obsvbs_dat['v'] = vdat
    obsvbs_dat['verr'] = verr  # baseline length in MegaLambda
    obsvbs_dat['vbase'] = vbase

    obsvbs_dat['v2uf'] = v2ufdat  # spatial freqs in 1/rad
    obsvbs_dat['v2vf'] = v2vfdat
    obsvbs_dat['v2wave'] = v2wavedat  # wavelengths in meter
    obsvbs_dat['v2'] = v2dat
    obsvbs_dat['v2err'] = v2err  # baseline length in MegaLambda
    obsvbs_dat['v2base'] = v2base

    obsvbs_dat['t3uf1'] = uf1dat  # spatial freqs in 1/rad
    obsvbs_dat['t3vf1'] = vf1dat
    obsvbs_dat['t3uf2'] = uf2dat
    obsvbs_dat['t3vf2'] = vf2dat
    obsvbs_dat['t3uf3'] = uf3dat
    obsvbs_dat['t3vf3'] = vf3dat
    obsvbs_dat['t3wave'] = t3wavedat  # wavelengths in meter
    obsvbs_dat['t3phi'] = t3phidat  # closure phases in degrees
    obsvbs_dat['t3phierr'] = t3phierr
    obsvbs_dat['t3bmax'] = t3bmax  # max baseline lengths in MegaLambda

    return obsvbs_dat


def calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False, disk_only=False,
                              fcorr=False, ebminv=0.0, wavelim_lower=None, wavelim_upper=None, v2lim=None,
                              plotting=False,
                              fig_dir=None, log_plotv=False, plot_vistype='vis2'):
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

    # retrieve dictionary observation observables
    obsvbs_dat = get_observation_data(data_dir, data_file, wavelim_lower=wavelim_lower,
                                      wavelim_upper=wavelim_upper, v2lim=v2lim)

    # Return model observables in a nice dictionary format
    obsvbs_mod = {}

    if monochr:
        # perform FFT on single image and return spatial frequencies and normalized FFT
        wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_dir, disk_only=disk_only, ebminv=ebminv)

        #  make interpolator on absolute image (in Jansky) and calculate model observables from that
        interpolator = RegularGridInterpolator((vf, uf), img_fft)
        if not fcorr:
            vmod = abs(interpolator((obsvbs_dat['vvf'], obsvbs_dat['vuf'])) / ftot)  # visibilities normalized
        else:
            vmod = abs(interpolator((obsvbs_dat['vvf'], obsvbs_dat['vuf'])))  # visibilities in correlated flux
        v2mod = abs(interpolator((obsvbs_dat['v2vf'], obsvbs_dat['v2uf'])) / ftot) ** 2  # squared visibilities
        phi1mod = np.angle(interpolator((obsvbs_dat['t3vf1'], obsvbs_dat['t3uf1'])), deg=True)
        phi2mod = np.angle(interpolator((obsvbs_dat['t3vf2'], obsvbs_dat['t3uf2'])), deg=True)
        phi3mod = np.angle(interpolator((obsvbs_dat['t3vf3'], obsvbs_dat['t3uf3'])), deg=True)
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
            # perform FFT on single image and return spatial frequencies and normalized FFT
            wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_subdir, disk_only=disk_only, ebminv=ebminv)
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
            vmod = abs(interpolator((obsvbs_dat['vwave'], obsvbs_dat['vvf'], obsvbs_dat['vuf'])))
        else:
            vmod = abs(interpolator_norm((obsvbs_dat['vwave'], obsvbs_dat['vvf'], obsvbs_dat['vuf'])))

        v2mod = abs(interpolator_norm((obsvbs_dat['v2wave'], obsvbs_dat['v2vf'], obsvbs_dat['v2uf']))) ** 2
        phi1mod = np.angle(interpolator_norm((obsvbs_dat['t3wave'], obsvbs_dat['t3vf1'], obsvbs_dat['t3uf1'])),
                           deg=True)
        phi2mod = np.angle(interpolator_norm((obsvbs_dat['t3wave'], obsvbs_dat['t3vf2'], obsvbs_dat['t3uf2'])),
                           deg=True)
        phi3mod = np.angle(interpolator_norm((obsvbs_dat['t3wave'], obsvbs_dat['t3vf3'], obsvbs_dat['t3uf3'])),
                           deg=True)
        # We use the convention such that if we have triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
        t3phimod = phi1mod + phi2mod - phi3mod

    obsvbs_mod['v'] = vmod  # fill in the model observables dictionary
    obsvbs_mod['v2'] = v2mod
    obsvbs_mod['phi1'] = phi1mod
    obsvbs_mod['phi2'] = phi2mod
    obsvbs_mod['phi3'] = phi3mod
    obsvbs_mod['t3phi'] = t3phimod

    if plotting:
        plot_data_vs_model(obsvbs_dat, obsvbs_mod, fig_dir=fig_dir, log_plotv=log_plotv, plot_vistype=plot_vistype)

    return obsvbs_dat, obsvbs_mod


def plot_data_vs_model(obsvbs_dat, obsvbs_mod, fig_dir=None, log_plotv=False, plot_vistype='vis2'):
    # create plotting directory if it doesn't exist yet
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    # set spatial frequencies, visibilities and plotting label based on specified option
    if plot_vistype == 'vis2':
        ufdata = obsvbs_dat['v2uf']
        vfdata = obsvbs_dat['v2vf']
        vismod = obsvbs_mod['v2']
        visdata = obsvbs_dat['v2']
        viserrdata = obsvbs_dat['v2err']
        wavedata = obsvbs_dat['v2wave']
        basedata = obsvbs_dat['v2base']
        vislabel = '$V^2$'
    elif plot_vistype == 'vis':
        ufdata = obsvbs_dat['vuf']
        vfdata = obsvbs_dat['vvf']
        vismod = obsvbs_mod['v']
        wavedata = obsvbs_dat['vwave']
        visdata = obsvbs_dat['v']
        viserrdata = obsvbs_dat['verr']
        basedata = obsvbs_dat['vbase']
        vislabel = '$V$'
    elif plot_vistype == 'fcorr':
        ufdata = obsvbs_dat['vuf']
        vfdata = obsvbs_dat['vvf']
        vismod = obsvbs_mod['v']
        wavedata = obsvbs_dat['vwave']
        visdata = obsvbs_dat['v']
        viserrdata = obsvbs_dat['verr']
        basedata = obsvbs_dat['vbase']
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
    ax[0].scatter(basedata, vismod, label=f'MCFOST model at {wave} mum', marker='o',
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

    ax[0].errorbar(obsvbs_dat['t3bmax'], obsvbs_dat['t3phi'], obsvbs_dat['t3phierr'], label='data', mec='royalblue',
                   marker='o', capsize=0, zorder=0, markersize=2, ls='', alpha=0.8, elinewidth=0.5)
    ax[0].scatter(obsvbs_dat['t3bmax'], obsvbs_mod['t3phi'], label=f'MCFOST model at {wave} mum', marker='o',
                  facecolor='white', edgecolor='r', s=4, alpha=0.6)
    ax[1].scatter(obsvbs_dat['t3bmax'], (obsvbs_mod['t3phi'] - obsvbs_dat['t3phi']) / obsvbs_dat['t3phierr'],
                  marker='o', facecolor='white', edgecolor='r', s=4, alpha=0.6)

    ax[0].set_ylabel(r'$\phi_{CP}$ ($^\circ$)')
    ax[0].legend()
    ax[0].set_title(f'Closure Phases')
    ax[0].tick_params(axis="x", direction="in", pad=-15)

    ax[1].set_xlim(0, np.max(obsvbs_dat['t3bmax']) * 1.05)
    ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
    ax[1].set_xlabel(r'$B_{max}$ ($\mathrm{M \lambda}$)')
    ax[1].set_ylabel(r'error $(\sigma_{\phi_{CP}})$')

    plt.savefig(f"{fig_dir}/closure_phases.png", dpi=300, bbox_inches='tight')

    return


if __name__ == '__main__':
    print('TESTS')

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
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=1.63, wavelim_upper=1.65,
    #                                                    plotting=True, fig_dir=fig_dir, plot_vistype='vis2', ebminv=0.0)
    #
    # # Chromatic model observables test
    # img_dir = 'PIONIER'
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
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
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=2.1, wavelim_upper=2.3,
    #                                                    plotting=True, fig_dir=fig_dir, ebminv=0.0,
    #                                                    plot_vistype='vis2')
    #
    # # Chromatic model observables test
    # img_dir = 'GRAVITY'
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
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
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=3.48, wavelim_upper=3.55,
    #                                                    plotting=True, fig_dir=fig_dir, log_plotv=True,
    #                                                    ebminv=0.0, plot_vistype='vis2')
    # Chromatic model observables test
    # img_dir = 'MATISSE_L'
    # # cut off the wavelength range edges because data is bad there
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
    #                                                    wavelim_lower=2.95, wavelim_upper=3.95, v2lim=1e-8,
    #                                                    plotting=True, log_plotv=True, fig_dir=fig_dir,
    #                                                    plot_vistype='vis2', ebminv=0.0)
    # # ------------------------

    # MATISSE N-BAND tests
    # ------------------------
    print('MATISSE N-BAND')
    data_dir = '/home/toond/Documents/phd/data/IRAS0844-4431/MATISSE_N/'
    data_file = '*.fits'
    mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
    img_dir = 'MATISSE_N/data_10.0/'
    fig_dir = '/home/toond/Downloads/figs/MATISSE_N/'

    # FFT test
    wave, uf, vf, ftot, img_fft = perform_fft(mod_dir, img_dir, plotting=True, freq_info=True,
                                              disk_only=False, fig_dir=fig_dir, log_plotv=True, plot_vistype='fcorr',
                                              ebminv=1.4)

    # # Monochromatic model observables test
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=9.75, wavelim_upper=10.20,
    #                                                    plotting=True, fig_dir=fig_dir, log_plotv=True, fcorr=True,
    #                                                    plot_vistype='fcorr', ebminv=1.4)
    # Chromatic model observables test
    img_dir = 'MATISSE_N'
    # cut off the wavelength range edges because data is bad there
    obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
                                                       wavelim_lower=8.5, wavelim_upper=12.0,
                                                       plotting=True, log_plotv=True, fig_dir=fig_dir, fcorr=True,
                                                       plot_vistype='fcorr', ebminv=1.4)
    # # ------------------------
