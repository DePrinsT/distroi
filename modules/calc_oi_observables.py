"""
File: calc_oi_observables.py
Author: Toon De Prins
Description: Contains functions to take MCFOST model images and convert them to interferometric observables at the
spatial frequencies of selected observational data stored in the  OIFITS format. NOTE: Assumes MCFOST images are
calculated under a single inclination/azimuthal viewing angle. Currently supports the following combinations of
obsevrables: Squared visibilities - Closure phases ; Correlated fluxes (formaly stored as visibilities) -
Closure phases.
"""

import os
import glob

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
import scipy.constants
import matplotlib.pyplot as plt

import SelectData

# setting some matplotlib parameters
plt.rc('font', size=10)  # controls default text sizes
plt.rc('axes', titlesize=12)  # fontsize of the axes title
plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
plt.rc('legend', fontsize=10)  # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title


def perform_fft(mod_dir, img_dir, plotting=False, addinfo=False, fig_dir=None,
                log_plotv=False, log_ploti=False, disk_only=False, fcorr=False, ebminv=0.0):
    """
    Function that takes an MCFOST model image .fits file, performs Fast Fourier Transform (FFT) and
    returns the image wavelength, fourier frequencies and FFT, allowing to make interpolators and calculate the
    associated interferometric observables at different (u, v) spatial frequencies in subsequent steps.
    Options allow for diagnostic plots and additional info. Note: assumes a regular rectangular pixel grid.
    """
    az, inc = 0, 0  # only one azimuthal/inc value in the .fits files to be loaded

    rad_to_mas = 206264806.2471  # radian to mili-arcsecond conversion
    mum_to_m = 1e-6  # micrometer to meter conversion

    # open the required fits file + get some header info
    hdul = fits.open(f'{mod_dir}/{img_dir}/RT.fits.gz')

    # read in the wavelength, pixelscale and 'window size' (e.g. size of axis images in radian)
    wave = hdul[0].header['WAVE']  # wavelength in micron
    pixelscale_x = abs(hdul[0].header['CDELT1']) * np.pi / 180  # loaded in degrees, converted to radian
    pixelscale_y = abs(hdul[0].header['CDELT2']) * np.pi / 180
    num_pix_x = hdul[0].header['NAXIS1']  # number of pixels
    num_pix_y = hdul[0].header['NAXIS2']
    image_size_x = pixelscale_x * num_pix_x  # image size along 1 axis directly in radian
    image_size_y = pixelscale_y * num_pix_y

    img_array = hdul[0].data  # read in image data in lam x F_lam units W m^-2
    img_array = np.flip(img_array, axis=3)  # flip y-array (to match numpy axis convention)

    img_tot = img_array[0, az, inc, :, :]  # full total flux image
    img_star = img_array[4, az, inc, :, :]
    img_disk = (img_tot - img_star)

    if disk_only:
        img = img_disk
    else:
        img = img_tot

    # image normalized by peak flux
    img_norm = img / np.max(img)

    # calculate fft normalized by total flux if the fcorr option is set to False, otherwise return the fft in
    # absolute flux, Jansky units
    if not fcorr:
        img_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img))) / np.sum(img)
    else:
        img *= (wave * 1e-6) / scipy.constants.c  # convert to F_nu in SI units (W m^-2 Hz^-1)
        img *= 1e26  # convert to Jansky
        img_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(img)))

    # extract info on the frequencies, note this is in units of 1/pixel
    # !!! NOTE: the first axis in a numpy array is the y-axis of the image, the second axis is the x-axis
    # !!! NOTE: we add a minus because the positive x- and y-axis convention in numpy
    # is the reverse of the interferometric one !!!

    w_x = -np.fft.fftshift(np.fft.fftfreq(img_fft.shape[1]))  # also use fftshift so the 0 frequency can lie
    w_y = -np.fft.fftshift(np.fft.fftfreq(img_fft.shape[0]))  # in the middle of the returned array

    # convert to uv-baseline units (i.e. 1st a conversion to 1/radian, then express in terms of baseline length
    # in MegaLambda). NOTE: strictly speaking we're expresing baseline 'length' as MegaLambda is a unit of length,
    # Note in OFITS file uv is not frequency, but u & v are baseline lengths (bad habit of community).
    # Conversion is straigthforward though. Baseline length u_x = 5 MLambda -> spatial_freq_x = 5e6 rad^-1.
    # Correspondingly, spatial_freq_x = 10e6 rad^-1 at Lambda = 1 micron -> projected baseline length
    # = 10e6 * 1 micron = 10 meter.

    uf = w_x / pixelscale_x  # spatial frequencies in units of 1/radian
    vf = w_y / pixelscale_y

    baseu = uf / 1e6  # Baseline length u in MegaLambda
    basev = vf / 1e6  # Baseline length v in MegaLambda

    step_baseu = abs(baseu[1] - baseu[0])  # retrieve the sampling steps in u baseline length
    step_basev = abs(basev[1] - basev[0])  # retrieve the sampling steps in v baseline length

    if plotting:
        # create plotting directory if it doesn't exist yet
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        if log_plotv:
            normv = 'log'
        else:
            normv = 'linear'
        if log_ploti:
            normi = 'log'
        else:
            normi = 'linear'

        # do some plotting
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        color_map = 'inferno'

        # normalized intensity plotted in pixel scale
        # also set the extent of the image when you plot it, take care that the number of pixels is even
        img_plot = ax[0][0].imshow(img_norm, cmap=color_map, norm=normi,
                                   extent=(num_pix_x / 2 + 0.5, -num_pix_x / 2 + 0.5,
                                           -num_pix_y / 2 + 0.5, num_pix_y / 2 + 0.5))
        fig.colorbar(img_plot, ax=ax[0][0], label='$I/I_{max}$', fraction=0.046, pad=0.04)
        ax[0][0].set_title('Normalized disk intensity')
        ax[0][0].set_xlabel("E-W [pixel]")
        ax[0][0].set_ylabel("S-N [pixel]")
        ax[0][0].arrow(0.90, 0.80, -0.1, 0, color='white', transform=ax[0][0].transAxes,
                       length_includes_head=True, head_width=0.015)  # draw arrows to indicate direction
        ax[0][0].text(0.78, 0.83, "E", color='white', transform=ax[0][0].transAxes)
        ax[0][0].arrow(0.90, 0.80, 0, 0.1, color='white', transform=ax[0][0].transAxes,
                       length_includes_head=True, head_width=0.015)
        ax[0][0].text(0.92, 0.90, "N", color='white', transform=ax[0][0].transAxes)
        ax[0][0].axhline(y=0, lw=0.2, color='white')
        ax[0][0].axvline(x=0, lw=0.2, color='white')

        # (squared) visibility of the direct fourier transform in pixel scale
        if not fcorr:
            vislabel = '$V^2$'
            vis = abs(img_fft) ** 2
        else:
            vislabel = r'$F_{corr}$ (Jy)'
            vis = abs(img_fft)

        v2plot = ax[0][1].imshow(vis, cmap=color_map, norm=normv,
                                 extent=(num_pix_x / 2 + 0.5, -num_pix_x / 2 + 0.5,
                                         -num_pix_y / 2 + 0.5, num_pix_y / 2 + 0.5))
        fig.colorbar(v2plot, ax=ax[0][1], label=vislabel, fraction=0.046, pad=0.04)

        ax[0][1].axhline(y=0, lw=0.2, color='black')
        ax[0][1].axvline(x=0, lw=0.2, color='black')

        ax[0][1].set_title(vislabel)
        ax[0][1].set_xlabel(r"$\leftarrow u$ [1/pixel]")
        ax[0][1].set_ylabel(r"$v \rightarrow$ [1/pixel]")

        # complex phase of the direct fourier transform in pixel scale
        phi_plot = ax[0][2].imshow(np.angle(img_fft, deg=True), cmap=color_map,
                                   extent=(num_pix_x / 2 + 0.5, -num_pix_x / 2 + 0.5,
                                           -num_pix_y / 2 + 0.5, num_pix_y / 2 + 0.5))
        fig.colorbar(phi_plot, ax=ax[0][2], label=r'$\phi$ [$^\circ$]', fraction=0.046, pad=0.04)
        ax[0][2].axhline(y=0, lw=0.2, color='black')
        ax[0][2].axvline(x=0, lw=0.2, color='black')

        ax[0][2].set_title(r'Complex phase $\phi$')
        ax[0][2].set_xlabel(r"$\leftarrow u$ [1/pixel]")
        ax[0][2].set_ylabel(r"$v \rightarrow$ [1/pixel]")

        # normalized intensity plotted in angle scale
        img_plot = ax[1][0].imshow(img_norm, cmap=color_map, aspect='auto', norm=normi,
                                   extent=((num_pix_x / 2) * pixelscale_x * rad_to_mas,
                                           (-num_pix_x / 2) * pixelscale_x * rad_to_mas,
                                           (-num_pix_y / 2) * pixelscale_y * rad_to_mas,
                                           (num_pix_y / 2) * pixelscale_y * rad_to_mas))
        fig.colorbar(img_plot, ax=ax[1][0], label='$I/I_{max}$', fraction=0.046, pad=0.04)
        ax[1][0].set_aspect(num_pix_y / num_pix_x)
        ax[1][0].set_title('Normalized disk intensity')
        ax[1][0].set_xlabel("E-W [mas]")
        ax[1][0].set_ylabel("S-N [mas]")
        ax[1][0].arrow(0.90, 0.80, -0.1, 0, color='white', transform=ax[1][0].transAxes,
                       length_includes_head=True, head_width=0.015)  # draw arrows to indicate direction
        ax[1][0].text(0.78, 0.83, "E", color='white', transform=ax[1][0].transAxes)
        ax[1][0].arrow(0.90, 0.80, 0, 0.1, color='white', transform=ax[1][0].transAxes,
                       length_includes_head=True, head_width=0.015)
        ax[1][0].text(0.92, 0.90, "N", color='white', transform=ax[1][0].transAxes)
        ax[1][0].axhline(y=0, lw=0.2, color='white')
        ax[1][0].axvline(x=0, lw=0.2, color='white')

        # (squared) visibility of the direct fourier transform in MegaLambda (baseline length) scale
        v2plot = ax[1][1].imshow(vis, cmap=color_map, norm=normv,
                                 extent=((num_pix_x / 2 + 0.5) * step_baseu, (-num_pix_x / 2 + 0.5) * step_baseu,
                                         (-num_pix_y / 2 + 0.5) * step_basev, (num_pix_y / 2 + 0.5) * step_basev))
        fig.colorbar(v2plot, ax=ax[1][1], label=vislabel, fraction=0.046, pad=0.04)
        ax[1][1].axhline(y=0, lw=0.2, color='black')
        ax[1][1].axvline(x=0, lw=0.2, color='black')

        ax[1][1].set_title(vislabel)
        ax[1][1].set_xlabel(r"$\leftarrow B_u$ [$\mathrm{M \lambda}$]")
        ax[1][1].set_ylabel(r"$B_v \rightarrow$ [$\mathrm{M \lambda}$]")

        # complex phase of the direct fourier transform in MegaLambda (baseline length) scale
        phi_plot = ax[1][2].imshow(np.angle(img_fft, deg=True), cmap=color_map,
                                   extent=((num_pix_x / 2 + 0.5) * step_baseu, (-num_pix_x / 2 + 0.5) * step_baseu,
                                           (-num_pix_y / 2 + 0.5) * step_basev, (num_pix_y / 2 + 0.5) * step_basev))
        fig.colorbar(phi_plot, ax=ax[1][2], label=r'$\phi$ [$^\circ$]', fraction=0.046, pad=0.04)
        ax[1][2].axhline(y=0, lw=0.2, color='black')
        ax[1][2].axvline(x=0, lw=0.2, color='black')
        ax[1][2].set_title(r'Complex phase $\phi$')
        ax[1][2].set_xlabel(r"$\leftarrow B_u$ [$\mathrm{M \lambda}$]")
        ax[1][2].set_ylabel(r"$B_v \rightarrow$ [$\mathrm{M \lambda}$]")

        # draw lines/cuts along which we will plot some curves
        ax[1][1].plot(np.zeros_like(basev[1:int(num_pix_y / 2) + 1]), basev[1:int(num_pix_y / 2) + 1], c='g', lw=2,
                      ls='--')
        ax[1][1].plot(baseu[1:int(num_pix_x / 2) + 1], np.zeros_like(baseu[1:int(num_pix_x / 2) + 1]), c='b', lw=2)

        ax[1][2].plot(np.zeros_like(basev[1:int(num_pix_y / 2) + 1]), basev[1:int(num_pix_y / 2) + 1], c='g', lw=2,
                      ls='--')
        ax[1][2].plot(baseu[1:int(num_pix_x / 2) + 1], np.zeros_like(baseu[1:int(num_pix_x / 2) + 1]), c='b', lw=2)

        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/fft2d_maps_{wave}mum.png", dpi=300, bbox_inches='tight')

        # Some plots of specific cuts in frequency space
        fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6))

        # Cuts of V^2 and complex phase plot in function of baseline length
        # note we cut away the point furthest along positive u-axis since it contains a strong artefact due to
        # the FFT algorithm, otherwise we move down to spatial frequency 0
        v2hor = abs(
            img_fft[int(num_pix_y / 2), 1:int(num_pix_x / 2) + 1]) ** 2  # extract squared visibility along u-axis
        phi_hor = np.angle(img_fft[int(num_pix_y / 2), 1:], deg=True)  # extract complex phase

        v2ver = abs(
            img_fft[1:int(num_pix_y / 2) + 1, int(num_pix_x / 2)]) ** 2  # extract squared visibility along u-axis
        phi_basever = np.angle(img_fft[1:, int(num_pix_x / 2)], deg=True)  # extract complex phase

        ax2[0].plot(baseu[1:int(num_pix_x / 2) + 1], v2hor, c='b', label="along u-axis", lw=0.7, zorder=1000)
        ax2[1].plot(baseu[1:], phi_hor, c='b', lw=0.7, zorder=1000)

        ax2[0].plot(basev[1:int(num_pix_y / 2) + 1], v2ver, c='g', label="along v-axis", lw=0.7, zorder=1000, ls='--')
        ax2[1].plot(basev[1:], phi_basever, c='g', lw=0.7, zorder=1000, ls='--')

        ax2[0].set_title('$V^2$ cuts')
        ax2[0].set_xlabel(r'$B$ [$\mathrm{M \lambda}$]')
        ax2[0].set_ylabel('$V^2$')
        ax2[0].axvline(x=0, c='k', lw=0.3, ls="-", zorder=0)
        if log_plotv:
            ax2[0].set_yscale("log")
            ax2[0].set_ylim(np.min(np.append(v2hor, v2ver)), 1.1)
        else:
            ax2[0].axhline(y=0, c='k', lw=0.3, ls="-", zorder=0)
            ax2[0].set_ylim(-0.1, 1.1)

        ax2[1].set_title(r'$\phi$ cuts')
        ax2[1].set_xlabel(r'$B$ [$\mathrm{M \lambda}$]')
        ax2[1].set_ylabel(r'$\phi$ [$^\circ$]')
        ax2[1].axvline(x=0, c='k', lw=0.3, ls="-", zorder=0)
        ax2[1].axhline(y=0, c='k', lw=0.3, ls="-", zorder=0)
        ax2[1].axhline(y=180, c='k', lw=0.3, ls="--", zorder=0)
        ax2[1].axhline(y=-180, c='k', lw=0.3, ls="--", zorder=0)

        ax2[0].legend()

        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/fft1d_cuts_{wave}mum.png", dpi=300, bbox_inches='tight')

    if addinfo:
        print("FREQUENCY INFORMATION IN PIXEL UNITS: \n" + "=====================================")
        print("Maximum frequency considered E-W [1/pixel]: " + str(np.max(w_x)))
        print("Maximum frequency considered S-N [1/pixel]: " + str(np.max(w_y)))
        print("This should equal the Nyquist frequency = 0.5 x 1/sampling_rate " +
              "(sampling_rate = 1 pixel in pixel units, = 1 pixelscale in physical units)")
        print("Spacing in frequency space E-W [1/pixel]: " + str(w_x[1] - w_x[0]))
        print("Spacing in frequency space South-North [1/pixel]: " + str(w_y[1] - w_y[0]))
        print("This should equal 1/window_size (i.e. = 1/(#pixels) in pixel units, " +
              "= 1/image_size in physical units)")
        print("===================================== \n")
        print("FREQUENCY INFORMATION IN ANGULAR UNITS: \n" + "=======================================")
        print("Pixel scale E-W [rad]: " + str(pixelscale_x))
        print("Pixel scale S-N [rad]: " + str(pixelscale_y))
        print("Image axes size E-W [rad]: " + str(image_size_x))
        print("Image axes size S-N [rad]: " + str(image_size_y) + "\n")
        print("Maximum frequency considered E-W [1/rad]: " + str(np.max(w_x) * 1 / pixelscale_x))
        print("Maximum frequency considered S-N [1/rad]: " + str(np.max(w_y) * 1 / pixelscale_y))
        print("Spacing in frequency space E-W [1/rad]: " + str(abs((w_x[1] - w_x[0]) * 1 / pixelscale_x)))
        print("Spacing in frequency space S-N [1/rad]: " + str(abs((w_y[1] - w_y[0]) * 1 / pixelscale_y)))
        print("-----------------------------------")
        print("Pixel scale E-W [mas]: " + str(pixelscale_x * rad_to_mas))  # 206264806.2471 mas in 1 rad
        print("Pixel scale S-N [mas]: " + str(pixelscale_y * rad_to_mas))
        print("Image axes size E-W [mas]: " + str(image_size_x * rad_to_mas))
        print("Image axes size S-N [mas]: " + str(image_size_y * rad_to_mas) + "\n")
        print("Maximum frequency considered E-W [1/mas]: " + str(np.max(w_x) * 1 / (pixelscale_x * rad_to_mas)))
        print(
            "Spacing in frequency space E-W [1/mas]: " + str(abs((w_x[1] - w_x[0]) * 1 / (pixelscale_x * rad_to_mas))))
        print("Maximum frequency considered S-N [1/mas]: " + str(np.max(w_y) * 1 / (pixelscale_y * rad_to_mas)))
        print(
            "Spacing in frequency space S-N [1/mas]: " + str(abs((w_y[1] - w_y[0]) * 1 / (pixelscale_y * rad_to_mas))))
        print("===================================== \n")
        print(r"FREQUENCY INFORMATION IN TERMS OF CORRESPONDING BASELINE LENGTH: " +
              "\n" + "===========================================================================")
        print("maximum projected baseline resolvable under current pixel sampling E-W [Mlambda]: " +
              str((np.max(w_x) * 1 / pixelscale_x) / 1e6))
        print("spacing in projected baseline length corresponding to frequency sampling E-W [Mlambda]: " +
              str(abs(((w_x[1] - w_x[0]) * 1 / pixelscale_x) / 1e6)))
        print("maximum projected baseline resolvable under current pixel sampling S-N [Mlambda]: " +
              str((np.max(w_y) * 1 / pixelscale_y) / 1e6))
        print("spacing in projected baseline length corresponding to frequency sampling S-N [Mlambda]: " +
              str(abs(((w_y[1] - w_y[0]) * 1 / pixelscale_y) / 1e6)))
        print("-----------------------------------")
        print("maximum projected baseline resolvable under current pixel sampling E-W [m]: " +
              str(((np.max(w_x) * 1 / pixelscale_x) / 1e6) * 1e6 * wave * mum_to_m))
        print("spacing in projected baseline length corresponding to frequency sampling E-W [m]: " +
              str(abs((((w_x[1] - w_x[0]) * 1 / pixelscale_x) / 1e6) * 1e6 * wave * mum_to_m)))
        print("maximum projected baseline resolvable under current pixel sampling S-N [m]: " +
              str(((np.max(w_y) * 1 / pixelscale_y) / 1e6) * 1e6 * wave * mum_to_m))
        print("spacing in projected baseline length corresponding to frequency sampling S-N [m]: " +
              str(abs((((w_y[1] - w_y[0]) * 1 / pixelscale_y) / 1e6) * 1e6 * wave * mum_to_m)))
        print("=========================================================================== \n")

    return wave, uf, vf, img_fft


def get_observation_data(data_dir, data_file, wavelim_lower=None, wavelim_upper=None, v2lim=None):
    """
    Function to retrieve observation data from OIFITS files and return it as raveled numpy arrays in a dictionary.
    This is basically a wrapper around ReadOIFITS.py and SelectData.py, but the raveled numpy arrays make things
    easier to calculate/interpolate using numpy/scipy. wave_1 and 2 are wavelength limits in meter to be applied.
    """
    # if condition because / 1e6 fails if wavelimits are False
    if wavelim_lower is not None and wavelim_upper is not None:
        oidata = SelectData.SelectData(data_dir=data_dir, data_file=data_file,
                                       wave_1=wavelim_lower / 1e6, wave_2=wavelim_upper / 1e6, lim_V2=v2lim)
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
    obsvbs_dat['vuf'] = v2ufdat  # spatial freqs in 1/rad
    obsvbs_dat['vvf'] = v2vfdat
    obsvbs_dat['vwave'] = v2wavedat  # wavelengths in meter
    obsvbs_dat['v'] = v2dat
    obsvbs_dat['verr'] = v2err  # baseline length in MegaLambda
    obsvbs_dat['vbase'] = v2base

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
                              wavelim_lower=None, wavelim_upper=None, v2lim=None, plotting=False, fig_dir=None,
                              log_plotv=False, fcorr=False, ebminv=0.0):
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
    """

    # retrieve dictionary observation observables
    obsvbs_dat = get_observation_data(data_dir, data_file, wavelim_lower=wavelim_lower,
                                      wavelim_upper=wavelim_upper, v2lim=v2lim)

    print(f"Wavelength range: {np.min(obsvbs_dat['v2wave'])}-{np.max(obsvbs_dat['v2wave'])}")

    # Return model observables in a nice dictionary format
    obsvbs_mod = {}

    if monochr:
        # perform FFT on single image and return spatial frequencies and normalized FFT
        wave, uf, vf, img_fft = perform_fft(mod_dir, img_dir, plotting=False, addinfo=False,
                                            disk_only=disk_only, fig_dir=fig_dir, fcorr=fcorr)

        #  make interpolator and calculate model observables
        interpolator = RegularGridInterpolator((vf, uf), img_fft)
        v2mod = abs(interpolator((obsvbs_dat['v2vf'], obsvbs_dat['v2uf']))) ** 2
        phi1mod = np.angle(interpolator((obsvbs_dat['t3vf1'], obsvbs_dat['t3uf1'])), deg=True)
        phi2mod = np.angle(interpolator((obsvbs_dat['t3vf2'], obsvbs_dat['t3uf2'])), deg=True)
        phi3mod = np.angle(interpolator((obsvbs_dat['t3vf3'], obsvbs_dat['t3uf3'])), deg=True)
        # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
        t3phimod = phi1mod + phi2mod - phi3mod

        obsvbs_mod['v2'] = v2mod  # fill in the model observables dictionary
        obsvbs_mod['phi1'] = phi1mod
        obsvbs_mod['phi2'] = phi2mod
        obsvbs_mod['phi3'] = phi3mod
        obsvbs_mod['t3phi'] = t3phimod
    elif not monochr:
        # list of MCFOST image subdirectories to use
        mod_img_subdirs = sorted(glob.glob(f'{img_dir}/data_*', root_dir=mod_dir))

        mod_waves = []  # model image wavelengths in meter
        img_fft_chromatic = []  # 3d array to store FFT at diferent wavelengths

        for img_subdir in mod_img_subdirs:
            # perform FFT on single image and return spatial frequencies and normalized FFT
            wave, uf, vf, img_fft = perform_fft(mod_dir, img_subdir, False, addinfo=False,
                                                disk_only=disk_only, fig_dir=fig_dir)
            mod_waves.append(wave / 1e6)
            img_fft_chromatic.append(img_fft)

        img_fft_chromatic = np.array(img_fft_chromatic)

        #  make interpolator and calculate model observables
        interpolator = RegularGridInterpolator((mod_waves, vf, uf), img_fft_chromatic)
        v2mod = abs(interpolator((obsvbs_dat['v2wave'], obsvbs_dat['v2vf'], obsvbs_dat['v2uf']))) ** 2
        phi1mod = np.angle(interpolator((obsvbs_dat['t3wave'], obsvbs_dat['t3vf1'], obsvbs_dat['t3uf1'])), deg=True)
        phi2mod = np.angle(interpolator((obsvbs_dat['t3wave'], obsvbs_dat['t3vf2'], obsvbs_dat['t3uf2'])), deg=True)
        phi3mod = np.angle(interpolator((obsvbs_dat['t3wave'], obsvbs_dat['t3vf3'], obsvbs_dat['t3uf3'])), deg=True)
        # We use the convention such that triangle ABC -> (u1,v1) = AB; (u2,v2) = BC; (u3,v3) = AC, not CA
        # This causes a minus sign shift for 3rd baseline when calculating closure phase (for real images)
        t3phimod = phi1mod + phi2mod - phi3mod

        obsvbs_mod['v2'] = v2mod  # fill in the model observables dictionary
        obsvbs_mod['phi1'] = phi1mod
        obsvbs_mod['phi2'] = phi2mod
        obsvbs_mod['phi3'] = phi3mod
        obsvbs_mod['t3phi'] = t3phimod

    if plotting:
        # create plotting directory if it doesn't exist yet
        if not os.path.exists(fig_dir):
            os.mkdir(fig_dir)

        # plot uv coverage
        if monochr:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))

            ax.scatter(obsvbs_dat['v2uf'] / 1e6, obsvbs_dat['v2vf'] / 1e6, s=1.5, color='royalblue')
            ax.scatter(-obsvbs_dat['v2uf'] / 1e6, -obsvbs_dat['v2vf'] / 1e6, s=1.5, color='royalblue')

            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_title(f'uv coverage, data selected from {wavelim_lower}-{wavelim_upper} mum')
            ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
            ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")
            plt.tight_layout()

            plt.savefig(f"{fig_dir}/monochromatic_uv_plane.png", dpi=300, bbox_inches='tight')
        elif not monochr:
            # color map for wavelengths
            color_map = 'gist_rainbow_r'

            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            fig.subplots_adjust(right=0.8)
            cax = fig.add_axes([0.85, 0.15, 0.02, 0.7])

            ax.scatter(obsvbs_dat['v2uf'] / 1e6, obsvbs_dat['v2vf'] / 1e6, c=obsvbs_dat['v2wave'] * 1e6, s=1,
                       cmap=color_map)
            sc = ax.scatter(-obsvbs_dat['v2uf'] / 1e6, -obsvbs_dat['v2vf'] / 1e6, c=obsvbs_dat['v2wave'] * 1e6, s=1,
                            cmap=color_map)
            clb = fig.colorbar(sc, cax=cax)
            clb.set_label(r'$\lambda$ ($\mu$m)', rotation=270, labelpad=15)

            ax.set_xlim(ax.get_xlim()[::-1])
            ax.set_title(f'uv coverage')
            ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
            ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

            plt.savefig(f"{fig_dir}/chromatic_uv_plane.png", dpi=300, bbox_inches='tight')

        # plot V2
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, hspace=0, height_ratios=[1, 0.3])
        ax = gs.subplots(sharex=True)

        ax[0].errorbar(obsvbs_dat['v2base'], obsvbs_dat['v2'], obsvbs_dat['v2err'], label='data', mec='royalblue',
                       marker='o', capsize=0, zorder=0, markersize=2, ls='', alpha=0.8, elinewidth=0.5)
        ax[0].scatter(obsvbs_dat['v2base'], obsvbs_mod['v2'], label=f'MCFOST model at {wave} mum', marker='o',
                      facecolor='white', edgecolor='r', s=4, alpha=0.6)
        ax[1].scatter(obsvbs_dat['v2base'], (obsvbs_mod['v2'] - obsvbs_dat['v2']) / obsvbs_dat['v2err'],
                      marker='o', facecolor='white', edgecolor='r', s=4, alpha=0.6)

        ax[0].set_ylabel('$V^2$')
        ax[0].legend()
        ax[0].set_title(f'Squared Visibilities, data selected from {wavelim_lower}-{wavelim_upper} mum')
        ax[0].tick_params(axis="x", direction="in", pad=-15)

        if log_plotv:
            ax[0].set_ylim(0.5*np.min(obsvbs_dat['v2']), 1)
            ax[0].set_yscale('log')
        else:
            ax[0].set_ylim(0, 1)

        ax[1].set_xlim(0, np.max(obsvbs_dat['v2base']) * 1.05)
        ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
        ax[1].set_xlabel(r'$B$ ($\mathrm{M \lambda}$)')
        ax[1].set_ylabel(r'error $(\sigma_{V^2})$')
        if monochr:
            plt.savefig(f"{fig_dir}/monochromatic_V2.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{fig_dir}/chromatic_V2.png", dpi=300, bbox_inches='tight')

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
        ax[0].set_title(f'Closure Phases, data selected from {wavelim_lower}-{wavelim_upper} mum')
        ax[0].tick_params(axis="x", direction="in", pad=-15)

        ax[1].set_xlim(0, np.max(obsvbs_dat['t3bmax']) * 1.05)
        ax[1].axhline(y=0, c='k', ls='--', lw=1, zorder=0)
        ax[1].set_xlabel(r'$B_{max}$ ($\mathrm{M \lambda}$)')
        ax[1].set_ylabel(r'error $(\sigma_{\phi_{CP}})$')
        if monochr:
            plt.savefig(f"{fig_dir}/monochromatic_t3phi.png", dpi=300, bbox_inches='tight')
        else:
            plt.savefig(f"{fig_dir}/chromatic_t3phi.png", dpi=300, bbox_inches='tight')

    return obsvbs_dat, obsvbs_mod


if __name__ == '__main__':
    print('TESTS')
    print(np.linspace(7.9, 13.1, 6))

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
    # wave, uf, vf, img_fft = perform_fft(mod_dir, img_dir, plotting=True, addinfo=True,
    #                                     disk_only=True, fig_dir=fig_dir)
    #
    # # Monochromatic model observables test
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=1.63, wavelim_upper=1.65,
    #                                                    plotting=True, fig_dir=fig_dir)
    #
    # # Chromatic model observables test
    # img_dir = 'PIONIER'
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
    #                                                    wavelim_lower=None, wavelim_upper=None, plotting=True,
    #                                                    fig_dir=fig_dir)
    # # ------------------------

    # # GRAVITY tests
    # # ------------------------
    # print('GRAVITY TESTS')
    # data_dir = '/home/toond/Documents/phd/data/IRAS0844-4431/GRAVITY/'
    # data_file = '*1.fits'
    # mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
    # img_dir = 'GRAVITY/data_2.2/'
    # fig_dir = '/home/toond/Downloads/figs/GRAVITY/'
    # #
    # # FFT test
    # wave, uf, vf, img_fft = perform_fft(mod_dir, img_dir, plotting=True, addinfo=True,
    #                                     disk_only=False, fig_dir=fig_dir)
    # #
    # # Monochromatic model observables test
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=2.1, wavelim_upper=2.3,
    #                                                    plotting=True, fig_dir=fig_dir)
    #
    # # Chromatic model observables test
    # img_dir = 'GRAVITY'
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
    #                                                    wavelim_lower=None, wavelim_upper=None, plotting=True,
    #                                                    fig_dir=fig_dir)
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
    # wave, uf, vf, img_fft = perform_fft(mod_dir, img_dir, plotting=True, addinfo=True,
    #                                     disk_only=False, fig_dir=fig_dir, log_ploti=True, log_plotv=True)
    # #
    # # Monochromatic model observables test
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
    #                                                    monochr=True, wavelim_lower=3.48, wavelim_upper=3.55,
    #                                                    plotting=True, fig_dir=fig_dir, log_plotv=True)
    # # Chromatic model observables test
    # img_dir = 'MATISSE_L'
    # # cut off the wavelength range edges because data is bad there
    # obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
    #                                                    wavelim_lower=2.95, wavelim_upper=3.95, v2lim=1e-8,
    #                                                    plotting=True, log_plotv=True, fig_dir=fig_dir)
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
    wave, uf, vf, img_fft = perform_fft(mod_dir, img_dir, plotting=True, addinfo=True,
                                        disk_only=False, fig_dir=fig_dir, fcorr=True, log_ploti=True, log_plotv=True)
    #
    # Monochromatic model observables test
    obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir,
                                                       monochr=True, wavelim_lower=9.75, wavelim_upper=10.20,
                                                       plotting=True, fig_dir=fig_dir, log_plotv=True)
    # Chromatic model observables test
    img_dir = 'MATISSE_N'
    # cut off the wavelength range edges because data is bad there
    obsvbs_dat, obsvbs_mod = calc_model_oi_observables(data_dir, data_file, mod_dir, img_dir, monochr=False,
                                                       wavelim_lower=8.5, wavelim_upper=12.0,
                                                       plotting=True, log_plotv=True, fig_dir=fig_dir)
    # ------------------------
