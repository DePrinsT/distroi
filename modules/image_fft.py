"""
Defines class and corresponding methods to load in and handle model images and their fast fourier transform.
"""
import os

import constants
import sed_analysis

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib_settings

matplotlib_settings.set_matplotlib_params()  # set project matplotlib parameters


class ImageFFT:
    """
    Class containing information on a model image and its fast fourier transform. While default options are tuned to
    MCFOST model disk images, it can easily be generalized by adding an extra option for 'read_method' in the
    initializer and defining a corresponding reader function analogous to 'read_from_mcfost'.
    """

    def __init__(self, img_path, read_method='mcfost', disk_only=False):
        """
        Initializer that sets the instance's attributes to default values. Assigns all properties that are expected to
        be contained in a singular instance in order to fully describe both the image and its fast fourier transform
        (FFT). Note that these properties are expected if other class methods are to work. Uses class methods like
        read_mcfost_image to read in a model image and assign useful values to the properties.

        Parameters:
            img_path (str): path to the model image file to read in
            read_method (str): which reader method to use to read in a model image. Defaults to 'mcfost' to read
                in MCFOST output images stored in .fits.gz files.
        Properties:
            wave (float): image wavelength in micron
            pixelscale_x (float): pixelscale in radian in x direction
            pixelscale_y (float): pixelscale in radian in y direction
            num_pix_x (float): amount of pixels in x direction
            num_pix_y (float): amount of pixels in y direction
            img (array): 2D numpy array containing the image in Jansky. 1st index = image y-axis,
                2nd index = image x-axis
            ftot (float): total flux in Jansky
            img_fft (array): numpy FFT of self.img in absolute flux (i.e. in Jansky)
            w_x (array): numpy FFT x-axis frequencies returned by np.fft.fftshift(np.fft.fftfreq()), i.e. units 1/pixel
            w_y (array): analogous to w_x but for the y-axis
            uf (array): FFT spatial frequencies in 1/radian, i.e. uf = w_x/pixelscale_x
            vf (array): FFT spatial frequencies in 1/radian, i.e. vf = w_y/pixelscale_y
        """
        self.wave = None  # image wavelength in micron

        self.pixelscale_x = None  # pixelscale in radian in x direction
        self.pixelscale_y = None
        self.num_pix_x = None  # number of pixels in x direction
        self.num_pix_y = None

        self.img = None  # 2d numpy array containing the flux, 1st index = image y-axis, 2nd index = image x-axis
        self.ftot = None  # total flux in Jy

        self.img_fft = None  # numpy FFT of self.img in absolute flux (Jansky)
        self.w_x = None  # numpy FFT frequencies returned by np.fft.fftshift(np.fft.fftfreq()), i.e. units 1/pixel
        self.w_y = None
        self.uf = None  # FFT spatial frequencies in 1/radian, i.e. uf = w_x/pixelscale_x; vf = w_y/pixelscale_y
        self.vf = None

        # choose reader function and initialize
        if read_method == 'mcfost':
            self.read_from_mcfost(img_path, disk_only=disk_only)

    def read_from_mcfost(self, img_path, disk_only=False):
        """
        Initializes a DiskImage instance by reading in an MCFOST image run output file.

        Parameters:
            img_path (str): path to an MCFOST output RT.fits.gz image file
            disk_only (bool): set to True if you only want to load in the flux from the disk
        """
        az, inc = 0, 0  # only load the first azimuthal/inc value image in the .fits file

        # open the required fits file + get some header info
        hdul = fits.open(img_path)

        # read in the wavelength, pixelscale and 'window size' (e.g. size of axis images in radian)
        self.wave = hdul[0].header['WAVE']  # wavelength in micron
        self.pixelscale_x = abs(hdul[0].header['CDELT1']) * constants.DEG2RAD  # loaded in degrees, converted to radian
        self.pixelscale_y = abs(hdul[0].header['CDELT2']) * constants.DEG2RAD
        self.num_pix_x = hdul[0].header['NAXIS1']  # number of pixels
        self.num_pix_y = hdul[0].header['NAXIS2']

        img_array = hdul[0].data  # read in image data in lam x F_lam units W m^-2
        img_array = np.flip(img_array, axis=3)  # flip y-array (to match numpy axis convention)

        img_tot = img_array[0, az, inc, :, :]  # full total flux image
        img_star = img_array[4, az, inc, :, :]
        img_disk = (img_tot - img_star)

        if disk_only:
            self.img = img_disk
        else:
            self.img = img_tot

        # calculate fft in Jy units
        self.img *= ((self.wave * constants.MICRON2M) /
                     constants.SPEED_OF_LIGHT)  # convert to F_nu in SI units (W m^-2 Hz^-1)
        self.img *= constants.WATT_PER_METER2_HZ_2JY  # convert image to Jansky
        self.ftot = np.sum(self.img)  # total flux in Jansky
        self.img_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(self.img)))  # complex fft in Jansky

        # extract info on the frequencies, note this is in units of 1/pixel
        # !!! NOTE: the first axis in a numpy array is the y-axis of the image, the second axis is the x-axis
        # !!! NOTE: we add a minus because the positive x- and y-axis convention in numpy
        # is the reverse of the interferometric one !!!

        self.w_x = -np.fft.fftshift(np.fft.fftfreq(self.img_fft.shape[1]))  # also use fftshift so the 0 frequency can
        self.w_y = -np.fft.fftshift(np.fft.fftfreq(self.img_fft.shape[0]))  # lie in the middle of the returned array

        self.uf = self.w_x / self.pixelscale_x  # spatial frequencies in units of 1/radian
        self.vf = self.w_y / self.pixelscale_y

    def redden(self, ebminv, reddening_law_path=constants.PROJECT_ROOT + '/utils/ISM_reddening'
                                                                         '/ISMreddening_law_Cardelli1989.dat'):
        """
        Further reddens the model image according to the approriate E(B-V) and a corresponding reddening law.

        Parameters:
            ebminv (Float):
            reddening_law_path (str): path to reddening law to be used. Defaults to ISM reddening law by
                Cardelli (1989) in the package's utils/ISM_reddening folder. See this file for the formatting of
                the reddening law.
        Properties affected:
            self.img
            self.img_fft
            self.ftot
        """
        self.img = sed_analysis.redden_flux(self.wave, self.img,
                                            reddening_law_path, ebminv)  # apply ISM reddening to the image
        if self.img_fft is not None:
            self.img_fft = sed_analysis.redden_flux(self.wave, self.img_fft,
                                                    reddening_law_path, ebminv)  # apply ISM reddening to the fft
        self.ftot = np.sum(self.img)  # recalculate total flux

    def freq_info(self):
        """
        Prints out information on both the frequency domain/sampling and the corresponding projected baselines.
        """
        image_size_x = self.pixelscale_x * self.num_pix_x
        image_size_y = self.pixelscale_y * self.num_pix_y

        print("FREQUENCY INFORMATION IN PIXEL UNITS: \n" + "=====================================")
        print("Maximum frequency considered E-W [1/pixel]: " + str(np.max(self.w_x)))
        print("Maximum frequency considered S-N [1/pixel]: " + str(np.max(self.w_y)))
        print("This should equal the Nyquist frequency = 0.5 x 1/sampling_rate " +
              "(sampling_rate = 1 pixel in pixel units, = 1 pixelscale in physical units)")
        print("Spacing in frequency space E-W [1/pixel]: " + str(abs(self.w_x[1] - self.w_x[0])))
        print("Spacing in frequency space South-North [1/pixel]: " + str(abs(self.w_y[1] - self.w_y[0])))
        print("This should equal 1/window_size (i.e. = 1/(#pixels) in pixel units, " +
              "= 1/image_size in physical units)")
        print("===================================== \n")
        print("FREQUENCY INFORMATION IN ANGULAR UNITS: \n" + "=======================================")
        print("Pixel scale E-W [rad]: " + str(self.pixelscale_x))
        print("Pixel scale S-N [rad]: " + str(self.pixelscale_y))
        print("Image axes size E-W [rad]: " + str(image_size_x))
        print("Image axes size S-N [rad]: " + str(image_size_y) + "\n")
        print("Maximum frequency considered E-W [1/rad]: " + str(np.max(self.w_x) * 1 / self.pixelscale_x))
        print("Maximum frequency considered S-N [1/rad]: " + str(np.max(self.w_y) * 1 / self.pixelscale_y))
        print("Spacing in frequency space E-W [1/rad]: " + str(abs((self.w_x[1] - self.w_x[0]) * 1 /
                                                                   self.pixelscale_x)))
        print("Spacing in frequency space S-N [1/rad]: " + str(abs((self.w_y[1] - self.w_y[0]) * 1 /
                                                                   self.pixelscale_y)))
        print("-----------------------------------")
        print("Pixel scale E-W [mas]: " + str(self.pixelscale_x * constants.RAD2MAS))  # 206264806.2471 mas in 1 rad
        print("Pixel scale S-N [mas]: " + str(self.pixelscale_y * constants.RAD2MAS))
        print("Image axes size E-W [mas]: " + str(image_size_x * constants.RAD2MAS))
        print("Image axes size S-N [mas]: " + str(image_size_y * constants.RAD2MAS) + "\n")
        print("Maximum frequency considered E-W [1/mas]: " + str(np.max(self.w_x) * 1 / (self.pixelscale_x *
                                                                                         constants.RAD2MAS)))
        print("Spacing in frequency space E-W [1/mas]: " +
              str(abs((self.w_x[1] - self.w_x[0]) * 1 / (self.pixelscale_x * constants.RAD2MAS))))
        print("Maximum frequency considered S-N [1/mas]: " + str(np.max(self.w_y) * 1 / (self.pixelscale_y *
                                                                                         constants.RAD2MAS)))
        print("Spacing in frequency space S-N [1/mas]: " +
              str(abs((self.w_y[1] - self.w_y[0]) * 1 / (self.pixelscale_y * constants.RAD2MAS))))
        print("===================================== \n")
        print(r"FREQUENCY INFORMATION IN TERMS OF CORRESPONDING BASELINE LENGTH: " +
              "\n" + "===========================================================================")
        print("maximum projected baseline resolvable under current pixel sampling E-W [Mlambda]: " +
              str((np.max(self.w_x) * 1 / self.pixelscale_x) / 1e6))
        print("spacing in projected baseline length corresponding to frequency sampling E-W [Mlambda]: " +
              str(abs(((self.w_x[1] - self.w_x[0]) * 1 / self.pixelscale_x) / 1e6)))
        print("maximum projected baseline resolvable under current pixel sampling S-N [Mlambda]: " +
              str((np.max(self.w_y) * 1 / self.pixelscale_y) / 1e6))
        print("spacing in projected baseline length corresponding to frequency sampling S-N [Mlambda]: " +
              str(abs(((self.w_y[1] - self.w_y[0]) * 1 / self.pixelscale_y) / 1e6)))
        print("-----------------------------------")
        print("maximum projected baseline resolvable under current pixel sampling E-W [m]: " +
              str(((np.max(self.w_x) * 1 / self.pixelscale_x) / 1e6) * 1e6 * self.wave * constants.MICRON2M))
        print("spacing in projected baseline length corresponding to frequency sampling E-W [m]: " +
              str(abs((((self.w_x[1] - self.w_x[0]) * 1 / self.pixelscale_x) / 1e6) * 1e6 *
                      self.wave * constants.MICRON2M)))
        print("maximum projected baseline resolvable under current pixel sampling S-N [m]: " +
              str(((np.max(self.w_y) * 1 / self.pixelscale_y) / 1e6) * 1e6 * self.wave * constants.MICRON2M))
        print("spacing in projected baseline length corresponding to frequency sampling S-N [m]: " +
              str(abs((((self.w_y[1] - self.w_y[0]) * 1 / self.pixelscale_y) / 1e6) * 1e6 *
                      self.wave * constants.MICRON2M)))
        print("=========================================================================== \n")

    def diagnostic_plot(self, fig_dir, plot_vistype='vis2', log_plotv=False, log_ploti=False):
        """
        Makes diagnostic plots showing both the model image and the FFT of the DiskImage objct.
        Parameters:
            fig_dir (str): path to folder where plots are saved0
            plot_vistype (str): type of visibility to be plotted, options are 'vis2', 'vis' or 'fcorr'
            log_plotv (bool): set to True to plot the visibilities on a logarithmic scale
            log_ploti (bool): set to True to plot the model image on a logarithmic scale
        """
        baseu = self.uf / 1e6  # Baseline length u in MegaLambda
        basev = self.vf / 1e6  # Baseline length v in MegaLambda

        step_baseu = abs(baseu[1] - baseu[0])  # retrieve the sampling steps in u baseline length
        step_basev = abs(basev[1] - basev[0])  # retrieve the sampling steps in v baseline length

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
        img_plot = ax[0][0].imshow(self.img, cmap=color_map, norm=normi,
                                   extent=(self.num_pix_x / 2 + 0.5, -self.num_pix_x / 2 + 0.5,
                                           -self.num_pix_y / 2 + 0.5, self.num_pix_y / 2 + 0.5))
        fig.colorbar(img_plot, ax=ax[0][0], label='$I$ (Jy/pixel)', fraction=0.046, pad=0.04)
        ax[0][0].set_title('Disk intensity')
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

        # set the (squared) visibility of the discrete fourier transform
        if plot_vistype == 'vis2':
            vislabel = '$V^2$'
            vis = abs(self.img_fft / self.ftot) ** 2
        elif plot_vistype == 'vis':
            vislabel = '$V$'
            vis = abs(self.img_fft / self.ftot)
        elif plot_vistype == 'fcorr':
            vislabel = r'$F_{corr}$ (Jy)'
            vis = abs(self.img_fft)
        else:
            print('vislabel not recognized, using vis2 as default')
            vislabel = '$V^2$'
            vis = abs(self.img_fft / self.ftot) ** 2

        # set the complex phase
        cphi = np.angle(self.img_fft, deg=True)

        v2plot = ax[0][1].imshow(vis, cmap=color_map, norm=normv,
                                 extent=(self.num_pix_x / 2 + 0.5, -self.num_pix_x / 2 + 0.5,
                                         -self.num_pix_y / 2 + 0.5, self.num_pix_y / 2 + 0.5))
        fig.colorbar(v2plot, ax=ax[0][1], label=vislabel, fraction=0.046, pad=0.04)

        ax[0][1].axhline(y=0, lw=0.2, color='black')
        ax[0][1].axvline(x=0, lw=0.2, color='black')

        ax[0][1].set_title(vislabel)
        ax[0][1].set_xlabel(r"$\leftarrow u$ [1/pixel]")
        ax[0][1].set_ylabel(r"$v \rightarrow$ [1/pixel]")

        # complex phase of the direct fourier transform in pixel scale
        phi_plot = ax[0][2].imshow(cphi, cmap=color_map,
                                   extent=(self.num_pix_x / 2 + 0.5, -self.num_pix_x / 2 + 0.5,
                                           -self.num_pix_y / 2 + 0.5, self.num_pix_y / 2 + 0.5))
        fig.colorbar(phi_plot, ax=ax[0][2], label=r'$\phi$ [$^\circ$]', fraction=0.046, pad=0.04)
        ax[0][2].axhline(y=0, lw=0.2, color='black')
        ax[0][2].axvline(x=0, lw=0.2, color='black')

        ax[0][2].set_title(r'Complex phase $\phi$')
        ax[0][2].set_xlabel(r"$\leftarrow u$ [1/pixel]")
        ax[0][2].set_ylabel(r"$v \rightarrow$ [1/pixel]")

        # normalized intensity plotted in angle scale
        img_plot = ax[1][0].imshow(self.img, cmap=color_map, aspect='auto', norm=normi,
                                   extent=((self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                                           (-self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                                           (-self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS,
                                           (self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS))
        fig.colorbar(img_plot, ax=ax[1][0], label='$I$ (Jy/pixel)', fraction=0.046, pad=0.04)
        ax[1][0].set_aspect(self.num_pix_y / self.num_pix_x)
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
                                 extent=((self.num_pix_x / 2 + 0.5) * step_baseu, (-self.num_pix_x / 2 + 0.5) *
                                         step_baseu,
                                         (-self.num_pix_y / 2 + 0.5) * step_basev, (self.num_pix_y / 2 + 0.5) *
                                         step_basev))
        fig.colorbar(v2plot, ax=ax[1][1], label=vislabel, fraction=0.046, pad=0.04)
        ax[1][1].axhline(y=0, lw=0.2, color='black')
        ax[1][1].axvline(x=0, lw=0.2, color='black')

        ax[1][1].set_title(vislabel)
        ax[1][1].set_xlabel(r"$\leftarrow B_u$ [$\mathrm{M \lambda}$]")
        ax[1][1].set_ylabel(r"$B_v \rightarrow$ [$\mathrm{M \lambda}$]")

        # complex phase of the direct fourier transform in MegaLambda (baseline length) scale
        phi_plot = ax[1][2].imshow(cphi, cmap=color_map,
                                   extent=((self.num_pix_x / 2 + 0.5) * step_baseu, (-self.num_pix_x / 2 + 0.5) *
                                           step_baseu,
                                           (-self.num_pix_y / 2 + 0.5) * step_basev, (self.num_pix_y / 2 + 0.5) *
                                           step_basev))
        fig.colorbar(phi_plot, ax=ax[1][2], label=r'$\phi$ [$^\circ$]', fraction=0.046, pad=0.04)
        ax[1][2].axhline(y=0, lw=0.2, color='black')
        ax[1][2].axvline(x=0, lw=0.2, color='black')
        ax[1][2].set_title(r'Complex phase $\phi$')
        ax[1][2].set_xlabel(r"$\leftarrow B_u$ [$\mathrm{M \lambda}$]")
        ax[1][2].set_ylabel(r"$B_v \rightarrow$ [$\mathrm{M \lambda}$]")

        # draw lines/cuts along which we will plot some curves
        ax[1][1].plot(np.zeros_like(basev[1:int(self.num_pix_y / 2) + 1]), basev[1:int(self.num_pix_y / 2) + 1],
                      c='g', lw=2,
                      ls='--')
        ax[1][1].plot(baseu[1:int(self.num_pix_x / 2) + 1], np.zeros_like(baseu[1:int(self.num_pix_x / 2) + 1]),
                      c='b', lw=2)

        ax[1][2].plot(np.zeros_like(basev[1:int(self.num_pix_y / 2) + 1]), basev[1:int(self.num_pix_y / 2) + 1],
                      c='g', lw=2,
                      ls='--')
        ax[1][2].plot(baseu[1:int(self.num_pix_x / 2) + 1], np.zeros_like(baseu[1:int(self.num_pix_x / 2) + 1]),
                      c='b', lw=2)

        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(f"{fig_dir}/fft2d_maps_{self.wave}mum.png", dpi=300, bbox_inches='tight')

        # Some plots of specific cuts in frequency space
        fig2, ax2 = plt.subplots(2, 1, figsize=(6, 6))

        # Cuts of visibility and complex phase plot in function of baseline length
        # note we cut away the point furthest along positive u-axis since it contains a strong artefact due to
        # the FFT algorithm, otherwise we move down to spatial frequency 0
        vhor = vis[int(self.num_pix_y / 2), 1:int(self.num_pix_x / 2) + 1]  # extract (squared) visibility along u-axis
        phi_hor = cphi[int(self.num_pix_y / 2), 1:]  # extract complex phase

        vver = vis[1:int(self.num_pix_y / 2) + 1, int(self.num_pix_x / 2)]  # extract (squared) visibility along u-axis
        phi_ver = cphi[1:, int(self.num_pix_x / 2)]  # extract complex phase

        ax2[0].plot(baseu[1:int(self.num_pix_x / 2) + 1], vhor, c='b', label="along u-axis", lw=0.7, zorder=1000)
        ax2[1].plot(baseu[1:], phi_hor, c='b', lw=0.7, zorder=1000)

        ax2[0].plot(basev[1:int(self.num_pix_y / 2) + 1], vver, c='g', label="along v-axis", lw=0.7, zorder=1000,
                    ls='--')
        ax2[1].plot(basev[1:], phi_ver, c='g', lw=0.7, zorder=1000, ls='--')

        ax2[0].set_title(f'{vislabel} cuts')
        ax2[0].set_xlabel(r'$B$ [$\mathrm{M \lambda}$]')
        ax2[0].set_ylabel(vislabel)

        if plot_vistype == 'vis' or plot_vistype == 'vis2':
            ax2[0].axhline(y=1, c='k', lw=0.3, ls="--", zorder=0)
        elif plot_vistype == 'fcorr':
            ax2[0].axhline(y=self.ftot, c='k', lw=0.3, ls="--", zorder=0)

        if log_plotv:
            ax2[0].set_yscale("log")
            ax2[0].set_ylim(0.5 * np.min(np.append(vhor, vver)), 2 * np.max(np.append(vhor, vver)))
        else:
            ax2[0].axhline(y=0, c='k', lw=0.3, ls="--", zorder=0)
            ax2[0].set_ylim(np.min(np.append(vhor, vver)), 1.1 * np.max(np.append(vhor, vver)))

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
            plt.savefig(f"{fig_dir}/fft1d_cuts_{self.wave}mum.png", dpi=300, bbox_inches='tight')


# if __name__ == "__main__":
#     mod_dir = '/home/toond/Documents/phd/MCFOST/recr_corporaal_et_al2023/models_akke_mcfost/best_model1_largeFOV/'
#     img_dir = 'PIONIER/data_1.65/'
#     fig_dir = '/home/toond/Downloads/'
#
#     image = ImageFFT(f'{mod_dir}{img_dir}RT.fits.gz', read_method='mcfost', disk_only=True)  # load in image
#     image.redden(ebminv=0)
#     image.diagnostic_plot(fig_dir, plot_vistype='fcorr', log_plotv=True, log_ploti=False)
#     image.freq_info()
