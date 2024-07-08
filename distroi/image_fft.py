"""
Defines a class and the corresponding methods to load in and handle model images and their fast fourier
transform (FFT).
"""

from distroi import constants
from distroi import sed

import os
import glob

import numpy as np
from astropy.io import fits

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class ImageFFT:
    """
    Class containing information on a model image and its FFT. Contains all properties in order to fully describe
    both the image and its FFT. Note that all these properties are expected if all class methods are to work. While
    default options are tuned to MCFOST model disk images, it can easily be generalized to different RT codes by
    defining a corresponding image reader function analogous to 'read_image_fft_mcfost'. Can handle any dimensions of
    image, as long as the amount of pixels in each dimension is even.

    :param dict dictionary: Dictionary containing keys and values representing several instance variables described
        below. Should include 'wavelength', 'pixelscale_x'/'y', 'num_pix_x'/'y', 'img', and 'ftot'. The other required
        instance variables (related to the FFT) are set automatically through perform_fft().
    :ivar float wavelength: ImageFFT wavelength in micron.
    :ivar float pixelscale_x: Pixelscale in radian in x (East-West) direction.
    :ivar float pixelscale_y: Pixelscale in radian in y (North-South) direction.
    :ivar int num_pix_x: Amount of pixels in the x direction.
    :ivar int num_pix_y: Amount of pixels in the y direction.
    :ivar np.ndarray img: 2D numpy array containing the image flux in Jy. 1st index = image y-axis,
        2nd index = image x-axis.
    :ivar float ftot: Total image flux in Jy
    :ivar np.ndarray fft: Complex 2D numpy FFT of img in Jy, i.e. in correlated flux formulation.
    :ivar np.ndarray w_x: 1D array with numpy FFT x-axis frequencies in units of 1/pixelscale_x.
    :ivar np.ndarray w_y: 1D array with numpy FFT y-axis frequencies in units of 1/pixelscale_y.
    :ivar np.ndarray uf: 1D array with FFT spatial x-axis frequencies in 1/radian, i.e. uf = w_x/pixelscale_x.
    :ivar np.ndarray vf: 1D array with FFT spatial y-axis frequencies in 1/radian, i.e. vf = w_y/pixelscale_y.
    """

    def __init__(self, dictionary: dict[str, np.ndarray | float | int]):
        """
        Constructor method. See class docstring for information on instance properties.
        """
        self.wavelength: float | None = None  # image wavelength in micron

        self.pixelscale_x: float | None = None  # pixelscale in radian in x direction
        self.pixelscale_y: float | None = None  # pixelscale in radian in y direction
        self.num_pix_x: int | None = None  # number of pixels in x direction
        self.num_pix_y: int | None = None  # number of pixels in y direction

        self.img: np.ndarray | None = None  # 2d numpy array containing the flux
        # 1st index = image y-axis, 2nd index = image x-axis
        self.ftot: float | None = None  # total flux in Jy

        self.fft: np.ndarray | None = (
            None  # complex numpy FFT of self.img in absolute flux units (Jansky)
        )
        self.w_x: np.ndarray | None = (
            None  # numpy FFT frequencies returned by np.fft.fftshift(np.fft.fftfreq()),
        )
        # i.e. units 1/pixel
        self.w_y: np.ndarray | None = None  # for x and y-axis respectively
        self.uf: np.ndarray | None = None  # FFT spatial frequencies in 1/radian,
        # i.e. uf = w_x/pixelscale_x; vf = w_y/pixelscale_y
        self.vf: np.ndarray | None = (
            None  # for x-axis (u in inteferometric convention) and y-axis
        )
        # (v in inteferometric convention)

        if dictionary is not None:
            # read in from dictionary
            self.wavelength = dictionary["wavelength"]
            self.pixelscale_x, self.pixelscale_y = (
                dictionary["pixelscale_x"],
                dictionary["pixelscale_y"],
            )
            self.num_pix_x, self.num_pix_y = (
                dictionary["num_pix_x"],
                dictionary["num_pix_y"],
            )
            self.img = dictionary["img"]
            self.ftot = dictionary["ftot"]
            # perform the fft to set the other instance variables
            self.perform_fft()

        if self.num_pix_x % 2 != 0 or self.num_pix_y % 2 != 0:
            print(
                "DISTROI currently only supports images with an even amount of pixels in each dimension. "
                "Program will be terminated!"
            )
            exit(1)
        return

    def perform_fft(self):
        """
        Perform the numpy FFT and set the required properties related to the image's FFT.

        :rtype: None
        """
        self.fft = np.fft.fftshift(
            np.fft.fft2(np.fft.fftshift(self.img))
        )  # complex fft in Jansky

        # extract info on the frequencies, note this is in units of 1/pixel
        # !!! NOTE: the first axis in a numpy array is the y-axis of the img, the second axis is the x-axis
        # !!! NOTE: we add a minus because the positive x- and y-axis convention in numpy
        # is the reverse of the interferometric one !!!

        self.w_x = -np.fft.fftshift(
            np.fft.fftfreq(self.fft.shape[1])
        )  # also use fftshift so the 0 frequency
        self.w_y = -np.fft.fftshift(
            np.fft.fftfreq(self.fft.shape[0])
        )  # lies in the middle of the returned array

        self.uf = (
            self.w_x / self.pixelscale_x
        )  # spatial frequencies in units of 1/radian
        self.vf = self.w_y / self.pixelscale_y
        return

    def redden(
        self,
        ebminv: float,
        reddening_law: str = constants.PROJECT_ROOT + "/utils/ISM_reddening"
        "/ISMreddening_law_Cardelli1989.dat",
    ) -> None:
        """
        Further reddens the model image according to the approriate E(B-V) and a corresponding reddening law.

        :param float ebminv: E(B-V) reddening factor to be applied.
        :param str reddening_law: Path to the reddening law to be used. Defaults to the ISM reddening law by
            Cardelli (1989) in DISTROI's 'utils/ISM_reddening folder'. See this file for the expected formatting
            of your own reddening laws.
        :rtype: None
        """
        self.img = constants.redden_flux(
            self.wavelength,
            self.img,  # apply additional reddening to the image
            ebminv,
            reddening_law=reddening_law,
        )
        if self.fft is not None:
            self.fft = constants.redden_flux(
                self.wavelength, self.fft, ebminv, reddening_law
            )  # apply additional reddening to the fft
        self.ftot = constants.redden_flux(
            self.wavelength, self.ftot, ebminv, reddening_law
        )  # apply additional reddening to the toal flux
        return

    def freq_info(self) -> str:
        """
        Returns a string containing information on both the spatial frequency domain/sampling and the corresponding
        projected baselines.

        :return info_str: String containing frequency info which can be printed.
        :rtype: str
        """
        image_size_x = self.pixelscale_x * self.num_pix_x
        image_size_y = self.pixelscale_y * self.num_pix_y

        info_str = (
            f"===================================== \n"
            f"FREQUENCY INFORMATION IN PIXEL UNITS: \n"
            f"===================================== \n"
            f"Maximum frequency considered E-W [1/pixel]: {np.max(self.w_x):.4E} \n"
            f"Maximum frequency considered S-N [1/pixel]: {np.max(self.w_y):.4E} \n"
            f"This should equal the Nyquist frequency = 0.5 x 1/sampling_rate "
            f"(sampling_rate = 1 pixel in pixel units, = 1 pixelscale in physical units) \n"
            f"Spacing frequency space E-W [1/pixel]: {abs(self.w_x[1] - self.w_x[0]):.4E} \n"
            f"Spacing frequency space S-N [1/pixel]: {abs(self.w_y[1] - self.w_y[0]):.4E} \n"
            f"This should equal 1/window_size "
            f"(i.e. = 1/(#pixels) in pixel units = 1/image_size in physical units) \n\n"
            f"======================================= \n"
            f"FREQUENCY INFORMATION IN ANGULAR UNITS: \n"
            f"======================================= \n"
            f"Pixel scale E-W [rad]: {self.pixelscale_x:.4E} \n"
            f"Pixel scale S-N [rad]: {self.pixelscale_y:.4E} \n"
            f"Image axis size E-W [rad]: {image_size_x:.4E} \n"
            f"Image axis size S-N [rad]: {image_size_y:.4E} \n"
            f"Maximum frequency considered E-W [1/rad]: {(np.max(self.w_x) * 1 / self.pixelscale_x):.4E} \n"
            f"Maximum frequency considered S-N [1/rad]: {(np.max(self.w_y) * 1 / self.pixelscale_y):.4E} \n"
            f"Spacing frequency space E-W [1/rad]: {abs((self.w_x[1] - self.w_x[0]) / self.pixelscale_x):.4E}\n"
            f"Spacing frequency space S-N [1/rad]: {abs((self.w_y[1] - self.w_y[0]) / self.pixelscale_y):.4E}\n"
            f"--------------------------------------- \n"
            f"Pixel scale E-W (mas): {(self.pixelscale_x * constants.RAD2MAS):.4E} \n"
            f"Pixel scale S-N (mas): {(self.pixelscale_y * constants.RAD2MAS):.4E} \n"
            f"Image axis size E-W (mas): {(image_size_x * constants.RAD2MAS):.4E} \n"
            f"Image axis size S-N (mas): {(image_size_y * constants.RAD2MAS):.4E} \n"
            f"Maximum frequency considered E-W [1/mas]: "
            f"{(np.max(self.w_x) * 1 / (self.pixelscale_x * constants.RAD2MAS)):.4E} \n"
            f"Maximum frequency considered S-N [1/mas]: "
            f"{(np.max(self.w_y) * 1 / (self.pixelscale_y * constants.RAD2MAS)):.4E} \n"
            f"Spacing in frequency space E-W [1/mas]: "
            f"{abs((self.w_x[1] - self.w_x[0]) * 1 / (self.pixelscale_x * constants.RAD2MAS)):.4E} \n"
            f"Spacing in frequency space S-N [1/mas]: "
            f"{abs((self.w_y[1] - self.w_y[0]) * 1 / (self.pixelscale_y * constants.RAD2MAS)):.4E} \n\n"
            f"================================================================ \n"
            f"FREQUENCY INFORMATION IN TERMS OF CORRESPONDING BASELINE LENGTH: \n"
            f"================================================================ \n"
            f"Maximum projected baseline resolvable under current pixel sampling E-W [Mlambda]: "
            f"{((np.max(self.w_x) * 1 / self.pixelscale_x) / 1e6):.4E} \n"
            f"Maximum projected baseline resolvable under current pixel sampling S-N [Mlambda]: "
            f"{((np.max(self.w_y) * 1 / self.pixelscale_y) / 1e6):.4E} \n"
            f"Spacing in projected baseline length corresponding to frequency sampling E-W [Mlambda]: "
            f"{abs(((self.w_x[1] - self.w_x[0]) * 1 / self.pixelscale_x) / 1e6):.4E} \n"
            f"Spacing in projected baseline length corresponding to frequency sampling S-N [Mlambda]: "
            f"{abs(((self.w_y[1] - self.w_y[0]) * 1 / self.pixelscale_y) / 1e6):.4E} \n"
            f"---------------------------------------------------------------- \n"
            f"Maximum projected baseline resolvable under current pixel sampling E-W [m]: "
            f"{((np.max(self.w_x) * 1 / self.pixelscale_x) * self.wavelength * constants.MICRON2M):.4E} \n"
            f"Maximum projected baseline resolvable under current pixel sampling S-N [m]: "
            f"{((np.max(self.w_y) * 1 / self.pixelscale_y) * self.wavelength * constants.MICRON2M):.4E} \n"
            f"Spacing in projected baseline length corresponding to frequency sampling E-W [m]: "
            f"{abs(((self.w_x[1] - self.w_x[0]) * 1 / self.pixelscale_x) * self.wavelength * constants.MICRON2M):.4E} \n"
            f"Spacing in projected baseline length corresponding to frequency sampling S-N [m]: "
            f"{abs(((self.w_y[1] - self.w_y[0]) * 1 / self.pixelscale_y) * self.wavelength * constants.MICRON2M):.4E} \n"
            f"================================================================ \n"
        )
        return info_str

    def add_point_source(self, flux: float, coords: tuple[float, float]) -> None:
        """
        Function that adds the effect of an additional point source to the ImageFFT instance. This affects the
        following instance variables: 'img', 'ftot' and 'fft'. The flux of the point source is added to the pixel in
        'img' closest to the specified source position as well as to the value of ftot. 'fft' is modified by the
        analytical addition of the point-source's complex correlated flux. Note that, after this function is called,
        there is thus a mismatch between 'fft' and 'img', since the effect of the source is added in an exact
        analytic manner in the former while it is added aproximately in the latter. For example, nothing prevents the
        addition of a point source far outside the image axes ranges. The addition to 'fft' will be analytically exact,
        while the point source flux will be added to the nearest border pixel in 'img'. The addition to 'img' serves
        more for visualization, e.g. the plots created by fft_diagonstic_plot(). As a result, be careful with future
        invocations of perform_fft(), the effect may not be as desired!

        :param float flux: The flux value (F_nu, in Jy) of the point source at the wavelength of the ImageFFT instance.
        :param tuple(float) coords: 2D tuple giving the x and y position of the point source (in milli-arcsecond).
            The positive x direction is towards the East. The positive y direction is towards the North.
        :rtype: None
        """
        x, y = coords[0], coords[1]  # point source position

        self.ftot += flux

        # define numpy arrays with coordinates for pixel centres in milli-arcsecond
        coords_pix_x = (
            np.linspace(
                self.num_pix_x / 2 - 0.5, -self.num_pix_x / 2 + 0.5, self.num_pix_x
            )
            * self.pixelscale_x
            * constants.RAD2MAS
        )
        coords_pix_y = (
            np.linspace(
                self.num_pix_y / 2 - 0.5, -self.num_pix_y / 2 + 0.5, self.num_pix_y
            )
            * self.pixelscale_y
            * constants.RAD2MAS
        )
        coords_pix_mesh_x, coords_pix_mesh_y = np.meshgrid(
            coords_pix_x, coords_pix_y
        )  # create a meshgrid
        distances = np.sqrt(
            (coords_pix_mesh_x - x) ** 2 + (coords_pix_mesh_y - y) ** 2
        )  # calc dist pixels to point

        min_dist_indices = np.where(
            distances == np.min(distances)
        )  # retrieve indices of where distance is minimum
        min_ind_x, min_ind_y = min_dist_indices[1][0], min_dist_indices[0][0]

        self.img[min_ind_y][min_ind_x] += (
            flux  # add point source flux to the nearest pixel
        )

        # define meshgrid for spatial frequencies in units of 1/milli-arcsecond
        freq_mesh_u, freq_mesh_v = np.meshgrid(
            self.uf * constants.MAS2RAD, self.vf * constants.MAS2RAD
        )
        # add the complex contribution of the secondary to the stored FFT
        self.fft += flux * np.exp(-2j * np.pi * (freq_mesh_u * x + freq_mesh_v * y))

        return

    def add_overresolved_flux(self, flux: float) -> None:
        """
        Function that adds the effect of an overresolved flux component to the ImageFFT instance. This affects the
        following instance variables: 'img', and 'ftot'. The flux of the overresolved component is added to 'img',
        spread uniformly accross all its pixels. The flux is also added to 'ftot'. Note that the complex visibilities
        in 'fft' remain unaffected. This is the exact, analytical formulation of infinitely extended flux,
        where the correlated flux of such a component is a direct delta at frequency 0. In this regime ad using the
        FFT, only 'ftot' is increased, while 'fft' remains unaffected. This only has an effect on (squared)
        visibilities if they are normalized. If they are expressed in correlated flux, this addition will have no
        effect. Note that this function causes a discrepancy between 'fft' and 'img', as the addition to the former
        assumes an infinitely extended uniform flux, while addition to the latter is spread over the finite image
        size. The addition to 'img' serves more for visualization, e.g. the plots created by fft_diagonstic_plot().
        As a result, be careful with future invocations of perform_fft(), the effect may not be as desired!

        :param float flux: The flux value (in Jy) of the overresolved component at the wavelength of the ImageFFT
            instance.
        :rtype: None
        """

        self.ftot += flux
        self.img += flux / (self.num_pix_x * self.num_pix_y)

        return

    def half_light_radius(self) -> float:
        """
        Calculate the half light radius of the image by adding up the fluxes of pixels within increasing circular
        aperatures. If you want only the half light radius excluding the central source, e.g. the model disk in an
        MCFOST model, one should exclude the central source when reading in the image file (e.g. using disk_only=True
        with read_image_fft_mcfost()). Due to the implImageFFTimentation, the returned value's accuracy is inherently
        limited by the model image pixelscale. Note also that the radius is calculated in the image plane, and thus
        depends on e.g. inclination of the RT model.

        :return hlr: The half light radius in milli-arcseconds.
        :rtype: float
        """
        # define numpy arrays with coordinates for pixel centres in milli-arcsecond
        coords_pix_x = (
            np.linspace(
                self.num_pix_x / 2 - 0.5, -self.num_pix_x / 2 + 0.5, self.num_pix_x
            )
            * self.pixelscale_x
            * constants.RAD2MAS
        )
        coords_pix_y = (
            np.linspace(
                self.num_pix_y / 2 - 0.5, -self.num_pix_y / 2 + 0.5, self.num_pix_y
            )
            * self.pixelscale_y
            * constants.RAD2MAS
        )
        coords_pix_mesh_x, coords_pix_mesh_y = np.meshgrid(
            coords_pix_x, coords_pix_y
        )  # create a meshgrid
        distances = np.sqrt(
            coords_pix_mesh_x**2 + coords_pix_mesh_y**2
        )  # calc dist pixels to origin

        rcont = 0  # variable denoting the radius of the circular aperature in milli-arcsecond
        fcont = 0  # variable containing the flux within ciruclar aperature contour
        rcont_interval = min(
            self.pixelscale_x * constants.RAD2MAS, self.pixelscale_y * constants.RAD2MAS
        )  # interval to increase aperature radius
        while fcont < 0.5 * self.ftot:
            rcont += rcont_interval
            fcont = np.sum(
                self.img[distances <= rcont]
            )  # sum up all pixel fluxes in the aperature
        hlr = rcont  # set the contour radius to be the half light radius

        return hlr

    def diagnostic_plot(
        self,
        fig_dir: str = None,
        plot_vistype: str = "vis2",
        log_plotv: bool = False,
        log_ploti: bool = False,
        show_plots: bool = True,
    ) -> None:
        """
        Makes diagnostic plots showing both the model image and the FFT (squared) visibilities and complex phases.

        :param str fig_dir: Directory to store plots in.
        :param str plot_vistype: Sets the type of visibility to be plotted. 'vis2' for squared visibilities, 'vis'
            for visibilities or 'fcorr' for correlated flux in Jy.
        :param bool log_plotv: Set to True for a logarithmic y-scale in the (squared) visibility plot.
        :param bool log_ploti: Set to True for a logarithmic intensity scale in the model image plot.
        :param bool show_plots: Set to False if you do not want the plots to be shown during your script run.
            Note that if True, this freazes further code execution until the plot windows are closed.
        :rtype: None
        """
        baseu = self.uf / 1e6  # Baseline length u in MegaLambda
        basev = self.vf / 1e6  # Baseline length v in MegaLambda

        step_baseu = abs(
            baseu[1] - baseu[0]
        )  # retrieve the sampling steps in u baseline length
        step_basev = abs(
            basev[1] - basev[0]
        )  # retrieve the sampling steps in v baseline length

        # create plotting directory if it doesn't exist yet
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        if log_plotv:
            normv = "log"
        else:
            normv = "linear"
        if log_ploti:
            normi = "log"
        else:
            normi = "linear"

        # do some plotting
        fig, ax = plt.subplots(2, 3, figsize=(15, 10))
        color_map = "inferno"

        # intensity plotted in pixel scale
        # also set the extent of the image when you plot it, take care that the number of pixels is even
        img_plot = ax[0][0].imshow(
            self.img,
            cmap=color_map,
            norm=normi,
            extent=(
                self.num_pix_x / 2 + 0.5,
                -self.num_pix_x / 2 + 0.5,
                -self.num_pix_y / 2 + 0.5,
                self.num_pix_y / 2 + 0.5,
            ),
        )
        fig.colorbar(
            img_plot, ax=ax[0][0], label="$I$ (Jy/pixel)", fraction=0.046, pad=0.04
        )
        ax[0][0].set_title("Intensity")
        ax[0][0].set_xlabel("E-W (pixel)")
        ax[0][0].set_ylabel("S-N (pixel)")
        ax[0][0].arrow(
            0.90,
            0.80,
            -0.1,
            0,
            color="white",
            transform=ax[0][0].transAxes,
            length_includes_head=True,
            head_width=0.015,
        )  # draw arrows to indicate direction
        ax[0][0].text(0.78, 0.83, "E", color="white", transform=ax[0][0].transAxes)
        ax[0][0].arrow(
            0.90,
            0.80,
            0,
            0.1,
            color="white",
            transform=ax[0][0].transAxes,
            length_includes_head=True,
            head_width=0.015,
        )
        ax[0][0].text(0.92, 0.90, "N", color="white", transform=ax[0][0].transAxes)
        ax[0][0].axhline(y=0, lw=0.2, color="white")
        ax[0][0].axvline(x=0, lw=0.2, color="white")

        # set the (squared) visibility of the FFT
        if plot_vistype == "vis2":
            vislabel = "$V^2$"
            vis = abs(self.fft / self.ftot) ** 2
        elif plot_vistype == "vis":
            vislabel = "$V$"
            vis = abs(self.fft / self.ftot)
        elif plot_vistype == "fcorr":
            vislabel = r"$F_{corr}$ (Jy)"
            vis = abs(self.fft)
        else:
            print("vislabel not recognized, using vis2 as default")
            vislabel = "$V^2$"
            vis = abs(self.fft / self.ftot) ** 2

        # set the complex phase
        cphi = np.angle(self.fft, deg=True)

        v2plot = ax[0][1].imshow(
            vis,
            cmap=color_map,
            norm=normv,
            extent=(
                self.num_pix_x / 2 + 0.5,
                -self.num_pix_x / 2 + 0.5,
                -self.num_pix_y / 2 + 0.5,
                self.num_pix_y / 2 + 0.5,
            ),
        )
        fig.colorbar(v2plot, ax=ax[0][1], label=vislabel, fraction=0.046, pad=0.04)

        ax[0][1].axhline(y=0, lw=0.2, color="black")
        ax[0][1].axvline(x=0, lw=0.2, color="black")

        ax[0][1].set_title(vislabel)
        ax[0][1].set_xlabel(r"$\leftarrow u$ (1/pixel)")
        ax[0][1].set_ylabel(r"$v \rightarrow$ (1/pixel)")

        # complex phase of the FFT in pixel scale
        phi_plot = ax[0][2].imshow(
            cphi,
            cmap=color_map,
            extent=(
                self.num_pix_x / 2 + 0.5,
                -self.num_pix_x / 2 + 0.5,
                -self.num_pix_y / 2 + 0.5,
                self.num_pix_y / 2 + 0.5,
            ),
        )
        fig.colorbar(
            phi_plot, ax=ax[0][2], label=r"$\phi$ ($^\circ$)", fraction=0.046, pad=0.04
        )
        ax[0][2].axhline(y=0, lw=0.2, color="black")
        ax[0][2].axvline(x=0, lw=0.2, color="black")

        ax[0][2].set_title(r"Complex Phase $\phi$")
        ax[0][2].set_xlabel(r"$\leftarrow u$ (1/pixel)")
        ax[0][2].set_ylabel(r"$v \rightarrow$ (1/pixel)")

        # intensity plotted in angle scale
        img_plot = ax[1][0].imshow(
            self.img,
            cmap=color_map,
            aspect="auto",
            norm=normi,
            extent=(
                (self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                (-self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                (-self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS,
                (self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS,
            ),
        )
        fig.colorbar(
            img_plot, ax=ax[1][0], label="$I$ (Jy/pixel)", fraction=0.046, pad=0.04
        )
        ax[1][0].set_aspect(self.num_pix_y / self.num_pix_x)
        ax[1][0].set_title("Intensity")
        ax[1][0].set_xlabel("E-W (mas)")
        ax[1][0].set_ylabel("S-N (mas)")
        ax[1][0].arrow(
            0.90,
            0.80,
            -0.1,
            0,
            color="white",
            transform=ax[1][0].transAxes,
            length_includes_head=True,
            head_width=0.015,
        )  # draw arrows to indicate direction
        ax[1][0].text(0.78, 0.83, "E", color="white", transform=ax[1][0].transAxes)
        ax[1][0].arrow(
            0.90,
            0.80,
            0,
            0.1,
            color="white",
            transform=ax[1][0].transAxes,
            length_includes_head=True,
            head_width=0.015,
        )
        ax[1][0].text(0.92, 0.90, "N", color="white", transform=ax[1][0].transAxes)
        ax[1][0].axhline(y=0, lw=0.2, color="white")
        ax[1][0].axvline(x=0, lw=0.2, color="white")

        # if res_beam is not None:  # plot FWHM Gaussian beam if passed along
        #     res_ellipse = Ellipse(xy=((self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS -
        #                               2 * res_beam.fwhm_min, (self.num_pix_x / 2) * self.pixelscale_x *
        #                               constants.RAD2MAS - 2 * res_beam.fwhm_maj), width=res_beam.fwhm_min,
        #                           height=res_beam.fwhm_maj, angle=-res_beam.pa, edgecolor='r', lw=1.0,
        #                           fc='none', alpha=1)
        #     ax[1][0].add_patch(res_ellipse)

        # (squared) visibility of the FFT in MegaLambda (baseline length) scale
        v2plot = ax[1][1].imshow(
            vis,
            cmap=color_map,
            norm=normv,
            extent=(
                (self.num_pix_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_y / 2 + 0.5) * step_basev,
                (self.num_pix_y / 2 + 0.5) * step_basev,
            ),
        )
        fig.colorbar(v2plot, ax=ax[1][1], label=vislabel, fraction=0.046, pad=0.04)
        ax[1][1].axhline(y=0, lw=0.2, color="black")
        ax[1][1].axvline(x=0, lw=0.2, color="black")

        ax[1][1].set_title(vislabel)
        ax[1][1].set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        ax[1][1].set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

        # complex phase of the FFT in MegaLambda (baseline length) scale
        phi_plot = ax[1][2].imshow(
            cphi,
            cmap=color_map,
            extent=(
                (self.num_pix_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_y / 2 + 0.5) * step_basev,
                (self.num_pix_y / 2 + 0.5) * step_basev,
            ),
        )
        fig.colorbar(
            phi_plot, ax=ax[1][2], label=r"$\phi$ ($^\circ$)", fraction=0.046, pad=0.04
        )
        ax[1][2].axhline(y=0, lw=0.2, color="black")
        ax[1][2].axvline(x=0, lw=0.2, color="black")
        ax[1][2].set_title(r"Complex Phase $\phi$")
        ax[1][2].set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        ax[1][2].set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

        # draw lines/cuts along which we will plot some curves
        ax[1][1].plot(
            np.zeros_like(basev[1 : int(self.num_pix_y / 2) + 1]),
            basev[1 : int(self.num_pix_y / 2) + 1],
            c="g",
            lw=2,
            ls="--",
        )
        ax[1][1].plot(
            baseu[1 : int(self.num_pix_x / 2) + 1],
            np.zeros_like(baseu[1 : int(self.num_pix_x / 2) + 1]),
            c="b",
            lw=2,
        )

        ax[1][2].plot(
            np.zeros_like(basev[1 : int(self.num_pix_y / 2) + 1]),
            basev[1 : int(self.num_pix_y / 2) + 1],
            c="g",
            lw=2,
            ls="--",
        )
        ax[1][2].plot(
            baseu[1 : int(self.num_pix_x / 2) + 1],
            np.zeros_like(baseu[1 : int(self.num_pix_x / 2) + 1]),
            c="b",
            lw=2,
        )
        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(
                f"{fig_dir}/fft2d_maps_{self.wavelength}mum.png",
                dpi=300,
                bbox_inches="tight",
            )

        # Some plots of specific cuts in frequency space
        fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

        # Cuts of visibility and complex phase plot in function of baseline length
        # note we cut away the point furthest along positive u-axis since it contains a strong artefact due to
        # the FFT algorithm, otherwise we move down to spatial frequency 0
        vhor = vis[
            int(self.num_pix_y / 2), 1 : int(self.num_pix_x / 2) + 1
        ]  # extract (squared) visibility along u-axis
        phi_hor = cphi[int(self.num_pix_y / 2), 1:]  # extract complex phase

        vver = vis[
            1 : int(self.num_pix_y / 2) + 1, int(self.num_pix_x / 2)
        ]  # extract (squared) visibility along u-axis
        phi_ver = cphi[1:, int(self.num_pix_x / 2)]  # extract complex phase

        ax2[0].plot(
            baseu[1 : int(self.num_pix_x / 2) + 1],
            vhor,
            c="b",
            label="along u-axis",
            lw=0.7,
            zorder=1000,
        )
        ax2[1].plot(baseu[1:], phi_hor, c="b", lw=0.7, zorder=1000)

        ax2[0].plot(
            basev[1 : int(self.num_pix_y / 2) + 1],
            vver,
            c="g",
            label="along v-axis",
            lw=0.7,
            zorder=1000,
            ls="--",
        )
        ax2[1].plot(basev[1:], phi_ver, c="g", lw=0.7, zorder=1000, ls="--")

        ax2[0].set_title(f"{vislabel} cuts")
        ax2[0].set_xlabel(r"$B$ ($\mathrm{M \lambda}$)")
        ax2[0].set_ylabel(vislabel)

        if plot_vistype == "vis" or plot_vistype == "vis2":
            ax2[0].axhline(y=1, c="k", lw=0.3, ls="--", zorder=0)
        elif plot_vistype == "fcorr":
            ax2[0].axhline(y=self.ftot, c="k", lw=0.3, ls="--", zorder=0)

        if log_plotv:
            ax2[0].set_yscale("log")
            ax2[0].set_ylim(
                0.5 * np.min(np.append(vhor, vver)), 2 * np.max(np.append(vhor, vver))
            )
        else:
            ax2[0].axhline(y=0, c="k", lw=0.3, ls="--", zorder=0)
            ax2[0].set_ylim(
                np.min(np.append(vhor, vver)), 1.1 * np.max(np.append(vhor, vver))
            )

        ax2[1].set_title(r"$\phi$ cuts")
        ax2[1].set_xlabel(r"$B$ ($\mathrm{M \lambda}$)")
        ax2[1].set_ylabel(r"$\phi$ ($^\circ$)")
        ax2[1].axvline(x=0, c="k", lw=0.3, ls="-", zorder=0)
        ax2[1].axhline(y=0, c="k", lw=0.3, ls="-", zorder=0)
        ax2[1].axhline(y=180, c="k", lw=0.3, ls="--", zorder=0)
        ax2[1].axhline(y=-180, c="k", lw=0.3, ls="--", zorder=0)
        ax2[0].legend()
        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(
                f"{fig_dir}/fft1d_cuts_{self.wavelength}mum.png",
                dpi=300,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        return


def read_image_fft_mcfost(img_path: str, disk_only: bool = False):
    """Image instance
    Retrieve image data from an MCFOST model image and return it as an ImageFFT class instance.

    :param str img_path: Path to an MCFOST output RT.fits.gz model image file.
    :param bool disk_only: Set to True if you only want to read in the flux from the disk.
    :return image: ImageFFT instance containing the information on the MCFOST RT image.
    :rtype: ImageFFT
    """
    dictionary = {}  # dictionary to construct ImageFFT instance

    az, inc = 0, 0  # only load the first azimuthal/inc value image in the .fits file

    # open the required fits file + get some header info
    hdul = fits.open(img_path)

    # read in the wavelength, pixelscale and 'window size' (e.g. size of axis images in radian)
    dictionary["wavelength"] = hdul[0].header["WAVE"]  # wavelength in micron
    dictionary["pixelscale_x"] = (
        abs(hdul[0].header["CDELT1"]) * constants.DEG2RAD
    )  # loads degrees, converted to radian
    dictionary["pixelscale_y"] = abs(hdul[0].header["CDELT2"]) * constants.DEG2RAD
    dictionary["num_pix_x"] = hdul[0].header["NAXIS1"]  # number of pixels
    dictionary["num_pix_y"] = hdul[0].header["NAXIS2"]

    img_array = hdul[0].data  # read in image data in lam x F_lam units W m^-2
    img_array = np.flip(
        img_array, axis=3
    )  # flip y-array (to match numpy axis convention)

    img_tot = img_array[0, az, inc, :, :]  # full total flux
    img_star = img_array[4, az, inc, :, :]
    img_disk = img_tot - img_star

    # Set all image-related class instance properties below.

    if disk_only:
        dictionary["img"] = img_disk
    else:
        dictionary["img"] = img_tot

    # calculate fft in Jy units
    dictionary["img"] *= (
        dictionary["wavelength"] * constants.MICRON2M
    ) / constants.SPEED_OF_LIGHT  # convert to F_nu in SI units (W m^-2 Hz^-1)
    dictionary["img"] *= constants.WATT_PER_M2_HZ_2JY  # convert image to Jansky
    dictionary["ftot"] = np.sum(dictionary["img"])  # total flux in Jansky

    # return an ImageFFT object
    image = ImageFFT(dictionary=dictionary)
    return image


def get_image_fft_list(
    mod_dir: str,
    img_dir: str,
    read_method: str = "mcfost",
    ebminv: float = 0.0,
    reddening_law: str = f"{constants.PROJECT_ROOT}"
    f"/utils/ISM_reddening/ISMreddening_law_Cardelli1989.dat",
) -> list[ImageFFT] | None:
    """
    Function that takes the path to an RT model's directory and a subdirectory containing image files, and returns a
    list of ImageFFT objects representing those model images. They should thus represent the same underlying physical
    model, but imaged at different wavelengths.

    :param str mod_dir: Parent directory of the RT model of interest.
    :param str img_dir: Subdirectory containing RT model images. All image files recursively found in the subdirectories
         of 'mod_dir+img_dir' are read.
    :param str read_method: Type of method used to read in RT model images when creating ImageFFT class instances.
        Currently only supports 'mcfost', in which case all files ending on the sufix 'RT.fits.gz' are read in.
    :param float ebminv: E(B-V) of additional reddening to be applied to the model images. Only useful if
        the visibilities need to be expressed in correlated flux at some point.
    :param str reddening_law: Path to the reddening law to be used. Defaults to the ISM reddening law by
        Cardelli (1989) in DISTROI's 'utils/ISM_reddening folder'. See this file for the expected formatting
        of your own reddening laws.
    :return: img_fft_list: List of ImageFFT objects representing all model image files found under 'mod_dir+img_dir'.
        Sorted by wavelength
    :rtype: list(ImageFFT)
    """

    img_fft_list = []  # list of ImageFFT objects to be held (1 element long in the case of monochr=True)
    wavelengths = []  # list of their wavelengths

    if read_method == "mcfost":  # different ways to read in model image file paths
        img_file_paths = sorted(
            glob.glob(f"{mod_dir}{img_dir}/**/*RT.fits.gz", recursive=True)
        )
    else:
        print(f"read_method '{read_method}' not recognized. Will return None!")
        return

    for img_path in img_file_paths:
        if read_method == "mcfost":  # choose reader function
            img_fft = read_image_fft_mcfost(img_path)
        else:
            print(f"read_method '{read_method}' not recognized. Will return None!")
            return
        img_fft.redden(
            ebminv=ebminv, reddening_law=reddening_law
        )  # redden the ImageFFT object
        img_fft_list.append(img_fft)  # append to the list of ImageFFT objects
        wavelengths.append(img_fft.wavelength)  # append wavelength

    wavelengths, img_fft_list = list(
        zip(*sorted(zip(wavelengths, img_fft_list)))
    )  # sort the objects in wavelength

    return img_fft_list
