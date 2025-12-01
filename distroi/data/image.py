"""A module for handling images and their fast Fourier transform (FFT).

Warnings
--------
Currently only supports images with an even amount of pixels.

"""
# TODO: add support for polarimetric observations, e.g. different flux types.
# TODO: add support for convolution of two images (and their FFT) with the same dimensions.
# this can then be used to convolve with e.g. the PSF of the UV coverage to reveal the resolvable features of a model.

from distroi.auxiliary import constants
from distroi.model.dep import spec_dep

import os
import glob

import numpy as np
from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d

from typing import Literal

import matplotlib.pyplot as plt

from distroi.model.geom_comp.geom_comp import GeomComp

constants.set_matplotlib_params()  # set project matplotlib parameters


class Image:
    """Contains information on an image and its FFT.

    Contains all attributes in order to fully describe both a regular grid image and its FFT. Note that all these
    attributes are expected if all class methods are to work. It can easily be generalized to different RT codes by
    defining a corresponding image reader function analogous to `read_image_fft_mcfost`. Can handle any amount
    of pixels in an image, as long as the amount of pixels in each dimension is even.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys and values representing several instance variables described below. Should include
        `wavelength`, `pixelscale_x/y`, `num_pix_x/y`, `img`, and `ftot`. The other required instance variables
        (related to the FFT) are set automatically through `do_fft`.
    sp_dep : SpecDep, optional
        Optional spectral dependence of the image. This will only be used if this `Image` is used on its own in methods
        calculating interferometric observables. If instead multiple `Image` objects or an `SED` are passed along as
        well, this property of the image will be ignored. By default, the spectral dependency will be assumed to be
        flat in F_lam across wavelengths.
    padding : tuple of int, optional
        Number of (x, y)-pixels to which an image should be 0-padded before performing the FFT. I.e.
        ``padding=(680, 540)`` will 0-pad an image to 680 and 540 pixels in the x and y dimensions, respectively.
        If smaller than the number of pixels already in the `img` array, no padding will be added in the respective
        dimension. These should both be even numbers!

    Attributes
    ----------
    wavelength : float
        Image wavelength in micron.
    pixelscale_x : float
        Pixelscale in radian in x (East-West) direction.
    pixelscale_y : float
        Pixelscale in radian in y (North-South) direction.
    num_pix_x : int
        Amount of image pixels in the x direction.
    num_pix_y : int
        Amount of image pixels in the y direction.
    img : np.ndarray
        2D numpy array containing the image flux in Jy. 1st index = image y-axis, 2nd index = image x-axis.
    ftot : float
        Total image flux in Jy.
    sp_dep : SpecDep
        Optional spectral dependence of the image. Assumed flat in F_lam by default.
    fft : np.ndarray
        Complex 2D numpy FFT of `img` in Jy, i.e. in correlated flux formulation.
    num_pix_fft_x : int
        Amount of image FFT pixels in the x direction. This can be different from `num_pix_x` due to padding.
    num_pix_fft_y : int
        Amount of image FFT pixels in the y direction. This can be different from `num_pix_y` due to padding.
    w_x : np.ndarray
        1D array with numpy FFT x-axis frequencies in units of ``1/pixelscale_x``.
    w_y : np.ndarray
        1D array with numpy FFT y-axis frequencies in units of ``1/pixelscale_y``.
    uf : np.ndarray
        1D array with FFT spatial x-axis frequencies in cycles/radian, i.e. ``uf = w_x/pixelscale_x``.
    vf : np.ndarray
        1D array with FFT spatial y-axis frequencies in cycles/radian, i.e. ``vf = w_y/pixelscale_y``.

    Warnings
    --------
    The default value of `sp_dep` assumes a flat spectrum in F_lam. This implies a spectral dependency ~ frequency^-2
    ~ wavelength^2 for F_nu. Thus the total correlated flux of the image FFT will not be flat accross wavelength.
    """

    def __init__(
        self,
        dictionary: dict[str, np.ndarray | float | int],
        sp_dep: spec_dep.SpecDep | None = None,
        padding: tuple[int, int] | None = None,
    ):
        self.wavelength: float | None = None  # image wavelength in micron
        # TODO: add option for adding polarimetric state information of light, maybe using self.stokes_type

        ## image information
        self.pixelscale_x: float | None = None  # pixelscale in radian in x direction
        self.pixelscale_y: float | None = None  # pixelscale in radian in y direction
        self.num_pix_x: int | None = None  # number of image pixels in x direction
        self.num_pix_y: int | None = None  # number of image pixels in y direction

        self.img: np.ndarray | None = None  # 2d numpy array containing the flux
        # 1st index = image y-axis, 2nd index = image x-axis
        self.ftot: float | None = None  # total image flux in Jy
        self.sp_dep: spec_dep.SpecDep | None = None  # spectral dependency

        ## fft information
        self.fft: np.ndarray | None = None  # complex numpy FFT of self.img in absolute flux units (Jansky)
        self.num_pix_fft_x: int | None = (
            None  # number of image FFT pixels in x direction. This can be different from num_pix_x/y due to padding
        )
        self.num_pix_fft_y: int | None = None  # number of image FFT pixels in y direction
        self.w_x: np.ndarray | None = None  # numpy FFT frequencies returned by np.fft.fftshift(np.fft.fftfreq()),
        # i.e. units 1/pixel
        self.w_y: np.ndarray | None = None  # for x and y-axis respectively
        self.uf: np.ndarray | None = None  # FFT spatial frequencies in 1/radian,
        # i.e. uf = w_x/pixelscale_x; vf = w_y/pixelscale_y
        self.vf: np.ndarray | None = None  # for x-axis (u in inteferometric convention) and y-axis
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

            # check for
            if self.num_pix_x % 2 != 0 or self.num_pix_y % 2 != 0:
                raise ValueError(
                    f"""Image dimensions are ({self.num_pix_x, self.num_pix_y}).
                    DISTROI currently only supports images with an even amount of pixels in each dimension."""
                )

            self.img = dictionary["img"]
            self.ftot = dictionary["ftot"]
            # perform the fft to set the other instance variables
            self.do_fft(padding=padding)

        # set spectral dependency
        if sp_dep is not None:
            self.sp_dep = sp_dep  # set spectral dependence if given
        else:
            self.sp_dep = spec_dep.FlatSpecDep(flux_form="flam")  # otherwise, assume flat spectrum in F_lam

        return

    def do_fft(self, padding: tuple[int, int] | None = None) -> None:
        """Perform an FFT on the image.

        Perform the numpy FFT on the `img` property and set the other required attributes related to the
        image's FFT.

        Parameters
        ----------
        padding : tuple of int, optional
            Number of (x, y)-pixels to which an image should be 0-padded before performing the FFT. I.e.
            ``padding=(680, 540)`` will 0-pad an image to 680 and 540 pixels in the x and y dimensions, respectively.
            If smaller than the number of pixels already in the `img` array, no padding will be added in the
            respective dimension. These should both be even numbers!

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the padding dimensions are not even numbers.
        """

        if padding is None:  # case if no padding is required
            image = self.img  # set image to perform FFT on to the one passed along in the constructor dictionary
            self.num_pix_fft_x = self.num_pix_x  # no padding is required -> the number of pixels of the FFT is the same
            self.num_pix_fft_y = self.num_pix_y  # as that of the image
        else:
            if padding[0] % 2 != 0 or padding[1] % 2 != 0:  # check for even amount of pixels in padding.
                raise ValueError(
                    f"""DISTROI currently only supports padding image FFTs to an even amount of pixels in 
                    each dimension. Requested padding is ({padding[0]}, {padding[1]})."""
                )

            # check if padding number of pixels is larger than passed along image
            # otherwise just use the number of pixels already in the image
            if padding[0] > self.num_pix_x:
                self.num_pix_fft_x = padding[0]
            else:
                self.num_pix_fft_x = self.num_pix_x
            if padding[1] > self.num_pix_y:
                self.num_pix_fft_y = padding[1]
            else:
                self.num_pix_fft_y = self.num_pix_y

            pad_num_x = int((self.num_pix_fft_x - self.num_pix_x) / 2)  # amount by which to pad on each side
            pad_num_y = int((self.num_pix_fft_y - self.num_pix_y) / 2)

            # perform padding
            image = np.pad(
                self.img,
                pad_width=((pad_num_y, pad_num_y), (pad_num_x, pad_num_x)),
                mode="constant",
                constant_values=(0, 0),
            )

        self.fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(image)))  # perform complex fft in Jansky

        # extract info on the frequencies, note this is in units of 1/pixel
        # !!! NOTE: the first axis in a numpy array is the y-axis of the img, the second axis is the x-axis
        # !!! NOTE: we add a minus because the positive x- and y-axis convention in numpy
        # is the reverse of the interferometric one !!!

        self.w_x = -np.fft.fftshift(np.fft.fftfreq(self.fft.shape[1]))  # also use fftshift so the 0 frequency
        self.w_y = -np.fft.fftshift(np.fft.fftfreq(self.fft.shape[0]))  # lies in the middle of the returned array

        self.uf = self.w_x / self.pixelscale_x  # spatial frequencies in units of 1/radian
        self.vf = self.w_y / self.pixelscale_y
        return

    def redden(
        self,
        ebminv: float,
        reddening_law: str = constants.PROJECT_ROOT + "/utils/ISM_reddening/ISMreddening_law_Cardelli1989.dat",
    ) -> None:
        """Redden the image.

        Further reddens the model image according to the appropriate E(B-V) and a corresponding reddening law.

        Parameters
        ----------
        ebminv : float
            E(B-V) reddening factor to be applied.
        reddening_law : str, optional
            Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
            'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

        Returns
        -------
        None
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
        """Get a string with frequency domain info.

        Returns a string containing information on both the spatial frequency domain/sampling and the corresponding
        projected baselines.

        Returns
        -------
        info_str : str
            String containing frequency info which can be printed.
        """
        image_size_x = self.pixelscale_x * self.num_pix_x
        image_size_y = self.pixelscale_y * self.num_pix_y

        info_str = (
            f"================= \n"
            f"AMOUNT OF PIXELS: \n"
            f"================= \n"
            f"Amount of pixels considered in East-to-West direction (E-W): {self.num_pix_x} \n"
            f"Amount of pixels considered in South-to-North direction (S-N): {self.num_pix_y} \n\n"
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

    def half_light_radius(self) -> float:
        """Calculate the half light radius.

        Calculate the half light radius of the image by adding up the fluxes of pixels within increasing circular
        apertures.

        Returns
        -------
        hlr : float
            The half light radius in milli-arcseconds.

        Warnings
        --------
        Due to the implementation, the returned value's accuracy is inherently
        limited by the model image pixelscale. Note also that the radius is calculated in the image plane, and thus
        depends on e.g. inclination of the RT model.

        Notes
        -----
        If you want only the half light radius excluding the central source, e.g. the model disk in an
        MCFOST model, one should exclude the central source when reading in the image file (e.g. using
        ``disk_only=True`` with `read_image_fft_mcfost`).
        """
        # define numpy arrays with coordinates for pixel centres in milli-arcsecond
        coords_pix_x = (
            np.linspace(self.num_pix_x / 2 - 0.5, -self.num_pix_x / 2 + 0.5, self.num_pix_x)
            * self.pixelscale_x
            * constants.RAD2MAS
        )
        coords_pix_y = (
            np.linspace(self.num_pix_y / 2 - 0.5, -self.num_pix_y / 2 + 0.5, self.num_pix_y)
            * self.pixelscale_y
            * constants.RAD2MAS
        )
        coords_pix_mesh_x, coords_pix_mesh_y = np.meshgrid(coords_pix_x, coords_pix_y)  # create a meshgrid
        distances = np.sqrt(coords_pix_mesh_x**2 + coords_pix_mesh_y**2)  # calc dist pixels to origin

        rcont = 0  # variable denoting the radius of the circular aperature in milli-arcsecond
        fcont = 0  # variable containing the flux within ciruclar aperature contour
        rcont_interval = min(
            self.pixelscale_x * constants.RAD2MAS, self.pixelscale_y * constants.RAD2MAS
        )  # interval to increase aperature radius
        while fcont < 0.5 * self.ftot:
            rcont += rcont_interval
            fcont = np.sum(self.img[distances <= rcont])  # sum up all pixel fluxes in the aperature
        hlr = rcont  # set the contour radius to be the half light radius

        return hlr

    def diagnostic_plot(
        self,
        fig_dir: str | None = None,
        plot_vistype: Literal["vis2", "vis", "fcorr"] = "vis2",
        log_plotv: bool = False,
        log_ploti: bool = False,
        show_plots: bool = True,
    ) -> None:
        """Create diagnostic plots.

        Makes diagnostic plots showing both the model image and the FFT (squared) visibilities and complex phases.

        Parameters
        ----------
        fig_dir : str, optional
            Directory to store plots in.
        plot_vistype : {'vis2', 'vis', 'fcorr'}, optional
            Sets the type of visibility to be plotted. 'vis2' for squared visibilities, 'vis' for visibilities or
            'fcorr' for correlated flux in Jy.
        log_plotv : bool, optional
            Set to True for a logarithmic y-scale in the (squared) visibility plot.
        log_ploti : bool, optional
            Set to True for a logarithmic intensity scale in the model image plot.
        show_plots : bool, optional
            Set to False if you do not want the plots to be shown during your script run. Note that if True, this
            freezes further code execution until the plot windows are closed.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If an invalid `plot_vistype` is provided.
        """
        valid_vistypes = ["vis2", "vis", "fcorr"]
        if plot_vistype not in valid_vistypes:
            raise ValueError(f"Warning: Invalid plot_vistype '{plot_vistype}'. Valid options are: {valid_vistypes}.")

        baseu = self.uf / 1e6  # Baseline length u in MegaLambda
        basev = self.vf / 1e6  # Baseline length v in MegaLambda

        step_baseu = abs(baseu[1] - baseu[0])  # retrieve the sampling steps in u baseline length
        step_basev = abs(basev[1] - basev[0])  # retrieve the sampling steps in v baseline length

        # create plotting directory if it doesn't exist yet
        if fig_dir is not None:
            if not os.path.isdir(fig_dir):
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

        # intensity plotted in pixel scale
        # also set the extent of the image when you plot it, take care that the number of pixels is even
        img_plot = ax[0][0].imshow(
            self.img,
            cmap=constants.IMG_CMAP,
            norm=normi,
            extent=(
                self.num_pix_x / 2 + 0.5,
                -self.num_pix_x / 2 + 0.5,
                -self.num_pix_y / 2 + 0.5,
                self.num_pix_y / 2 + 0.5,
            ),
        )
        fig.colorbar(img_plot, ax=ax[0][0], label="$I$ (Jy/pixel)", fraction=0.046, pad=0.04)
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

        # set the complex phase
        cphi = np.angle(self.fft, deg=True)

        v2plot = ax[0][1].imshow(
            vis,
            cmap=constants.IMG_CMAP,
            norm=normv,
            extent=(
                self.num_pix_fft_x / 2 + 0.5,
                -self.num_pix_fft_x / 2 + 0.5,
                -self.num_pix_fft_y / 2 + 0.5,
                self.num_pix_fft_y / 2 + 0.5,
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
            cmap=constants.IMG_CMAP_DIVERGING,
            extent=(
                self.num_pix_fft_x / 2 + 0.5,
                -self.num_pix_fft_x / 2 + 0.5,
                -self.num_pix_fft_y / 2 + 0.5,
                self.num_pix_fft_y / 2 + 0.5,
            ),
            vmin=-max(abs(np.max(cphi)), abs(np.min(cphi))),
            vmax=max(abs(np.max(cphi)), abs(np.min(cphi))),
        )
        fig.colorbar(phi_plot, ax=ax[0][2], label=r"$\phi$ ($^\circ$)", fraction=0.046, pad=0.04)
        ax[0][2].axhline(y=0, lw=0.2, color="black")
        ax[0][2].axvline(x=0, lw=0.2, color="black")

        ax[0][2].set_title(r"Complex Phase $\phi$")
        ax[0][2].set_xlabel(r"$\leftarrow u$ (1/pixel)")
        ax[0][2].set_ylabel(r"$v \rightarrow$ (1/pixel)")

        # intensity plotted in angle scale
        img_plot = ax[1][0].imshow(
            self.img,
            cmap=constants.IMG_CMAP,
            aspect="auto",
            norm=normi,
            extent=(
                (self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                (-self.num_pix_x / 2) * self.pixelscale_x * constants.RAD2MAS,
                (-self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS,
                (self.num_pix_y / 2) * self.pixelscale_y * constants.RAD2MAS,
            ),
        )
        fig.colorbar(img_plot, ax=ax[1][0], label="$I$ (Jy/pixel)", fraction=0.046, pad=0.04)
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
            cmap=constants.IMG_CMAP,
            norm=normv,
            extent=(
                (self.num_pix_fft_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_fft_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_fft_y / 2 + 0.5) * step_basev,
                (self.num_pix_fft_y / 2 + 0.5) * step_basev,
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
            cmap=constants.IMG_CMAP_DIVERGING,
            extent=(
                (self.num_pix_fft_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_fft_x / 2 + 0.5) * step_baseu,
                (-self.num_pix_fft_y / 2 + 0.5) * step_basev,
                (self.num_pix_fft_y / 2 + 0.5) * step_basev,
            ),
        )
        fig.colorbar(phi_plot, ax=ax[1][2], label=r"$\phi$ ($^\circ$)", fraction=0.046, pad=0.04)
        ax[1][2].axhline(y=0, lw=0.2, color="black")
        ax[1][2].axvline(x=0, lw=0.2, color="black")
        ax[1][2].set_title(r"Complex Phase $\phi$")
        ax[1][2].set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        ax[1][2].set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

        # draw lines/cuts along which we will plot some curves
        ax[1][1].plot(
            np.zeros_like(basev[1 : int(self.num_pix_fft_y / 2) + 1]),
            basev[1 : int(self.num_pix_fft_y / 2) + 1],
            c="g",
            lw=2,
            ls="--",
        )
        ax[1][1].plot(
            baseu[1 : int(self.num_pix_fft_x / 2) + 1],
            np.zeros_like(baseu[1 : int(self.num_pix_fft_x / 2) + 1]),
            c="b",
            lw=2,
        )

        ax[1][2].plot(
            np.zeros_like(basev[1 : int(self.num_pix_fft_y / 2) + 1]),
            basev[1 : int(self.num_pix_fft_y / 2) + 1],
            c="g",
            lw=2,
            ls="--",
        )
        ax[1][2].plot(
            baseu[1 : int(self.num_pix_fft_y / 2) + 1],
            np.zeros_like(baseu[1 : int(self.num_pix_fft_y / 2) + 1]),
            c="b",
            lw=2,
        )
        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(
                os.path.join(fig_dir, f"fft2d_maps_{self.wavelength}mum.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )

        # Some plots of specific cuts in frequency space
        fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8))

        # Cuts of visibility and complex phase plot in function of baseline length
        # note we cut away the point furthest along positive u-axis since it contains a strong artefact due to
        # the FFT algorithm, otherwise we move down to spatial frequency 0
        vhor = vis[
            int(self.num_pix_fft_y / 2), 1 : int(self.num_pix_fft_x / 2) + 1
        ]  # extract (squared) visibility along u-axis
        phi_hor = cphi[int(self.num_pix_fft_y / 2), 1:]  # extract complex phase

        vver = vis[
            1 : int(self.num_pix_fft_y / 2) + 1, int(self.num_pix_fft_x / 2)
        ]  # extract (squared) visibility along u-axis
        phi_ver = cphi[1:, int(self.num_pix_fft_x / 2)]  # extract complex phase

        ax2[0].plot(
            baseu[1 : int(self.num_pix_fft_x / 2) + 1],
            vhor,
            c="b",
            label="along u-axis",
            lw=0.7,
            zorder=1000,
        )
        ax2[1].plot(baseu[1:], phi_hor, c="b", lw=0.7, zorder=1000)

        ax2[0].plot(
            basev[1 : int(self.num_pix_fft_y / 2) + 1],
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
            ax2[0].set_ylim(0.5 * np.min(np.append(vhor, vver)), 2 * np.max(np.append(vhor, vver)))
        else:
            ax2[0].axhline(y=0, c="k", lw=0.3, ls="--", zorder=0)
            ax2[0].set_ylim(np.min(np.append(vhor, vver)), 1.1 * np.max(np.append(vhor, vver)))

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
                os.path.join(fig_dir, f"fft1d_cuts_{self.wavelength}mum.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()
        return


def read_image_mcfost(img_path: str, padding: tuple[int, int] | None = None, disk_only: bool = False) -> Image:
    """Read in MCFOST model image.

    Retrieve image data from an MCFOST model image file and return it as an `Image` class instance.

    Parameters
    ----------
    img_path : str
        Path to an MCFOST output RT.fits.gz model image file.
    padding : tuple of int, optional
        Number of (x, y)-pixels to which an image should be 0-padded before performing the FFT. I.e.
        ``padding=(680, 540)`` will 0-pad an image to 680 and 540 pixels in the x and y dimensions, respectively.
        If smaller than the number of pixels already in the `img` array, no padding will be added in the respective
        dimension. These should both be even numbers!
    disk_only : bool, optional
        Set to True if you only want to read in the flux from the disk.

    Returns
    -------
    image : Image
        Image instance containing the information on the MCFOST RT image.

    Raises
    ------
    FileNotFoundError
        If the specified `img_path` does not exist.
    """
    dictionary = {}  # dictionary to construct Image instance

    az, inc = 0, 0  # only load the first azimuthal/inc value image in the .fits file

    # open the required fits file + get some header info
    hdul = fits.open(img_path)

    # read in the wavelength, pixelscale and 'window size' (e.g. size of axis images in radian)
    dictionary["wavelength"] = hdul[0].header["WAVE"]  # wavelength in micron
    dictionary["pixelscale_x"] = abs(hdul[0].header["CDELT1"]) * constants.DEG2RAD  # loads degrees, converted to radian
    dictionary["pixelscale_y"] = abs(hdul[0].header["CDELT2"]) * constants.DEG2RAD
    dictionary["num_pix_x"] = hdul[0].header["NAXIS1"]  # number of pixels
    dictionary["num_pix_y"] = hdul[0].header["NAXIS2"]

    img_array = hdul[0].data  # read in image data in lam x F_lam units W m^-2
    img_array = np.flip(img_array, axis=3)  # flip y-array (to match numpy axis convention)

    img_tot = img_array[0, az, inc, :, :]  # full total flux
    # disk flux (scattered starlight + direct thermal emission + scattered thermal emission)
    img_disk = img_array[6, az, inc, :, :] + img_array[7, az, inc, :, :]

    # Set all image-related class instance attributes below.

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

    # return an Image object
    image = Image(dictionary=dictionary, padding=padding)
    return image


def read_image_organic(img_path: str, padding: tuple[int, int] | None = None) -> tuple[Image, list[GeomComp]]:
    """Read in image from the ORGANIC image reconstruction code.

    Retrieve image data from an ORGANIC reconstructed image file and return it as a `Image` class instance and a
    list of geometric components (representing the SPARCO components in ORGANIC). Reads in the PCA filtered median
    image if present.

    Parameters
    ----------
    img_path : str
        Path to an MCFOST output RT.fits.gz model image file.
    padding : tuple of int, optional
        Number of (x, y)-pixels to which an image should be 0-padded before performing the FFT. I.e.
        ``padding=(680, 540)`` will 0-pad an image to 680 and 540 pixels in the x and y dimensions, respectively.
        If smaller than the number of pixels already in the `img` array, no padding will be added in the respective
        dimension. These should both be even numbers!

    Returns
    -------
    organic_model : tuple(Image, list(GeomComp))
        Returns a tuple containing the ORGANIC image and a list of Geometric components representing the
        SPARCO components used in ORGANIC.

    """
    return


def read_image_list(
    mod_dir: str,
    img_dir: str,
    read_method: Literal["mcfost"] = "mcfost",
    padding: tuple[int, int] | None = None,
    ebminv: float = 0.0,
    reddening_law: str = f"{constants.PROJECT_ROOT}/utils/ISM_reddening/ISMreddening_law_Cardelli1989.dat",
) -> list[Image] | None:
    """Read in multiple model image files into a list of `Image` objects.

    Function that takes the path to an model's directory and a subdirectory containing image files, and returns a
    list of `Image` objects representing those model images. They should thus represent the same underlying physical
    model, but imaged at different wavelengths.

    Parameters
    ----------
    mod_dir : str
        Parent directory of the RT model of interest.
    img_dir : str
        Subdirectory containing RT model images. All image files recursively found in the subdirectories of
        ``mod_dir+img_dir`` are read.
    read_method : {'mcfost'}, optional
        Type of method used to read in RT model images when creating `Image` class instances. Currently supports
        `'mcfost'`, in which case all files ending on the suffix 'RT.fits.gz' are read in, and `'organic'`.
    padding : tuple of int, optional
        Number of (x, y)-pixels to which an image should be 0-padded before performing the FFT. I.e.
        ``padding=(680, 540)`` will 0-pad an image to 680 and 540 pixels in the x and y dimensions, respectively.
        If smaller than the number of pixels already in the `img` array, no padding will be added in the respective
        dimension. These should both be even numbers!
    ebminv : float, optional
        E(B-V) of additional reddening to be applied to the model images. Only useful if the visibilities need to be
        expressed in correlated flux at some point.
    reddening_law : str, optional
        Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
        'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

    Returns
    -------
    img_ffts : list of Image
        List of Image objects representing all model image files found under ``mod_dir+img_dir``. Sorted by wavelength.

    Raises
    ------
    ValueError
        If an invalid `read_method` is provided.
    """
    valid_read_methods = ["mcfost", "organic"]
    if read_method not in valid_read_methods:
        raise ValueError(f"Warning: Invalid read_method '{read_method}'. Valid options are: {valid_read_methods}.")

    imgs = []  # list of Image objects to be held (1 element long in the case of monochr=True)
    wavelengths = []  # list of their wavelengths

    if read_method == "mcfost":  # different ways to read in model image file paths
        img_file_paths = sorted(glob.glob(f"{mod_dir}/{img_dir}/**/*RT.fits.gz", recursive=True))

    for img_path in img_file_paths:
        if read_method == "mcfost":  # choose reader function
            img = read_image_mcfost(img_path, padding=padding)

        img.redden(ebminv=ebminv, reddening_law=reddening_law)  # redden the Image object
        imgs.append(img)  # append to the list of Image objects
        wavelengths.append(img.wavelength)  # append wavelength

    wavelengths, imgs = list(zip(*sorted(zip(wavelengths, imgs))))  # sort the objects in wavelength

    return imgs


def image_fft_comp_vis_interpolator(
    img_ffts: list[Image],
    normalised: bool = False,
    interp_method: str = "linear",
) -> RegularGridInterpolator:
    """Create a regular grid interpolator for model image complex FFTs.

    Creates a `scipy RegularGridInterpolator` from model `Image` objects, which can be used to interpolate the complex
    visibility to different spatial frequencies than those returned by the FFT algorithm and, optionally,
    different wavelengths than those of the RT model images themselves.

    Parameters
    ----------
    img_ffts : list of Image
        List of Image objects to create an interpolator from. If the list has length one, i.e. a monochromatic model
        for the emission, the returned interpolator can only take the 2 spatial frequencies (units 1/Hz) as arguments.
        If the list contains multiple objects, i.e. a chromatic model for the emission, the interpolator will also be
        able to take wavelength (in micron) as an argument and will be able to interpolate along the wavelength
        dimension.
    normalised : bool, optional
        Set to True if you want the returned interpolator to produce normalised, non-absolute complex visibilities
        (for calculating e.g. squared visibilities). By default normalised = False, meaning the interpolator returns
        absolute complex visibilities, i.e. complex correlated fluxes (in units Jy).
    interp_method : str, optional
        Interpolation method used by the returned scipy RegularGridInterpolator. Can support 'linear', 'nearest',
        'slinear', 'cubic', 'quintic' or 'pchip'.

    Returns
    -------
    interpolator : scipy.interpolate.RegularGridInterpolator
        Interpolator for the model image FFTs. If ``len(img_ffts) == 1``, only takes the uv spatial frequencies (units
        1/rad) as arguments as follows: interpolator(v, u). If ``len(img_ffts) > 1``, then it also can interpolate
        between wavelengths (units micron) as follows: ``interpolator(wavelength, v, u)``.

    Raises
    ------
    ValueError
        If the length of img_ffts is less than 1.

    Warnings
    --------
    The interpolators order of arguments for `u` and `v` is reversed compared to their normal order, and the
    interpolator should be called upon as interpolator(v, u). This is because the 'first index' of an array
    representing an image in the ordinary python convention represents the y-axis, not the x-axis.

    The interpolator will throw errors if arguments outside their bounds are supplied! Expects, in case of multiple
    model images, that every image included has the same pixelscale and amount of pixels (in both x- and y-direction).
    """

    if len(img_ffts) == 1:  # single image -> monochromatic emission model
        img = img_ffts[0]
        wavelength, ftot, fft, uf, vf = (
            img.wavelength,
            img.ftot,
            img.fft,
            img.uf,
            img.vf,
        )
        if not normalised:  # create interpolator and normalize FFT to normalised complex visibilities if needed
            interpolator = RegularGridInterpolator((vf, uf), fft, method=interp_method)  # make interpol absolute FFT
        else:
            interpolator = RegularGridInterpolator((vf, uf), fft / ftot, method=interp_method)  # same normalized

    else:  # multiple images -> chromatic emission model
        img_wavelengths = []  # list of model image wavelengths in micron
        fft_chromatic = []  # 3d 'array' list to store the different model image FFTs accros wavelength

        for img in img_ffts:
            wavelength, ftot, fft, uf, vf = (
                img.wavelength,
                img.ftot,
                img.fft,
                img.uf,
                img.vf,
            )
            if not normalised:  # attach FFTs to chromatic list and normalize FFTs to complex visibilities if needed
                fft_chromatic.append(fft)  # store image's FFT in chromatic list
            else:
                fft_chromatic.append(fft / ftot)
            img_wavelengths.append(wavelength)  # store image wavelength in micron

        # sort lists according to ascending wavelength just to be sure (required for making the interpolator)
        img_wavelengths, fft_chromatic = list(zip(*sorted(zip(img_wavelengths, fft_chromatic))))

        # make interpolator from multiple FFTs, note this assumes all images have the same pixelscale
        # and amount of pixels (in both x and y directions)
        fft_chromatic = np.array(fft_chromatic)
        interpolator = RegularGridInterpolator((img_wavelengths, vf, uf), fft_chromatic, method=interp_method)
    return interpolator


def image_fft_ftot_interpolator(
    img_ffts: list[Image],
    interp_method: str = "linear",
) -> interp1d:
    """Create a regular grid interpolator for the total flux in model images.

    Creates a scipy interp1d object from a list of model Image objects, allowing to interpolate the total
    flux (F_nu format in unit Jansky) along the wavelength dimension.

    Parameters
    ----------
    img_ffts : list of Image
        List of `Image` objects to create an interpolator from. Must have length longer than one.
    interp_method : str, optional
        Interpolation method used by scipy's interp1d method. Default is `'linear'`. Can support 'linear', 'nearest',
        'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'.

    Returns
    -------
    interpolator : scipy.interpolate.interp1d
        Interpolator for the total flux in F_nu format and units Jy. Takes the wavelength in micron as its only
        argument.

    Raises
    ------
    Exception
        If the length of `img_ffts` is less than 2.
    """

    if len(img_ffts) < 2:
        raise Exception("Argumnent list img_ffts needs to contain at least 2 objects to build an interpolator.")

    img_wavelengths = []  # list of model image wavelengths in micron
    img_ftots = []  # list to store total F_nu fluxes in Jy

    for img in img_ffts:  # read in data from objects
        img_ftots.append(img.ftot)
        img_wavelengths.append(img.wavelength)

    img_wavelengths = np.array(img_wavelengths)
    img_ftots = np.array(img_ftots)

    # sort lists according to ascending wavelength just to be sure (required for making the interpolator)
    img_wavelengths, img_ftots = list(zip(*sorted(zip(img_wavelengths, img_ftots))))

    interpolator = interp1d(img_wavelengths, img_ftots, kind=interp_method, bounds_error=True)

    return interpolator


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import distroi.auxiliary.constants

    distroi.auxiliary.constants.FIG_OUTPUT_TYPE = "pdf"

    mod_dir = "/home/toond/Documents/phd/python/distroi/examples/models/IRAS08544-4431_test_model/"
    img_dir = "PIONIER/data_1.65/"
    img = read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz", disk_only=True, padding=None)
    print(img.freq_info())
    img.diagnostic_plot(fig_dir="/home/toond/Downloads/", log_plotv=True, show_plots=True)
