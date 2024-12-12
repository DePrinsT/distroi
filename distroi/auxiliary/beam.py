"""A module to calculate the resolution beam from the uv coverage.

Models the OI point spread function (PSF, i.e. 'dirty beam') with a Gaussian fit to the inner image regions
called the (clean) beam.
"""

from distroi.auxiliary import constants
from distroi.data import oi_container

import os

import numpy as np
from scipy.optimize import curve_fit

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

constants.set_matplotlib_params()  # set project matplotlib parameters


class Beam:
    """Represents a 2D elliptical Gaussian beam.

    Class containing information for a 2D Gaussian beam, typically acquired from a fit to the inner regions of a
    dirty beam (the formal interferometric PSF).

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys and values representing several instance variables described below.
        Should include 'sig_min', 'sig_maj' and 'pa'.

    Attributes
    ----------
    sig_min : float
        Standard deviation of the Gaussian along the minor axis.
    sig_maj : float
        Standard deviation of the Gaussian along the major axis.
    pa : float
        Position angle of the Gaussian's major axis, anticlockwise from North to East.
    fwhm_min : float
        Full-width-half-maximum (FWHM) of the Gaussian along the minor axis. This defines the resolution
        corresponding to the uv coverage along this axis.
    fwhm_maj : float
        FWHM of the Gaussian along the major axis. This defines the resolution corresponding to the uv coverage
        along this axis.
    """

    def __init__(self, dictionary):
        self.sig_min = dictionary["sig_min"]  # sigma along short axis in mas
        self.sig_maj = dictionary["sig_maj"]  # sigma y in mas
        self.pa = dictionary["pa"]  # position angle (PA) in degrees

        self.fwhm_min = 2 * np.sqrt(2 * np.log(2)) * self.sig_min  # FWHM
        self.fwhm_maj = 2 * np.sqrt(2 * np.log(2)) * self.sig_maj

    def plot(self) -> None:
        """Image plot of the beam.

        Makes a colour image plot of the Beam, including contours representing the sigma/FWHM levels.
        """
        # TODO actually implement this
        return


def oi_container_calc_gaussian_beam(
    container: oi_container.OIContainer,
    vistype: Literal["vis2", "vis"] = "vis2",
    make_plots: bool = False,
    fig_dir: str = None,
    show_plots: bool = False,
    num_res: int = 3,
    pix_per_res: int = 32,
) -> Beam:
    """Calculate the beam from an `OIContainer` object.

    Given an `OIContainer` instance and the uv frequencies to be used, calculates the clean beam Gaussian parameters by
    making a Gaussian fit to the dirty beam. The dirty beam acts as the interferometric point spread funtion (PSF)
    corresponding to the chosen uv coverage, by setting visibilities constant at the observed uv points and inverting
    the Fourier transform directly to the image plane.

    Parameters
    ----------
    container : OIContainer
        Container with observables for which we want to calculate the resolution corresponding to its uv coverage.
    vistype : {'vis2', 'vis', 'fcorr'}, optional
        Sets the uv coverage to be used for the Gaussian beam calculation. `'vis2'` for the coverage corresponding to
        the squared visibility measurements or `'vis'` for the uv coverage corresponding to the visibility/correlated
        flux measurements.
    make_plots : bool, optional
        Set to True to make plots of the dirty beam.
    fig_dir : str, optional
        Set to a directory in which you want the plots to be saved.
    show_plots : bool, optional
        Set to True if you want the generated plots to be shown in a window.
    num_res : int, optional
        The number of resolution elements to be included in the calculation. A resolution element is defined as
        1 / (2 x 'max_uv'), with max_uv the maximum norm of the probed uv frequency points. Set to 2 by default.
        Going above this can skew the Gaussian fit or cause it to fail, as the non-Gaussian behavior of the PSF
        becomes more apparent further away from the dirty beam center. It will also increase calculation time as O(n^2).
    pix_per_res : int, optional
        Amount of dirty beam pixels used per resolution element. This should be even. Set to 32 by default.
        Increasing this can significantly increase computation time (scales as O(n^2)).

    Returns
    -------
    gauss_beam : Beam
        `Beam` object containing the information of the Gaussian fit.

    Raises
    ------
    ValueError
        If an invalid vistype is provided or if `pix_per_res` is not even.
    """
    valid_vistypes = ["vis2", "vis", "fcorr"]
    # check for valid vistype
    if vistype not in valid_vistypes:
        raise ValueError(f"Warning: Invalid vistype '{vistype}'. Valid options are: {valid_vistypes}.")

    if pix_per_res % 2 != 0:
        raise ValueError("calc_gaussian_beam() currently only supports even values for 'pix_per_res'.")

    if vistype == "vis2":
        u = container.v2_uf
        v = container.v2_vf
    elif vistype == "vis" or vistype == "fcorr":
        u = container.v_uf
        v = container.v_vf

    max_uv_dist = np.max(np.sqrt(u**2 + v**2))  # max distance in 1/rad from origin, sets pixelscale for image space
    pix_res = (0.5 / max_uv_dist) * constants.RAD2MAS  # smallest resolution element (at Nyquist sampling)
    pixelscale = pix_res / pix_per_res  # overresolve the dirty beam so the image is clearer
    num_pix = num_res * pix_per_res
    fov = num_pix * pixelscale  # fov in mas

    # note we do not make x go from high to low in this array, since plotting with extent already does that for us
    x = np.linspace(-fov / 2 + 0.5 * pixelscale, fov / 2 - 0.5 * pixelscale, num_pix)  # get image pixel centres
    y = np.linspace(-fov / 2 + 0.5 * pixelscale, fov / 2 - 0.5 * pixelscale, num_pix)
    x, y = np.meshgrid(x, y)  # put into a meshgrid

    # calculate the inverse transform assuming delta functions in Fourier space (i.e. the dirty beam) at image positions
    img_dirty = np.zeros_like(x)
    for i in range(np.shape(img_dirty)[0]):
        for j in range(np.shape(img_dirty)[1]):
            # convert image positions to rad for multiplication with u, v (in 1/rad)
            img_dirty[i][j] = np.sum(np.real(np.exp(2j * np.pi * ((x[i][j] * u) + (y[i][j] * v)) * constants.MAS2RAD)))
    # normalize dirty image to maximum value (makes fitting easier)
    img_dirty /= np.max(img_dirty)

    # Fit a 2D Gaussian to the dirty beam
    # Initial guesses for amplitude, x0, y0, sig_min, sig_maj_min_sig_min, position angle and offset
    init_guess = [1, 0, 0, pix_res, 0.1 * pix_res, 0, 0]
    bounds = (
        [0, -np.inf, -np.inf, 0, 0, -90.01, -np.inf],
        [np.inf, np.inf, np.inf, np.inf, np.inf, 90.01, np.inf],
    )  # defined so sig_maj >= sig_min
    popt_and_cov = curve_fit(
        constants.gaussian_2d_elliptical_ravel, (x, y), np.ravel(img_dirty), p0=init_guess, bounds=bounds
    )
    popt = popt_and_cov[0]  # extract optimized parameter.

    # make beam object
    dictionary = {"sig_min": popt[3], "sig_maj": popt[3] + popt[4]}
    if popt[5] < 0:  # PA in degrees
        dictionary["pa"] = popt[5] + 180  # always set PA as positive (between 0 and 180)
    else:
        dictionary["pa"] = popt[5]

    gauss_beam = Beam(dictionary)  # create Beam object

    if make_plots:
        fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharey=True)
        # plot dirty beam
        ax[0][0].imshow(
            img_dirty,
            aspect="auto",
            cmap=constants.IMG_CMAP,
            extent=(
                (num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (num_pix / 2) * pixelscale,
            ),
        )
        ax[0][0].set_title("Dirty beam")
        ax[0][0].set_xlabel("E-W (mas)")
        ax[0][0].set_ylabel("S-N (mas)")
        ax[0][0].set_xlim((num_pix / 2) * pixelscale, (-num_pix / 2) * pixelscale)
        ax[0][0].set_ylim((-num_pix / 2) * pixelscale, (num_pix / 2) * pixelscale)
        ax[0][0].arrow(
            0.90,
            0.80,
            -0.1,
            0,
            color="white",
            transform=ax[0][0].transAxes,
            length_includes_head=True,
            head_width=0.015,
            zorder=2000,
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
            zorder=2000,
        )
        ax[0][0].text(0.92, 0.90, "N", color="white", transform=ax[0][0].transAxes)
        fit_text = ax[0][0].text(
            0.05,
            0.05,
            r"$\mathrm{FWHM}_{min}/2 = $"
            + f"{gauss_beam.fwhm_min/2:.3g} mas ; "
            + r"$\mathrm{FWHM}_{maj}/2 = $"
            + f"{gauss_beam.fwhm_maj/2:.3g} mas ; "
            + "\n"
            + "PA = "
            + f"{gauss_beam.pa:.4g}"
            + r"$^{\circ}$",
            color="black",
            transform=ax[0][0].transAxes,
            fontsize=9,
        )
        fit_text.set_bbox(dict(facecolor="white", alpha=0.5, edgecolor="black"))
        ax[0][0].axhline(y=0, lw=0.5, color="white", alpha=0.5, zorder=0)
        ax[0][0].axvline(x=0, lw=0.5, color="white", alpha=0.5, zorder=0)

        # plot Gaussian fit to the beam, note the output of gaussian_2d is 1D so needs to be reshaped
        img_fitted = np.reshape(constants.gaussian_2d_elliptical_ravel((x, y), *popt), np.shape(img_dirty))
        img_fit_plot = ax[0][1].imshow(
            img_fitted,
            aspect="auto",
            cmap=constants.IMG_CMAP,
            extent=(
                (num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (num_pix / 2) * pixelscale,
            ),
        )
        ax[0][1].set_title("Gaussian fit")
        ax[0][1].set_xlabel("E-W (mas)")
        ax[0][1].set_xlim((num_pix / 2) * pixelscale, (-num_pix / 2) * pixelscale)
        ax[0][1].set_ylim((-num_pix / 2) * pixelscale, (num_pix / 2) * pixelscale)
        ax[0][1].axhline(y=0, lw=0.5, color="white", alpha=0.5, zorder=0)
        ax[0][1].axvline(x=0, lw=0.5, color="white", alpha=0.5, zorder=0)

        # plot residuals
        img_res_plot = ax[1][0].imshow(
            img_dirty - img_fitted,
            aspect="auto",
            cmap="grey",
            extent=(
                (num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (-num_pix / 2) * pixelscale,
                (num_pix / 2) * pixelscale,
            ),
        )
        ax[1][0].set_title("Residuals")
        ax[1][0].set_xlabel("E-W (mas)")
        ax[1][0].set_ylabel("S-N (mas)")
        ax[1][0].set_xlim((num_pix / 2) * pixelscale, (-num_pix / 2) * pixelscale)
        ax[1][0].set_ylim((-num_pix / 2) * pixelscale, (num_pix / 2) * pixelscale)
        ax[1][0].axhline(y=0, lw=0.5, color="white", alpha=0.5, zorder=0)
        ax[1][0].axvline(x=0, lw=0.5, color="white", alpha=0.5, zorder=0)

        res_ellipse1 = Ellipse(
            xy=(0, 0),
            width=gauss_beam.fwhm_min / 2,
            height=gauss_beam.fwhm_maj / 2,
            angle=-gauss_beam.pa,
            edgecolor="b",
            lw=2.0,
            fc="none",
            alpha=1,
        )
        res_ellipse2 = Ellipse(
            xy=(0, 0),
            width=gauss_beam.fwhm_min / 2,
            height=gauss_beam.fwhm_maj / 2,
            angle=-gauss_beam.pa,
            edgecolor="b",
            lw=2.0,
            fc="none",
            alpha=1,
        )
        res_ellipse3 = Ellipse(
            xy=(0, 0),
            width=gauss_beam.fwhm_min / 2,
            height=gauss_beam.fwhm_maj / 2,
            angle=-gauss_beam.pa,
            edgecolor="b",
            lw=2.0,
            fc="none",
            alpha=1,
            label="FWHM",
        )
        ax[0][0].add_patch(res_ellipse1)
        ax[0][1].add_patch(res_ellipse2)
        ax[1][0].add_patch(res_ellipse3)
        ax[0][0].plot([], [], label=r"$\mathrm{FWHM}/2$ ellipse", color="b")
        ax[0][0].legend(loc="upper left", frameon=True, framealpha=0.5, fontsize="small")

        plt.tight_layout()  # colorbar after tight layout, otherwise it messes up the plot
        fig.colorbar(
            img_fit_plot,
            ax=ax[0].ravel().tolist(),
            label=r"$I_{dirty}/ \mathrm{max}(I_{dirty})$",
            pad=0.02,
        )
        fig.colorbar(
            img_res_plot,
            ax=ax[1][0],
            label=r"$I_{dirty}/ \mathrm{max}(I_{dirty})$",
            pad=0.04,
        )
        ax[1][1].remove()

        if fig_dir is not None:
            if not os.path.isdir(fig_dir):
                os.makedirs(fig_dir)
            # save if fig_dir not None
            plt.savefig(
                os.path.join(fig_dir, f"dirty_beam_fit.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()  # show plot if asked

    return gauss_beam
