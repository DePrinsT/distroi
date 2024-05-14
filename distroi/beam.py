"""
Contains classes to represent the OI point spread function (PSF) and a Gaussian fit to it. The former is called the
'dirty beam', the latter just 'beam'. Methods to calculate the dirty beam from the uv coverage of an OIContainer object
and get the beam from a fit to the dirty beam's inner few resolution elements (a 'Gaussian beam') are included.
"""
from distroi import constants

import os

import numpy as np
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class Beam:
    """
    Class containing information for a 2D Gaussian beam, typically acquired from a fit to the inner regions of a
    dirty beam.

    :param dict dictionary: Dictionary containing keys and values representing several instance variables described
        below. Should include 'sig_min', 'sig_maj' and 'pa'.
    :ivar float sig_min: Standard deviation of the Gaussian along the minor axis.
    :ivar float sig_maj: Standard deviation of the Gaussian along the major axis.
    :ivar float pa: Position angle of the Gaussian's major axis, anticlockwise from North to East.
    :ivar float fwhm_min: Full-width-half-maximum (FWHM) of the Gaussian along the minor axis. This defines the
        resolution corresponding to the uv-coverage along this axis.
    :ivar float fwhm_maj: FWHM of the Gaussian along the major axis. This defines the resolution
        corresponding to the uv-coverage along this axis.
    """

    def __init__(self, dictionary):
        self.sig_min = dictionary['sig_min']  # sigma along short axis in mas
        self.sig_maj = dictionary['sig_maj']  # sigma y in mas
        self.pa = dictionary['pa']  # position angle (PA) in degrees

        self.fwhm_min = 2 * np.sqrt(2 * np.log(2)) * self.sig_min  # FWHM
        self.fwhm_maj = 2 * np.sqrt(2 * np.log(2)) * self.sig_maj

    def plot(self):
        """
        Makes a colour plot of the Beam, including contours representing the sigma/FWHM levels
        :return:
        """
        return


def gaussian_2d(points, amp, x0, y0, sig_min, sig_maj_min_sig_min, pa, offset):
    """
    Definition for calculating the value of a 2D Elliptical Gaussian at a given point. Gaussian defined by an amplitude,
     xy center, standard deviations along major/minor axis, a major axis position angle and an offset.

    :param tuple(float) points: 2D tuples describing the (x, y) points to be inserted. Note that positive x is defined
        as leftward and positive y as upward (i.e. the East and North repesctively in the OI convention).
    :param float amp: Amplitude of the Gaussian.
    :param float x0: x-coordinate center of the Guassian.
    :param float y0: y-coordinate center of the Gaussian.
    :param float sig_min: Standard deviation in the minor axis direction.
    :param sig_maj_min_sig_min: How much the standard deviation in major ellipse axis direction is greater than that of
        minor axis direction. Defined so it can always be greater than or equal to sig_min when used in
        scipy.optimize.curve_fit.
    :param float pa: Position angle of the Gaussian (i.e. the major axis direction) anti-clockwise, starting North
        (positive y).
    :param offset: Base level offset from 0.
    :return values: A raveled 1D array containg the values of the Gaussian calculated at the points.
    :rtype: np.ndarray
    """
    x, y = points  # unpack tuple point coordinates in OI definition
    theta = (pa * constants.DEG2RAD)

    sig_maj = sig_min + sig_maj_min_sig_min  # calculate std in y direction

    # set multiplication factors for representing the rotation matrix
    # note the matrix assumes positive x is to the right, so we also add a minus to the
    a = (np.cos(theta) ** 2) / (2 * sig_min ** 2) + (np.sin(theta) ** 2) / (2 * sig_maj ** 2)
    b = -(np.sin(2 * theta)) / (4 * sig_min ** 2) + (np.sin(2 * theta)) / (4 * sig_maj ** 2)
    c = (np.sin(theta) ** 2) / (2 * sig_min ** 2) + (np.cos(theta) ** 2) / (2 * sig_maj ** 2)
    values = offset + amp * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0)
                                      + c * ((y - y0) ** 2)))
    values = values.ravel()  # ravel to a 1D array, so it can be used in scipy curve fitting

    return values


def calc_gaussian_beam(container, vistype='vis2', make_plots=False, fig_dir=None, show_plots=False, num_res=2,
                       pix_per_res=16):
    """
    Given an OIContainer and the uv frequencies to be used, calculates the clean beam Gaussian parameters by making a
    Gaussian fit to the dirty beam. The dirty beam acts as the interferometric point spread funtion (PSF)
    corresponding to the chosen uv coverage, by setting visibilities constant at the observed uv points and inverting
    the Fourier transform directly to the image plane.

    :param OIContainer container: Container with observables for which we want to calculate the resolution corresponding
        to its uv coverage.
    :param str, optional vistype: Sets the uv coverage to be plotted. 'vis2' for the coverage corresponding to the
        squared visibility measurements or 'vis' for the uv coverage corresponding to the visibility/correlated flux
        measurements.
    :param bool make_plots: Set to True to make plots of the dirty beam.
    :param str fig_dir: Set to a directory in which you want the plots to be saved.
    :param show_plots: Set to True if you want the generated plots to be shown in a window.
    :param int num_res: The number of resolution elements to be included in the calculation. A resolution element is
        defined as 1 / 'max_uv', with max_uv the maximum norm of the uv frequency points. Set to 2 by default.
        Going abive this can skew the Gaussian fit or cause it to fail, as the non-Gaussian behavior of the PSF becomes
        more apparent further away from the dirty beam center. It will also increase calculation time as O(n^2).
    :param int pix_per_res: Amount of dirty beam pixels used per resolution element. This should be even.
        Set to 16 by default. Increasing this can significantly increase computation time (scales as O(n^2)).
    :return gauss_beam: Beam object containing the information of the Gaussian fit.
    :rtype: Beam

    """
    if pix_per_res % 2 != 0:
        print("calc_gaussian_beam() currently only supports even values for 'pix_per_res'. Function will return None!")
        return None

    if vistype == 'vis2':
        u = container.v2uf
        v = container.v2vf
    elif vistype == 'vis' or vistype == 'fcorr':
        u = container.vuf
        v = container.vvf

    max_uv_dist = np.max(np.sqrt(u ** 2 + v ** 2))  # max distance in 1/rad from origin, sets pixelscale for image space
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
            img_dirty[i][j] = np.sum(np.real(np.exp(2j * np.pi * ((x[i][j] * u) + (y[i][j] * v))
                                                    * constants.MAS2RAD)))
    # normalize dirty image to maximum value (makes fitting easier)
    img_dirty /= np.max(img_dirty)

    # Fit a 2D Gaussian to the dirty beam
    # Initial guesses for amplitude, x0, y0, sig_min, sig_maj_min_sig_min, position angle and offset
    init_guess = [1, 0, 0, pix_res, 0.1 * pix_res, 0, 0]
    bounds = ([0, -np.inf, -np.inf, 0, 0, -90.01, -np.inf],
              [np.inf, np.inf, np.inf, np.inf, np.inf, 90.01, np.inf])  # defined so sig_maj >= sig_min
    popt_and_cov = curve_fit(gaussian_2d, (x, y), np.ravel(img_dirty), p0=init_guess, bounds=bounds)
    popt, pcov = popt_and_cov[0], popt_and_cov[1]  # extract optimized parameters and covariance matrix

    if make_plots:
        fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        color_map = 'inferno_r'

        # plot dirty beam
        img_dirty_plot = ax[0].imshow(img_dirty, aspect='auto', cmap=color_map,
                                      extent=((num_pix / 2) * pixelscale,
                                              (-num_pix / 2) * pixelscale,
                                              (-num_pix / 2) * pixelscale,
                                              (num_pix / 2) * pixelscale))
        ax[0].set_title("Dirty beam")
        ax[0].set_xlabel("E-W [mas]")
        ax[0].set_ylabel("S-N [mas]")
        ax[0].arrow(0.90, 0.80, -0.1, 0, color='white', transform=ax[0].transAxes,
                    length_includes_head=True, head_width=0.015)  # draw arrows to indicate direction
        ax[0].text(0.78, 0.83, "E", color='white', transform=ax[0].transAxes)
        ax[0].arrow(0.90, 0.80, 0, 0.1, color='white', transform=ax[0].transAxes,
                    length_includes_head=True, head_width=0.015)
        ax[0].text(0.92, 0.90, "N", color='white', transform=ax[0].transAxes)
        ax[0].axhline(y=0, lw=0.5, color='white')
        ax[0].axvline(x=0, lw=0.5, color='white')

        # plot Gaussian fit to the beam, note the output of gaussian_2d is 1D so needs to be reshaped
        img_fitted = np.reshape(gaussian_2d((x, y), *popt), np.shape(img_dirty))
        img_fit_plot = ax[1].imshow(img_fitted, aspect='auto', cmap=color_map,
                                    extent=((num_pix / 2) * pixelscale,
                                            (-num_pix / 2) * pixelscale,
                                            (-num_pix / 2) * pixelscale,
                                            (num_pix / 2) * pixelscale))
        ax[1].set_title("Gaussian fit")
        ax[1].set_xlabel("E-W [mas]")
        ax[1].axhline(y=0, lw=0.5, color='white')
        ax[1].axvline(x=0, lw=0.5, color='white')

        plt.tight_layout()  # colorbar after tight layout, otherwise it messes up the plot
        fig.colorbar(img_fit_plot, ax=ax.ravel().tolist(), label=r'$I_{dirty}/ \mathrm{max}(I_{dirty})$', pad=0.02)

        if fig_dir is not None:
            # create plotting directory if it doesn't exist yet
            if not os.path.exists(fig_dir):
                os.makedirs(fig_dir)
            plt.savefig(f"{fig_dir}/dirty_beam_fit.png", dpi=300, bbox_inches='tight')  # save if fig_dir not None
        if show_plots:
            plt.show()  # show plot if asked

    dictionary = {'sig_min': popt[3], 'sig_maj': popt[3] + popt[4]}
    if popt[5] < 0:  # PA in degrees
        dictionary['pa'] = popt[5] + 180  # always set PA as positive (between 0 and 180)
    else:
        dictionary['pa'] = popt[5]

    gauss_beam = Beam(dictionary)  # create Beam object
    return gauss_beam


if __name__ == "__main__":
    from distroi import oi_observables
    from distroi import image_fft

    fig_dir = '/home/toond/Downloads/pionier_resolution'
    data_dir, data_file = '../examples/data/IRAS0844-4431/PIONIER/', '*.fits'
    container_data = oi_observables.read_oicontainer_oifits(data_dir, data_file)
    beam = calc_gaussian_beam(container_data, vistype='vis2', make_plots=True, show_plots=True, fig_dir=fig_dir,
                              num_res=2)

    # FFT test + output info on frequencies
    mod_dir = '~/Downloads'
    img_dir = '/data_1.65/'
    img = image_fft.read_image_fft_mcfost(img_path=f'{mod_dir}{img_dir}/RT.fits.gz', disk_only=True)
    img.diagnostic_plot(fig_dir=fig_dir, log_plotv=True, show_plots=True, beam=beam)

    # data_dir, data_file = '../examples/data/IRAS0844-4431/GRAVITY/', '*.fits'
    # container_data = oi_observables.read_oicontainer_oifits(data_dir, data_file)
    # calc_gaussian_beam(container_data, vistype='vis2', make_plots=True, show_plots=True)
    #
    # data_dir, data_file = '../examples/data/IRAS0844-4431/MATISSE_L/', '*.fits'
    # container_data = oi_observables.read_oicontainer_oifits(data_dir, data_file)
    # calc_gaussian_beam(container_data, vistype='vis2', make_plots=True, show_plots=True)
    #
    # data_dir, data_file = '../examples/data/IRAS0844-4431/MATISSE_N/', '*.fits'
    # container_data = oi_observables.read_oicontainer_oifits(data_dir, data_file)
    # calc_gaussian_beam(container_data, vistype='vis', make_plots=True, show_plots=True)
