"""
Defines a class and the corresponding methods to load in and handle both observed and model spectral energy
distributions (SEDs).
"""

from distroi import constants

import os

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class SED:
    """
    Class containing information on an SED. Can contain SED information either related to astronomical observations
    or model RT images.

    :param dict dictionary: Dictionary containing keys and values representing several instance variables described
        below. Should include 'wavelengths', 'flam', and 'flam_err'. The other required instance variables are set
        automatically through add_freq_vars().
    :ivar np.ndarray wavelengths: 1D array containing the wavelengths in micron.
    :ivar np.ndarray frequencies: 1D array containing the frequencies in Hz.
    :ivar np.ndarray flam: 1D array containing the flux in F_lam format, the unit is erg s^-1 cm^-2 micron^-1.
    :ivar np.ndarray flam_err: 1D array containing the errors on flam_err (set to 0 if reading in a model SED).
    :ivar np.ndarray fnu: 1D array containing the flux in F_nu format, the unit is Jansky (Jy).
    :ivar np.ndarray fnu_err: 1D array containing the error on F_nu.
    """

    def __init__(self, dictionary: dict[str, np.ndarray]):
        """
        Constructor method. See class docstring for information on instance properties.
        """
        self.wavelengths = None  # wavelengths in micron
        self.frequencies = None  # frequencies in Hz
        self.flam = None  # wavelength-based flux densities (F_lambda) in erg s^-1 cm^-2 micron^-1
        self.flam_err = None  # error on flam
        self.fnu = None  # frequency-based flux densities (F_nu) in Jansky
        self.fnu_err = None  # error on fnu

        if dictionary is not None:
            self.wavelengths, self.flam, self.flam_err = (dictionary['wavelengths'], dictionary['flam'],
                                                          dictionary['flam_err'])
            # calculate and add frequency-based variables
            self.add_freq_vars()

    def redden(self, ebminv: float, reddening_law: str = constants.PROJECT_ROOT + '/utils/ISM_reddening/'
                                                                                  'ISMreddening_law_Cardelli1989.dat')\
            -> None:
        """
        Further reddens the SED according to the approriate E(B-V) and a corresponding reddening law.

        :param float ebminv: E(B-V) reddening factor to be applied.
        :param str reddening_law: Path to the reddening law to be used. Defaults to the ISM reddening law by
            Cardelli (1989) in DISTROI's 'utils/ISM_reddening folder'. See this file for the expected formatting
            of your own reddening laws.
        :rtype: None
        """
        self.flam = constants.redden_flux(self.wavelengths, self.flam, ebminv, reddening_law=reddening_law)
        self.fnu = constants.redden_flux(self.wavelengths, self.fnu, ebminv, reddening_law=reddening_law)

        return

    def plot(self, fig_dir: str = None, flux_form: str = 'lam_flam', log_plot: bool = True, show_plots: bool = True)\
            -> None:
        """
        Make a scatter plot of the SED.

        :rtype: None
        """
        # todo: plot
        return

    def add_freq_vars(self) -> None:
        """
        Calculate and set frequency-based instance variables from the wavelength-based ones.

        :rtype: None
        """
        wavelengths = self.wavelengths * constants.MICRON2M  # wavelength in SI
        self.frequencies = constants.SPEED_OF_LIGHT / wavelengths  # set frequencies in Hz

        flam = self.flam * constants.ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # flam in SI
        flam_err = self.flam_err * constants.ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # flam_err in SI
        fnu = flam * wavelengths ** 2 / constants.SPEED_OF_LIGHT  # fnu in SI units
        fnu_err = flam_err * wavelengths ** 2 / constants.SPEED_OF_LIGHT  # fnu_err in SI units

        self.fnu = fnu * constants.WATT_PER_M2_HZ_2JY  # set fnu in Jy
        self.fnu_err = fnu_err * constants.WATT_PER_M2_HZ_2JY  # set fnu_err in Jy

        return


def read_sed_mcfost(sed_path: str, star_only: bool = False) -> SED:
    """
    Retrieve SED data from an MCFOST model SED and return it as an SED class instance.

    :param str sed_path: Path to an MCFOST output sed_rt.fits.gz model SED file.
    :param bool star_only: Set to True if you only want to read in the flux from the star.
    :return sed: SED instance containing the information on the MCFOST model SED. Note that the errors on the flux
        'flam_err' are set to zero (since this is a model SED).
    :rtype: SED
    """
    dictionary = {}  # dictionary to construct SED instance

    az, inc = 0, 0  # only load the first azimuthal/inc value image in the .fits file

    # open the required ray-traced SED fits file
    hdul = fits.open(sed_path)
    # read in entire sed array containing all components and corresponding wavelength array (in second HUD)
    sed_array = hdul[0].data  # lam*F_lam in SI (W m^-2)
    wavelengths = np.array(hdul[1].data) * constants.MICRON2M  # wavelengths in SI units

    dictionary['wavelengths'] = wavelengths * constants.M2MICRON  # store SED object wavelengths in micron
    if not star_only:
        flam = np.array(sed_array[0, az, inc, :]) / wavelengths  # flam in SI units (W m^-2 m^-1)
        dictionary['flam'] = flam * constants.WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON  # store in erg s^-1 cm^-2 micron^-1
    else:
        flam = np.array(sed_array[1, az, inc, :]) / wavelengths  # single out star only, flam in SI units (W m^-2 m^-1)
        dictionary['flam'] = flam * constants.WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON  # store in erg s^-1 cm^-2 micron^-1
    dictionary['flam_err'] = np.zeros_like(flam)  # set errors to 0 since we're dealing with a model

    # return an SED object
    sed = SED(dictionary)
    return sed


def read_sed_repo_phot(sed_path: str, wave_lims: tuple[float, float] = None) -> SED:
    """
    Retrieve observed SED data stored in a .phot file from the SED catalog presented in Kluska et al. 2022 (
    A&A, 658 (2022) A36). Such files are stored in the local system of KU Leuven's Institute of Astronomy.
    Return it as an SED class instance.

    :param str sed_path: Path to an MCFOST output sed_rt.fits.gz model SED file.
    :param tuple(float) wave_lims: The lower and upper wavelength limits in micron used when reading in data.
    :return sed: SED instance containing the information on the MCFOST model SED.
    :rtype: SED
    """
    dictionary = {}  # dictionary to construct SED instance

    # Open up the observed photometric data, note that the flux units are in erg s^-1 cm^-2 angstrom^-1 and wavelengths
    # in angstrom. We convert these below.
    df = pd.read_csv(sed_path, sep=r'\s+', header=3,
                     names=['meas', 'e_meas', 'flag', 'unit', 'photband', 'source',
                            '_r', '_RAJ2000', '_DEJ2000', 'cwave', 'cmeas', 'e_cmeas',
                            'cunit', 'color', 'include', 'phase', 'bibcode', 'comments'])
    df = df[df['cwave'].notna()]  # filter out rows with NaN in the wavelength (these are often colour measurements)

    df['cwave'] *= constants.AA2MICRON  # convert from angstrom to micron
    df['cmeas'] *= constants.MICRON2AA  # convert from erg s^-1 cm^-2 angstrom^-1 to erg s^-1 cm^-2 micron^-1
    df['e_cmeas'] *= constants.MICRON2AA  # convert from erg s^-1 cm^-2 angstrom^-1 to erg s^-1 cm^-2 micron^-1

    if wave_lims is not None:
        df = df[df['cwave'] > wave_lims[0]]  # filter according to wavelength limits
        df = df[df['cwave'] < wave_lims[1]]

    dictionary['wavelengths'] = np.array(df['cwave'])
    dictionary['flam'] = np.array(df['cmeas'])
    dictionary['flam_err'] = np.array(df['e_cmeas'])

    # Return an SED object
    sed = SED(dictionary=dictionary)
    return sed


# todo: add support for fitting reddening on Fnu, lam_Flam or nu_Fnu instead
def chi2reddened(sed_obs: SED, sed_mod: SED, ebminv: float, reddening_law: str = f'{constants.PROJECT_ROOT}'
                                                                                 f'/utils/ISM_reddening'
                                                                                 f'/ISMreddening_law_'
                                                                                 f'Cardelli1989.dat') -> float:
    """
    Returns the chi2 between an RT model SED and an observed SED under a certain amount of additional reddening.
    Note that this doesn't actually redden any of the SED objects, only calculates the chi2 assuming the model SED
    were to be reddened.

    :param SED sed_obs: Observed SED.
    :param SED sed_mod: RT model SED.
    :param float ebminv: E(B-V) reddening factor to be applied.
    :param str reddening_law: Path to the reddening law to be used. Defaults to the ISM reddening law by
        Cardelli (1989) in DISTROI's 'utils/ISM_reddening folder'. See this file for the expected formatting
        of your own reddening laws.
    :return chi2: The chi2 value between the reddened model SED and the observed SED.
    :rtype: float
    """
    flam_obs = sed_obs.flam
    flam_obs_err = sed_obs.flam_err
    wavelengths_obs = sed_obs.wavelengths
    flam_mod = sed_mod.flam
    wavelengths_mod = sed_mod.wavelengths

    # get reddened model SED flux
    flam_mod_red = constants.redden_flux(wavelengths_mod, flam_mod, ebminv=ebminv,
                                         reddening_law=reddening_law)
    # interpolate the model to the wavelength values of the data
    f = interp1d(wavelengths_mod, flam_mod_red)
    flam_mod_red_interp = np.array(f(wavelengths_obs))
    # calculate chi2
    chi2 = np.sum((flam_obs - flam_mod_red_interp) ** 2 / flam_obs_err ** 2)

    return chi2


# todo: add support for fitting reddening on Fnu, lam_Flam or nu_Fnu instead
def reddening_fit(sed_obs: SED, sed_mod: SED, ebminv_guess: float, reddening_law: str = f'{constants.PROJECT_ROOT}'
                                                                                        f'/utils/ISM_reddening'
                                                                                        f'/ISMreddening_law_'
                                                                                        f'Cardelli1989.dat')\
        -> tuple[float, float]:
    """
    Fits an additional reddening E(B-V) value to make a model SED match up to an observed SED as much as possible. In
    case of a successfull fit, the model SED is subsequently reddened according to the fitted value E(B-V) and the
    chi2 value between model and observations is returned.

    :param SED sed_obs: Observed SED.
    :param SED sed_mod: RT model SED.
    :param float ebminv_guess: Initial guess for the E(B-V) reddening factor.
    :param str reddening_law:
    :return tuple(ebminv_opt, chi2): The optimal E(B-V) and corresponding chi2 value between the reddened model SED and
        the observed SED.
    :rtype: tuple(float)
    """
    par_min = minimize(lambda ebminv: chi2reddened(sed_obs, sed_mod, ebminv, reddening_law=reddening_law),
                       np.array(ebminv_guess))
    ebminv_opt = par_min["x"][0]  # optimal value of E(B-V)
    chi2 = chi2reddened(sed_obs, sed_mod, ebminv=ebminv_opt, reddening_law=reddening_law)
    sed_mod.redden(ebminv=ebminv_opt, reddening_law=reddening_law)  # redden model SED according to the fitted E(B-V)

    return ebminv_opt, chi2


def plot_data_vs_model(sed_dat: SED, sed_mod: SED, fig_dir: str = None, flux_form: str = 'lam_flam',
                       sed_mod_alt: SED = None, alt_label: str = None, log_plot: bool = True,
                       show_plots: bool = True) -> None:
    """
    Plots the data (observed) SED against the model SED. Note that this function shares a name with a similar function
    in the oi_observables module. Take care with your namespace if you use both functions in the same script.

    :param SED sed_dat: Data SED. Typically corresponds to observations.
    :param SED sed_mod: RT model SED.
    :param str fig_dir: Directory to store plots in.
    :param str flux_form: Format for the flux. By default, it is set to 'lam_flam', meaning we represent the flux in
        lam*F_lam format (units erg s^-1 cm^-2). Analogously, other options are 'flam' (erg s^-1 cm^-2 micron^-1),
        'fnu' (Jy) and 'nu_fnu' (Jy Hz).
    :param SED sed_mod_alt: Alternative RT model SED included in the plot. This is useful, for instance, if you want
        to include the model SED with only the flux from the star in the plot (such SEDs can be created using e.g.
        read_sed_mcfost() with star_only=True).
    :param str alt_label: Label to be associated to sed_mod_alt in the plot's legend.
    :param bool log_plot: Set to False if you want the plot axes to be in linear scale.
    :param bool show_plots: Set to False if you do not want the plots to be shown during your script run.
        Note that if True, this freazes further code execution until the plot windows are closed.
    :rtype: None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(sed_dat.wavelengths, sed_dat.wavelengths * sed_dat.flam, sed_dat.wavelengths * sed_dat.flam_err,
                label='data', fmt='bd', mfc='white', capsize=5, zorder=100)
    ax.plot(sed_mod.wavelengths, sed_mod.wavelengths * sed_mod.flam, ls='-', c='r', zorder=1)
    ax.plot(sed_mod_alt.wavelengths, sed_mod_alt.wavelengths * sed_mod_alt.flam, ls='--', c='r', alpha=0.4, zorder=2)
    ax.set_title('SED')

    if flux_form == 'lam_flam':
        ax.set_xlabel(r"$\lambda \, \mathrm{(\mu m)}$")
        ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{(erg \, cm^{-2} \, s^{-1})}$")
    elif flux_form == 'nu_fnu':
        ax.set_xlabel(r"$\nu \, \mathrm{(Hz)}$")
        ax.set_ylabel(r"$\nu F_{\nu} \, \mathrm{(Hz \, Jy)}$")
    if log_plot:
        ax.set_xlim(0.5 * np.min(sed_dat.wavelengths), 2 * np.max(sed_dat.wavelengths))
        ax.set_ylim(0.5 * np.min(sed_dat.wavelengths * sed_dat.flam), 2.0 *
                    max(np.max(sed_dat.wavelengths * sed_dat.flam), np.max(sed_mod.wavelengths * sed_mod.flam)))
        ax.set_xscale('log')
        ax.set_yscale('log')
    plt.tight_layout()

    if fig_dir is not None:
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(f"{fig_dir}/sed_comparison.png", dpi=300, bbox_inches='tight')
    if show_plots:
        plt.show()

    return


# def plot_reddened_model_fit(filepath_data, folder_mcfost, reddening_law, ebminv_guess, fig_dir='./', az=0, inc=0):
#     """
#     Takes the paths to both an SED data .phot file from the SED repository and the directory of an MCFOST model run.
#     Then performs a fit for the remaining ISM reddening, and plots the resulting SEDs.
#
#     Parameters:
#         filepath_data (str): Path to the .phot filename.
#         folder_mcfost (str): Path to the folder in which the MCFOST run results are stored.
#         reddening_law (str): Path to the ISM reddening law to be used. In a two-column format:
#         1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
#         Standard file used is the ISM law by Cardelli et al. 1989.
#         ebminv_guess (float): Initial E(B-V) guess to be used for the minimization routine.
#         fig_dir (str): Path to directory where the resulting figure is saved.
#         az (int): Number of the azimuthal viewing angle value considered (MCFOST can output results for
#         multiple angles simultaneously). Default = 0 (1st value).
#         inc (int): Number of the inclination viewing angle value considered (MCFOST can output results for
#         multiple angles simultaneously). Default = 0 (1st value).
#
#     Returns:
#         Nothing. A saved plot of the results is saved in the specified directory.
#     """
#     # loading data
#     data_wave, data_flux, data_err = read_sed_data(filepath_data)
#     model_wave, model_flux, model_star_flux = read_sed_mcfost(folder_mcfost, az, inc)
#     # fitting the ISM E(B-V)
#     model_flux_red, ebminv_opt, chi2 = ism_reddening_fit(data_wave, data_flux, data_err, model_wave, model_flux,
#                                                          reddening_law, ebminv_guess)
#     model_star_flux_red = redden_flux(model_wave, model_star_flux, reddening_law, ebminv_opt)
#     # plotting
#     fig, ax = plt.subplots(figsize=(7, 7))
#     ax.errorbar(data_wave, data_flux, data_err, label='data', fmt='bd', mfc='white', capsize=5, zorder=1000)
#     ax.plot(model_wave, model_flux_red, ls='-', c='r', label='MCFOST SED reddened', zorder=1)
#     ax.plot(model_wave, model_flux, ls='--', c='r', label='MCFOST SED no reddening', zorder=0, alpha=0.4)
#     ax.plot(model_wave, model_star_flux_red, ls='-', c='k', label='STAR reddened', zorder=0)
#     ax.plot(model_wave, model_star_flux, ls='--', c='k', label='STAR no reddening', alpha=0.4, zorder=0)
#     # some nice labeling
#     ax.set_xlabel(r"$\lambda \, \mathrm{[\mu m]}$")
#     ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{[erg \, cm^{-2} \, s^{-1}]}$")
#     ax.set_xlim(np.min(model_wave), np.max(model_wave))
#     ax.set_ylim(model_flux[np.isfinite(model_flux)].min(), model_flux[np.isfinite(model_flux)].max() * 10)
#     ax.set_xscale('log')
#     ax.set_yscale('log')
#     ax.set_title('SED analysis')
#     ax.text(0.05, 0.93, 'E(B-V) = ' + '{:.2f}'.format(ebminv_opt) + r'; $\chi^2_{red} = $' +
#             '{:.2f}'.format(chi2 / (np.size(data_flux) - 1)), c='k',
#             verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=17)
#     ax.legend()
#     plt.tight_layout()
#     # make directory to save the figure in if necessary
#     if (fig_dir != './') and not os.path.exists(fig_dir):
#         subprocess.Popen('mkdir ' + fig_dir, shell=True).wait()
#     fig.savefig(fig_dir + '/SED_analysis' + '_inc' + str(inc) + '_az' + str(az) +
#                 '.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()
#
#     return


if __name__ == "__main__":
    sed_data = read_sed_repo_phot('../examples/data/IRAS08544-4431/SED/IRAS08544-4431.phot')
    sed_model = read_sed_mcfost('../examples/models/IRAS08544-4431_test_model/data_th/sed_rt.fits.gz')
    sed_star = read_sed_mcfost('../examples/models/IRAS08544-4431_test_model/data_th/sed_rt.fits.gz',
                               star_only=True)

    ebminv_fitted, chi2_value = reddening_fit(sed_data, sed_model, ebminv_guess=1.4)
    print(ebminv_fitted)
    plt.show()
    plot_data_vs_model(sed_data, sed_model, flux_form='lam_flam', sed_mod_alt=sed_star)
