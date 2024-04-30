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

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class SED:
    """
    Class containing information on an SED. Can contain SED information either related to astronomical observations
    or model RT images.

    :param dict dictionary: Dictionary containing keys and values representing several instance variables described
        below. Should include 'wavelengths', 'flam', and 'flam_err'. The other required instance variables are set
        automatically through add_freq_vars().
    """

    def __init__(self, dictionary):
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

    def add_freq_vars(self):
        """
        Calculate and set frequency-based instance variables from the wavelength-based ones.

        :rtype: None
        """
        wavelengths = self.wavelengths * constants.MICRON2M  # wavelength in SI
        self.frequencies = constants.SPEED_OF_LIGHT / wavelengths  # set frequencies in Hz

        flam = self.flam * constants.ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # flam in SI
        flam_err = self.flam_err * constants.ERG_PER_S_CM2_MICRON_2WATT_PER_M2_M  # flam_err in SI
        fnu = flam * wavelengths**2 / constants.SPEED_OF_LIGHT  # fnu in SI units
        fnu_err = flam_err * wavelengths ** 2 / constants.SPEED_OF_LIGHT  # fnu_err in SI units

        self.fnu = fnu * constants.WATT_PER_M2_HZ_2JY  # set fnu in Jy
        self.fnu_err = fnu_err * constants.WATT_PER_M2_HZ_2JY  # set fnu_err in Jy

        return


def read_sed_mcfost(sed_path, star_only=False):
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
        flam = np.array(sed_array[1, az, inc, :])  # single out the star only, flam in SI units (W m^-2 m^-1)
        dictionary['flam'] = flam * constants.WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON  # store in erg s^-1 cm^-2 micron^-1
    dictionary['flam_err'] = np.zeros_like(flam)  # set errors to 0 since we're dealing with a model

    # return an SED object
    sed = SED(dictionary)
    return sed


def read_sed_repo_phot(sed_path, wave_lims=None):
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


# 'lam'=array of wavelengths of data 'flux'=flux values of data, lam_model=wavelengths of model,
# flux_model = model flux, flux errors of data, 'E'=E(B-V) magnitude 'path'=path to reddening law file
def chi2reddened(lam, flux, flux_error, lam_model, flux_model, reddening_law_path, ebminv):
    """
    Returns the chi2 value of an MCFOST model's SED versus an observed SED.

    Parameters:
        lam (numpy array): Array containing the observed data's wavelength values in micron.
        flux (numpy array): The data's flux values, in lam x F_lam format, in erg/s/cm2/micron.
        flux_error (numpy array): The data's error values, in lam x F_lam format, in erg/s/cm2/micron.
        lam_model (numpy array): Same as lam but for the model SED.
        flux_model (numpy array): Same as flux but for the model SED.
        reddening_law_path (str): Path to the ISM reddening law to be used. In a two-column format:
        1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
        Standard file used is the ISM law by Cardelli et al. 1989.
        ebminv (float): E(B-V) magnitude of the reddening to be applied.

    Returns:
        chi2 (float): The resulting chi2 value
    """
    # redden the model SED
    flux_model_red = redden_flux(lam_model, flux_model, reddening_law_path, ebminv)
    # interpolate the model to the wavelength values of the data
    f = interp1d(lam_model, flux_model_red)
    flux_model_red_interpol = np.array(f(lam))
    # calculate chi2
    chi2 = np.sum((flux - flux_model_red_interpol) ** 2 / flux_error ** 2)

    return chi2


def ism_reddening_fit(lam, flux, flux_error, lam_model, flux_model, reddening_law_path, ebminv_guess=1.4):
    """
    Fits the ISM E(B-V) value to make the model SED match up to the observations as much as possible.
    Returns the full reddened model SED (at the model's wavelengths),
    the fitted E(B-V) value and the resulting chi2.

    Parameters:
        lam (numpy array): Array containing the observed data's wavelength values in micron.
        flux (numpy array): The data's flux values, in lam x F_lam format, in erg/s/cm2/micron.
        flux_error (numpy array): The data's error values, in lam x F_lam format, in erg/s/cm2/micron.
        lam_model (numpy array): Same as lam but for the model SED.
        flux_model (numpy array): Same as flux but for the model SED.
        reddening_law_path (str): Path to the ISM reddening law to be used. In a two-column format:
        1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
        Standard file used is the ISM law by Cardelli et al. 1989.
        ebminv_guess (float): Initial E(B-V) guess to be used for the scipy minimization routine to fit ISM reddening.

    Returns:
        model_sed_full_red (numpy array): Array containing the model's reddened SED,
        using the fitted E(B-V) (lam x F_lam format, in erg/s/cm2/micron).
        ebminv_opt (float): The fitted E(B-V) value.
        chi2 (float): The corresponding chi2 value.
    """
    par_min = scipy.optimize.minimize(lambda x: chi2reddened(lam, flux, flux_error, lam_model, flux_model,
                                                             reddening_law_path, x), np.array(ebminv_guess))
    ebminv_opt = par_min["x"][0]
    chi2 = chi2reddened(lam, flux, flux_error, lam_model, flux_model, reddening_law_path, ebminv_opt)
    model_sed_full_red = redden_flux(lam_model, flux_model, reddening_law_path, ebminv_opt)

    return model_sed_full_red, ebminv_opt, chi2


def plot_reddened_model_fit(filepath_data, folder_mcfost, reddening_law_path, ebminv_guess, fig_dir='./', az=0, inc=0):
    """
    Takes the paths to both an SED data .phot file from the SED repository and the directory of an MCFOST model run.
    Then performs a fit for the remaining ISM reddening, and plots the resulting SEDs.

    Parameters:
        filepath_data (str): Path to the .phot filename.
        folder_mcfost (str): Path to the folder in which the MCFOST run results are stored.
        reddening_law_path (str): Path to the ISM reddening law to be used. In a two-column format:
        1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
        Standard file used is the ISM law by Cardelli et al. 1989.
        ebminv_guess (float): Initial E(B-V) guess to be used for the minimization routine.
        fig_dir (str): Path to directory where the resulting figure is saved.
        az (int): Number of the azimuthal viewing angle value considered (MCFOST can output results for
        multiple angles simultaneously). Default = 0 (1st value).
        inc (int): Number of the inclination viewing angle value considered (MCFOST can output results for
        multiple angles simultaneously). Default = 0 (1st value).

    Returns:
        Nothing. A saved plot of the results is saved in the specified directory.
    """
    # loading data
    data_wave, data_flux, data_err = read_sed_data(filepath_data)
    model_wave, model_flux, model_star_flux = read_sed_mcfost(folder_mcfost, az, inc)
    # fitting the ISM E(B-V)
    model_flux_red, ebminv_opt, chi2 = ism_reddening_fit(data_wave, data_flux, data_err, model_wave, model_flux,
                                                         reddening_law_path, ebminv_guess)
    model_star_flux_red = redden_flux(model_wave, model_star_flux, reddening_law_path, ebminv_opt)
    # plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.errorbar(data_wave, data_flux, data_err, label='data', fmt='bd', mfc='white', capsize=5, zorder=1000)
    ax.plot(model_wave, model_flux_red, ls='-', c='r', label='MCFOST SED reddened', zorder=1)
    ax.plot(model_wave, model_flux, ls='--', c='r', label='MCFOST SED no reddening', zorder=0, alpha=0.4)
    ax.plot(model_wave, model_star_flux_red, ls='-', c='k', label='STAR reddened', zorder=0)
    ax.plot(model_wave, model_star_flux, ls='--', c='k', label='STAR no reddening', alpha=0.4, zorder=0)
    # some nice labeling
    ax.set_xlabel(r"$\lambda \, \mathrm{[\mu m]}$")
    ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{[erg \, cm^{-2} \, s^{-1}]}$")
    ax.set_xlim(np.min(model_wave), np.max(model_wave))
    ax.set_ylim(model_flux[np.isfinite(model_flux)].min(), model_flux[np.isfinite(model_flux)].max() * 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('SED analysis')
    ax.text(0.05, 0.93, 'E(B-V) = ' + '{:.2f}'.format(ebminv_opt) + r'; $\chi^2_{red} = $' +
            '{:.2f}'.format(chi2 / (np.size(data_flux) - 1)), c='k',
            verticalalignment='bottom', horizontalalignment='left', transform=ax.transAxes, fontsize=17)
    ax.legend()
    plt.tight_layout()
    # make directory to save the figure in if necessary
    if (fig_dir != './') and not os.path.exists(fig_dir):
        subprocess.Popen('mkdir ' + fig_dir, shell=True).wait()
    fig.savefig(fig_dir + '/SED_analysis' + '_inc' + str(inc) + '_az' + str(az) + '.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    sed_data = read_sed_repo_phot('../examples/data/IRAS0844-4431/SED/IRAS08544-4431.phot')
    sed_model = read_sed_mcfost('../examples/models/IRAS08544-4431_test_model/data_th/sed_rt.fits.gz')
    plt.scatter(sed_data.wavelengths, sed_data.wavelengths * sed_data.flam)
    plt.plot(sed_model.wavelengths, sed_model.wavelengths * sed_model.flam)
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
