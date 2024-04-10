"""
Defines a class and the corresponding methods to load in and handle both observed and model spectral energy
distributions (SEDs).
"""
import os

from distroi import constants

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from distroi.constants import set_matplotlib_params

set_matplotlib_params()  # set project matplotlib parameters


class SED:
    """
    Class containing information on an SED. Can contain SED information either related to astronomical observations
    or model RT images.

    :param dict dictionary: Dictionary containing keys and values representing several instance variables described
        below. Should include 'wavelengths', 'flam', and 'flamerr'. The other required instance variables are set
        automatically through add_freq_vars().
    """

    def __init__(self, dictionary):
        """
        Constructor method. See class docstring for information on instance properties.
        """
        self.wavelengths = None  # wavelengths in micron
        self.frequencies = None  # frequencies in Hz
        self.flam = None  # wavelength-based flux densities (F_lambda) in erg s^-1 cm^-2 micron^-1
        self.flamerr = None  # error on flam
        self.fnu = None  # frequency-based flux densities (F_nu) in Jansky
        self.fnuerr = None  # error on fnu

        if dictionary is not None:
            self.wavelengths, self.flam, self.flamerr = (dictionary['wavelengths'], dictionary['flam'],
                                                         dictionary['flamerr'])
            # calculate and add frequency-based variables
            self.add_freq_vars()

    def add_freq_vars(self):
        """
        Calculate and set frequency-based instance variables.

        :rtype: None
        """
        return


def read_sed_mcfost(sed_path, star_only=False):
    """
    Retrieve SED data from an MCFOST model SED and return it as an SED class instance.

    :param str sed_path: Path to an MCFOST output sed_rt.fits.gz model SED file.
    :param bool star_only: Set to True if you only want to read in the flux from the star.
    :return sed: SED instance containing the information on the MCFOST model SED. Note that the errors on the flux
        'flamerr' are set to zero (since this is a model SED).
    :rtype: SED
    """
    #  TODO: implement
    # open the required ray-traced SED fits file, add a slash to the folder's name if necessary
    if folder[-1] != '/':
        folder += '/'
    hdul = fits.open(folder + 'data_th/sed_rt.fits.gz')
    # read in entire sed array and corresponding wavelength array (in second HUD)
    sed_array = hdul[0].data * 10 ** 3  # converted to cgs
    lam = np.array(hdul[1].data)
    # single out full sed lambda times flux values
    # this is only so that the non-infinite minimum
    # can be easily found for setting the y-axis limits
    full_sed = np.array(sed_array[0, az, inc, :])
    # single out the star only
    star_sed = np.array(sed_array[1, az, inc, :])

    return lam, full_sed, star_sed


def read_sed_repo_phot(sed_path, wave_lims=None):
    """
    Retrieve observed SED data stored in a .phot file from the SED repository presented in Kluska et al. 2022 (
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
    dictionary['flamerr'] = np.array(df['e_cmeas'])

    # Return an SED object
    sed = SED(dictionary=dictionary)
    return sed


def redden_flux(lam, flux, reddening_law_path, ebminv):
    """
    Takes an SED's flux values (typically an MCFOST model SED) and returns the SED after ISM redenning.

    Parameters:
        lam (numpy array): Array containing the wavelength values in micron.
        flux (numpy array): Flux values, in lam x F_lam format, to be redenned, in erg/s/cm2/micron
        reddening_law_path (str): Path to the ISM reddening law to be used. In a two-column format:
        1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
        Standard file used is the ISM law by Cardelli et al. 1989.
        ebminv (float): E(B-V) magnitude of the reddening to be applied.

    Returns:
        flux_reddened (numpy array): Reddened flux values, in lam x F_lam format, in erg/s/cm2/micron.
    """
    if ebminv == 0:
        return flux
    else:
        # read in the ISM reddening law wavelengths in Angström and A/E in magnitude
        df_law = pd.read_csv(reddening_law_path, header=2, names=['WAVE', 'A/E'],
                             sep=r'\s+', engine='python')
        # set wavelength to micrometer
        lam_law = np.array(df_law['WAVE']) * 10 ** -4
        ae_law = np.array(df_law['A/E'])
        # linearly interpolate A/E(B-V) values to used wavelengths
        f = interp1d(lam_law, ae_law, kind='linear', bounds_error=False, fill_value=0)
        ae = f(lam)
        flux_reddened = np.array(flux * 10 ** (-ae * ebminv / 2.5))

        return flux_reddened


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


def plot_sed_decomp(folderpath, reddening_law_path, redden_mag=0, fig_dir='./', az=0, inc=0):
    """
    Function to plot an SED directly from the MCFOST output in all its contributions
    (NOTE: no ISM reddening fitted to any data).

    Parameters:
        folderpath (str): Path to the previously ran MCFOST 'Thermal' run for which we want to make an img
        (e.g. the dir_out argument of run_mcfost_th()).
        reddening_law_path (str): Path to the ISM reddening law to be used. In a two-column format:
        1) wavelength (Angstrom)  2) A(wavelength)/E(B-V) (magn.).
        redden_mag (float): E(B-V) magnitude of applied ISM reddening.
        fig_dir (str): Path to directory where the resulting figure is saved.
        az (int): Number of the azimuthal viewing angle value considered (MCFOST can output results for
        multiple angles simultaneously). Default = 0 (1st value).
        inc (int): Number of the inclination viewing angle value considered (MCFOST can output results
        for multiple angles simultaneously). Default = 0 (1st value).

    Returns:
        Nothing. A saved plot of the results is saved in the specified directory.
    """
    # open the required ray-traced SED fits file
    hdul = fits.open(folderpath + '/data_th/sed_rt.fits.gz')
    # read in entire sed array and corresponding wavelength array (in second HUD). put flux in cgs units.
    sed_array = hdul[0].data * 10 ** 3
    lam = hdul[1].data
    # single out full sed lambda times flux values
    # this is only so that the non-infinite minimum
    # can be easily found for setting the y-axis limits
    full_sed = redden_flux(lam, sed_array[0, az, inc, :], reddening_law_path, redden_mag)
    # single out the star only
    star_sed = redden_flux(lam, sed_array[1, az, inc, :], reddening_law_path, redden_mag)
    # sed of starlight scattered from the disk
    # replaces zeros with 10**-40 so taking log space doesn't crash
    starscatter_sed = redden_flux(lam, sed_array[2, az, inc, :], reddening_law_path, redden_mag)
    starscatter_sed[starscatter_sed == 0] = 10 ** -40
    # sed of disk thermal emission
    # replaces zeros with 10**-40 so taking log space doesn't crash
    diskthermal_sed = redden_flux(lam, sed_array[3, az, inc, :], reddening_law_path, redden_mag)
    diskthermal_sed[diskthermal_sed == 0] = 10 ** -40
    # sed of scattered thermal disk emission
    thermalscatter_sed = redden_flux(lam, sed_array[4, az, inc, :], reddening_law_path, redden_mag)
    thermalscatter_sed[thermalscatter_sed == 0] = 10 ** -40
    # plotting
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(lam, full_sed, ls='-', c='k', label='full')
    ax.plot(lam, star_sed, ls='-', c='grey', label='star', alpha=0.6)
    ax.plot(lam, starscatter_sed, ls='--', c='b', label='starlight scattered', lw=0.8)
    ax.plot(lam, diskthermal_sed, ls='dotted', c='r', label='disk thermal', lw=0.8)
    ax.plot(lam, thermalscatter_sed, ls='-.', c='g', label='thermal scattered', lw=0.8)
    ax.set_title('MCFOST SED' + ' #inclination ' + str(inc) + ', #azimuthal viewing angle ' + str(az) +
                 ', E(B-V)=' + str(redden_mag))
    ax.set_xlabel(r"$\lambda \, \mathrm{[\mu m]}$")
    ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{[erg \, cm^{-2} \, s^{-1}]}$")
    ax.set_xlim(np.min(lam), np.max(lam))
    ax.set_ylim(full_sed[np.isfinite(full_sed)].min(), full_sed[np.isfinite(full_sed)].max() * 10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    # make directory to save the figure in if necessary
    if (fig_dir != './') and not os.path.exists(fig_dir):
        subprocess.Popen('mkdir ' + fig_dir, shell=True).wait()
    fig.savefig(fig_dir + '/SED_decomposed_' + '_inc' + str(inc) + '_az' + str(az) + '_ext' + str(redden_mag) + '.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    read_sed_repo_phot('../examples/data/IRAS0844-4431/SED/IRAS08544-4431.phot')
    plt.show()
