"""A module to handle spectral energy distributions (SEDs)."""

from distroi.auxiliary import constants

import os

import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.optimize import minimize

import matplotlib.pyplot as plt

constants.set_matplotlib_params()  # set project matplotlib parameters


class SED:
    """Contains information on an SED.

    Parameters
    ----------
    dictionary : dict
        Dictionary containing keys and values representing several instance variables described below. Should include
        `'wavelengths'`, `'flam'`, and `'flam_err'`. The other required instance variables are set automatically through
        `add_freq_vars`.

    Attributes
    ----------
    wavelengths : np.ndarray
        1D array containing the wavelengths in micron.
    frequencies : np.ndarray
        1D array containing the frequencies in Hz.
    flam : np.ndarray
        1D array containing the flux in F_lam format, the unit is erg s^-1 cm^-2 micron^-1.
    flam_err : np.ndarray
        1D array containing the errors on flam_err (set to 0 if reading in a model SED).
    fnu : np.ndarray
        1D array containing the flux in F_nu format, the unit is Jansky (Jy).
    fnu_err : np.ndarray
        1D array containing the error on F_nu.
    bands : list of str or None
        Optional list of strings containing the names of the associated photometric bands. Mostly useful when reading
        in observed SED data tables from e.g. VizieR.
    sources : list of str or None
        Similar to the 'bands' attribute, but listing the source catalogue.
    """

    def __init__(self, dictionary: dict[str, np.ndarray]):
        self.wavelengths = None  # wavelengths in micron
        self.frequencies = None  # frequencies in Hz
        self.flam = None  # wavelength-based flux densities (F_lambda) in erg s^-1 cm^-2 micron^-1
        self.flam_err = None  # error on flam
        self.fnu = None  # frequency-based flux densities (F_nu) in Jansky
        self.fnu_err = None  # error on fnu
        # TODO: add option for adding polarimetric state information of light, e.g. using self.stokes_type
        # optional
        self.bands = None  # photometric bands associated in case of an observed SED
        self.sources = None  # catalogues associated in case of an observed SED

        if dictionary is not None:
            self.wavelengths, self.flam, self.flam_err = (
                dictionary["wavelengths"],
                dictionary["flam"],
                dictionary["flam_err"],
            )

            if "bands" in dictionary.keys():
                self.bands = dictionary["bands"]
            if "sources" in dictionary.keys():
                self.sources = dictionary["sources"]

            # calculate and add frequency-based variables
            self.add_freq_vars()

    def redden(
        self,
        ebminv: float,
        reddening_law: str = constants.PROJECT_ROOT + "/utils/ISM_reddening/" "ISMreddening_law_Cardelli1989.dat",
    ) -> None:
        """Redden the SED.

        Further reddens the SED according to the appropriate E(B-V) and a corresponding reddening law.

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
        self.flam = constants.redden_flux(self.wavelengths, self.flam, ebminv, reddening_law=reddening_law)
        self.fnu = constants.redden_flux(self.wavelengths, self.fnu, ebminv, reddening_law=reddening_law)

        return

    def get_flux(
        self, x: np.ndarray | float, flux_form: str = "flam", interp_method: str = "linear"
    ) -> np.ndarray | float:
        """Get the flux at specified wavelengths or frequencies.

        Retrieve the flux at certain wavelengths/frequencies by interpolating the contained SED data.

        Parameters
        ----------
        x : np.ndarray or float
            Wavelengths/frequencies (in micron/Hz) at which to calculate the flux.
        flux_form : str, optional
            The format of the flux to be calculated. Options are `'flam'` (default) and `'lam_flam'`, as well as their
            frequency analogues `'fnu'` and `'nu_fnu'`. In case ``flux_form = 'flam'`` or ``'lam_flam'``, `x` is assumed
            to represent wavelengths, while in case of ``'fnu'`` and ``'nu_fnu'``, `x` is assumed to be frequencies.
        interp_method : str, optional
            Interpolation method used by scipy's interp1d method. Default is `'linear'`. Can support `'linear'`,
            `'nearest'`, `'nearest-up'`, `'zero'`, `'slinear'`, `'quadratic'`, `'cubic'`, `'previous'`, or `'next'`.

        Returns
        -------
        flux : np.ndarray or float
            The flux calculated at `x` using the reference wavelength/frequency and reference flux value. In case of
            ``flux_form='flam'``, output will be in units erg s^-1 cm^-2 micron^-1. In case of ``flux_form='fnu'``,
            output will be in units Jansky. In case of ``flux_form='lam_flam'``, units will be in erg s^-1 cm^-2.
            In case of ``flux_form='nu_fnu'``, units will be Jy Hz.

        Warnings
        --------
        This method throws an error if you try to retrieve fluxes outside the wavelength/frequency bounds.
        """
        # set what flux to interpolate (e.g. 'flam') and what quantity to interpolate in (wavelengths/frequencies)
        if flux_form == "flam":
            x_sed = self.wavelengths
            flux_sed = self.flam
        elif flux_form == "lam_flam":
            x_sed = self.wavelengths
            flux_sed = x_sed * self.flam
        elif flux_form == "fnu":
            x_sed = self.frequencies
            flux_sed = self.fnu
        elif flux_form == "nu_fnu":
            x_sed = self.frequencies
            flux_sed = x_sed * self.fnu

        interpolator = interp1d(x_sed, flux_sed, kind=interp_method, bounds_error=True)  # create interpolator
        flux = interpolator(x)  # calculate interpolated flux

        return flux

    def plot(
        self,
        fig_dir: str = None,
        flux_form: str = "lam_flam",
        log_plot: bool = True,
        show_plots: bool = True,
    ) -> None:
        """Make a scatter plot of the SED.

        Parameters
        ----------
        fig_dir : str, optional
            Directory to store plots in.
        flux_form : str, optional
            Format for the flux. By default, it is set to `'lam_flam'`, meaning we represent the flux in lam*F_lam
            format  (units erg s^-1 cm^-2). Analogously, other options are `'flam'` (erg s^-1 cm^-2 micron^-1),
            `'fnu'` (Jy) and `'nu_fnu'` (Jy Hz).
        log_plot : bool, optional
            Set to False if you want the plot axes to be in linear scale.
        show_plots : bool, optional
            Set to False if you do not want the plots to be shown during your script run. Note that if True, this
            freezes further code execution until the plot windows are closed.

        Returns
        -------
        None
        """
        # TODO: implement this, including the printing of optional source/catalog and photband names
        pass

    def add_freq_vars(self) -> None:
        """Calculate and set frequency-based attributes from the wavelength-based ones.

        Returns
        -------
        None
        """
        wavelengths_si = self.wavelengths * constants.MICRON2M  # wavelength in SI
        self.frequencies = constants.SPEED_OF_LIGHT / wavelengths_si  # set frequencies in Hz

        self.fnu = constants.flam_cgs_per_mum_to_fnu_jansky(self.flam, self.wavelengths)  # fnu in Jy
        self.fnu_err = constants.flam_cgs_per_mum_to_fnu_jansky(self.flam_err, self.wavelengths)  # fnu_err in Jy

        return


def read_sed_mcfost(sed_path: str, star_only: bool = False) -> SED:
    """Retrieve SED data from an MCFOST model SED.

    Parameters
    ----------
    sed_path : str
        Path to an MCFOST output sed_rt.fits.gz model SED file.
    star_only : bool, optional
        Set to True if you only want to read in the flux from the star.

    Returns
    -------
    sed : SED
        `SED` instance containing the information on the MCFOST model SED. Note that the errors on the flux `flam_err`
        are set to zero (since this is a model SED).
    """
    dictionary = {}  # dictionary to construct SED instance

    az, inc = 0, 0  # only load the first azimuthal/inc value image in the .fits file

    # open the required ray-traced SED fits file
    hdul = fits.open(sed_path)
    # read in entire sed array containing all components and corresponding wavelength array (in second HUD)
    sed_array = hdul[0].data  # lam*F_lam in SI (W m^-2)
    wavelengths = np.array(hdul[1].data) * constants.MICRON2M  # wavelengths in SI units

    dictionary["wavelengths"] = wavelengths * constants.M2MICRON  # store SED object wavelengths in micron
    if not star_only:
        flam = np.array(sed_array[0, az, inc, :]) / wavelengths  # flam in SI units (W m^-2 m^-1)
        dictionary["flam"] = flam * constants.WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON  # store in erg s^-1 cm^-2 micron^-1
    else:
        flam = np.array(sed_array[1, az, inc, :]) / wavelengths  # single out star only, flam in SI units (W m^-2 m^-1)
        dictionary["flam"] = flam * constants.WATT_PER_M2_M_2ERG_PER_S_CM2_MICRON  # store in erg s^-1 cm^-2 micron^-1
    dictionary["flam_err"] = np.zeros_like(flam)  # set errors to 0 since we're dealing with a model

    # return an SED object
    sed = SED(dictionary)
    return sed


# TODO: implement reader function for basic ViZier output


def read_sed_repo_phot(sed_path: str, wave_lims: tuple[float, float] = None) -> SED:
    """Retrieve SED data from a KU Leuven Insitute of Astronomy SED repository .phot file.

    Retrieve observed SED data stored in a .phot file as presented in e.g. the SED catalog presented in
    Kluska et al. 2022 (A&A, 658 (2022) A36). Such files are stored in the local system of KU Leuven's
    Institute of Astronomy.

    Parameters
    ----------
    sed_path : str
        Path to an MCFOST output sed_rt.fits.gz model SED file.
    wave_lims : tuple of float, optional
        The lower and upper wavelength limits in micron used when reading in data.

    Returns
    -------
    sed : SED
        `SED` instance containing the information on the MCFOST model SED.
    """
    dictionary = {}  # dictionary to construct SED instance

    # Open up the observed photometric data, note that the flux units are in erg s^-1 cm^-2 angstrom^-1 and wavelengths
    # in angstrom. We convert these below.
    df = pd.read_csv(
        sed_path,
        sep=r"\s+",
        header=3,
        names=[
            "meas",
            "e_meas",
            "flag",
            "unit",
            "photband",
            "source",
            "_r",
            "_RAJ2000",
            "_DEJ2000",
            "cwave",
            "cmeas",
            "e_cmeas",
            "cunit",
            "color",
            "include",
            "phase",
            "bibcode",
            "comments",
        ],
    )
    df = df[df["cwave"].notna()]  # filter out rows with NaN in the wavelength (these are often colour measurements)

    df["cwave"] *= constants.AA2MICRON  # convert from angstrom to micron
    df["cmeas"] *= constants.MICRON2AA  # convert from erg s^-1 cm^-2 angstrom^-1 to erg s^-1 cm^-2 micron^-1
    df["e_cmeas"] *= constants.MICRON2AA  # convert from erg s^-1 cm^-2 angstrom^-1 to erg s^-1 cm^-2 micron^-1

    if wave_lims is not None:
        df = df[df["cwave"] > wave_lims[0]]  # filter according to wavelength limits
        df = df[df["cwave"] < wave_lims[1]]

    # sort values according to ascending wavelength for convenience
    wavelengths, flam, flam_err, photband, source = list(
        zip(
            *sorted(
                zip(
                    df["cwave"],
                    df["cmeas"],
                    df["e_cmeas"],
                    df["photband"],
                    df["source"],
                )
            )
        )
    )

    dictionary["wavelengths"] = np.array(wavelengths)
    dictionary["flam"] = np.array(flam)
    dictionary["flam_err"] = np.array(flam_err)

    dictionary["bands"] = list(photband)
    dictionary["sources"] = list(source)

    # Return an SED object
    sed = SED(dictionary=dictionary)
    return sed


# TODO: add support for fitting reddening on Fnu, lam_Flam or nu_Fnu instead
# TODO: add support for different interpolation methods
def sed_chi2reddened(
    sed_obs: SED,
    sed_mod: SED,
    ebminv: float,
    reddening_law: str = f"{constants.PROJECT_ROOT}" f"/utils/ISM_reddening" f"/ISMreddening_law_" f"Cardelli1989.dat",
) -> float:
    """Get the chi2 between a data SED and a reddened model SED.

    Returns the chi2 between an RT model `SED` and an observed `SED` under a certain amount of additional reddening.
    Note that this doesn't actually redden any of the `SED` object class instances, only calculates the chi2 assuming
    the model `SED` were to be reddened.

    Parameters
    ----------
    sed_obs : SED
        Observed `SED`.
    sed_mod : SED
        RT model `SED`.
    ebminv : float
        E(B-V) reddening factor to be applied.
    reddening_law : str, optional
        Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
        'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

    Returns
    -------
    chi2 : float
        The chi2 value between the reddened model `SED` and the observed `SED`.
    """
    flam_obs = sed_obs.flam
    flam_obs_err = sed_obs.flam_err
    wavelengths_obs = sed_obs.wavelengths
    flam_mod = sed_mod.flam
    wavelengths_mod = sed_mod.wavelengths

    # get reddened model SED flux
    flam_mod_red = constants.redden_flux(wavelengths_mod, flam_mod, ebminv=ebminv, reddening_law=reddening_law)
    # interpolate the model to the wavelength values of the data
    f = interp1d(wavelengths_mod, flam_mod_red)
    flam_mod_red_interp = np.array(f(wavelengths_obs))
    # calculate chi2
    chi2 = np.sum((flam_obs - flam_mod_red_interp) ** 2 / flam_obs_err**2)

    return chi2


# TODO: add support for fitting reddening on Fnu, lam_Flam or nu_Fnu instead
# TODO: add support for different interpolation methods
def sed_reddening_fit(
    sed_obs: SED,
    sed_mod: SED,
    ebminv_guess: float,
    redden_mod: bool = True,
    reddening_law: str = f"{constants.PROJECT_ROOT}" f"/utils/ISM_reddening" f"/ISMreddening_law_" f"Cardelli1989.dat",
) -> tuple[float, float]:
    """Fits an additional reddening to make a model SED match an observed SED.

    Fits an additional reddening E(B-V) value to make a model `SED` match up to an observed `SED` as much as possible.
    In case of a successful fit, the model `SED` is subsequently reddened according to the fitted value of E(B-V) and
    the chi2 value between model and observations is returned.

    Parameters
    ----------
    sed_obs : SED
        Observed `SED`.
    sed_mod : SED
        Model `SED`.
    ebminv_guess : float
        Initial guess for the E(B-V) reddening factor.
    redden_mod : bool, optional
        Redden the model `SED` according to the fitted value. Set to True by default.
    reddening_law : str, optional
        Path to the reddening law to be used. Defaults to the ISM reddening law by Cardelli (1989) in DISTROI's
        'utils/ISM_reddening folder'. See this file for the expected formatting of your own reddening laws.

    Returns
    -------
    tuple of float
        The optimal E(B-V) and corresponding chi2 value between the reddened model `SED` and the observed `SED`.
    """
    par_min = minimize(
        lambda ebminv: sed_chi2reddened(sed_obs, sed_mod, ebminv, reddening_law=reddening_law),
        np.array(ebminv_guess),
    )
    ebminv_opt = par_min["x"][0]  # optimal value of E(B-V)
    chi2 = sed_chi2reddened(sed_obs, sed_mod, ebminv=ebminv_opt, reddening_law=reddening_law)
    if redden_mod:
        sed_mod.redden(ebminv=ebminv_opt, reddening_law=reddening_law)  # redden model SED according to fitted E(B-V)

    return (ebminv_opt, chi2)


def sed_plot_data_vs_model(
    sed_dat: SED,
    sed_mod: SED,
    fig_dir: str = None,
    flux_form: str = "lam_flam",
    log_plot: bool = True,
    show_plots: bool = True,
) -> None:
    """Plot an observed SED against a model SED.

    Plots the data (observed) SED against the model SED.

    Parameters
    ----------
    sed_dat : SED
        Data SED. Typically corresponds to observations.
    sed_mod : SED
        RT model SED.
    fig_dir : str, optional
        Directory to store plots in.
    flux_form : str, optional
        Format for the flux. By default, it is set to `'lam_flam'`, meaning we represent the flux in lam*F_lam format
        (units erg s^-1 cm^-2). Analogously, other options are `'flam'` (erg s^-1 cm^-2 micron^-1), `'fnu'` (Jy) and
        `'nu_fnu'` (Jy Hz).
    log_plot : bool, optional
        Set to False if you want the plot axes to be in linear scale.
    show_plots : bool, optional
        Set to False if you do not want the plots to be shown during your script run. Note that if True, this freezes
        further code execution until the plot windows are closed.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.errorbar(
        sed_dat.wavelengths,
        sed_dat.wavelengths * sed_dat.flam,
        sed_dat.wavelengths * sed_dat.flam_err,
        label="data",
        fmt="bd",
        mfc="white",
        capsize=5,
        zorder=100,
    )
    ax.plot(sed_mod.wavelengths, sed_mod.wavelengths * sed_mod.flam, ls="-", c="r", zorder=1)
    ax.set_title("SED")

    if flux_form == "lam_flam":
        ax.set_xlabel(r"$\lambda \, \mathrm{(\mu m)}$")
        ax.set_ylabel(r"$\lambda F_{\lambda} \, \mathrm{(erg \, cm^{-2} \, s^{-1})}$")
    elif flux_form == "nu_fnu":
        ax.set_xlabel(r"$\nu \, \mathrm{(Hz)}$")
        ax.set_ylabel(r"$\nu F_{\nu} \, \mathrm{(Hz \, Jy)}$")
    if log_plot:
        ax.set_xlim(0.5 * np.min(sed_dat.wavelengths), 2 * np.max(sed_dat.wavelengths))
        ax.set_ylim(
            0.5 * np.min(sed_dat.wavelengths * sed_dat.flam),
            2.0
            * max(
                np.max(sed_dat.wavelengths * sed_dat.flam),
                np.max(sed_mod.wavelengths * sed_mod.flam),
            ),
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
    plt.tight_layout()

    if fig_dir is not None:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(
            os.path.join(fig_dir, f"sed_comparison.{constants.FIG_OUTPUT_TYPE}"),
            dpi=constants.FIG_DPI,
            bbox_inches="tight",
        )
    if show_plots:
        plt.show()

    return


# if __name__ == "__main__":
#     import numpy
#     import matplotlib.pyplot as plt

# object_id = "HD93662"
# iras_id = "IRAS10456-5712"
# plotting = True
# low_wave_lims = [3, 4.2, 8]
# high_wave_lims = [4, 5, 13]

# sed_data = read_sed_repo_phot(f"/home/toond/Documents/phd/data/{object_id}/SED/{iras_id}.phot")
# for i in range(len(low_wave_lims)):
#     low_wave_lim = low_wave_lims[i]
#     high_wave_lim = high_wave_lims[i]
#     for i, wave in enumerate(sed_data.wavelengths):
#         if low_wave_lim <= wave <= high_wave_lim:
#             print(
#                 f"wavelength: \t {wave} \t flux: \t {sed_data.fnu[i]} \t eflux: \t {sed_data.fnu_err[i]}"
#                 f"\t photband: \t {sed_data.bands[i]} \t source: \t {sed_data.sources[i]} \t "
#             )
#     print("")
# if plotting:
#     fig, ax = plt.subplots()
#     ax.errorbar(sed_data.wavelengths, sed_data.fnu, sed_data.fnu_err, fmt="o", markersize=4)
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     plt.show()

# sed_data = read_sed_repo_phot("../examples/data/IRAS08544-4431/SED/IRAS08544-4431.phot")
# sed_model = read_sed_mcfost("../examples/models/IRAS08544-4431_test_model/data_th/sed_rt.fits.gz")
# sed_star = read_sed_mcfost("../examples/models/IRAS08544-4431_test_model/data_th/sed_rt.fits.gz", star_only=True)

# waves = np.linspace(0.2, 10, 10000)
# flux_interpol = sed_model.get_flux(x=waves, flux_form="flam", interp_method="cubic")
# plt.plot(waves, flux_interpol)
# plt.plot(sed_model.wavelengths, sed_model.flam, ls="--")
# plt.show()

# flux_interpol = sed_model.get_flux(x=waves, flux_form="lam_flam", interp_method="cubic")
# plt.plot(waves, flux_interpol)
# plt.plot(sed_model.wavelengths, sed_model.wavelengths * sed_model.flam, ls="--")
# plt.show()

# frequencies = constants.SPEED_OF_LIGHT / (waves * constants.MICRON2M)
# flux_interpol = sed_model.get_flux(x=frequencies, flux_form="fnu")
# plt.plot(frequencies, flux_interpol)
# plt.plot(sed_model.frequencies, sed_model.fnu, ls="--")
# plt.show()

# frequencies = constants.SPEED_OF_LIGHT / (waves * constants.MICRON2M)
# flux_interpol = sed_model.get_flux(x=frequencies, flux_form="nu_fnu")
# plt.plot(frequencies, flux_interpol)
# plt.plot(sed_model.frequencies, sed_model.frequencies * sed_model.fnu, ls="--")
# plt.show()

# ebminv_fitted, chi2_value = sed_reddening_fit(sed_data, sed_model, ebminv_guess=1.4, redden_mod=True)
# print(ebminv_fitted)
# sed_plot_data_vs_model(sed_data, sed_model, flux_form="lam_flam")
# plt.show()
