"""
Contains a class to store optical inteferometric (OI) observables, taken using the same instrument setup, and additional
functions to take radiative transfer (RT) model images and convert them to interferometric observables at the spatial
frequencies of the observables.

NOTE: With the aim of flexibility in including multiple files and computational ease, this class does not fully reflect
all the infromation stored in the OIFITS file format. For example, the wavelengths and baselines/triplets at which
observables were taken is stored in the relevant observable properties. E.g. the wavelengths and baselines associated
to the OI_VIS2 measurements are stored as a np.ndarray inside the OIData.vis2 instance property itself, instead of in
a separate property mirroring the OI_WAVELENGTH and OI_ARRAY tables inside the OIFITS files.
"""

import glob
from astropy.io import fits

# TODO: add units everywhere


class OIData:
    """
    Class to describe optical interferometry data, allowing to represent simultaneously the data contained in multiple
    OIFits files observed using the same array and instrument setup. The constructor simply initiates all properties
    of the instance to None. Data is meant to be read in using the read_oifits method instead.

    Make sure the data in different files is consistent in terms of target and setup. E.g. if first file read in uses
    differential instead of absolute visibilities in the OI_VIS table, all other files should also contain differential
    visibilities. If you wish to model observations taken with different instrument setup simultaneously, different
    OIData objects should be created for each setup.

    :ivar dict header: Primary header dictionary. Taken from the first file that's read in.
    :ivar OITarget target: Object representing the OI_TARGET table of an OIFITS file. Taken from the first file
        that's read in.
    :ivar OIVis vis: Object representing data in the OI_VIS tables concatenated over the considered OIFITS files.
    :ivar OIVis2 vis2: Object representing data in the OI_VIS2 tables concatenated over the considered OIFITS files.
    :ivar OIVis2 vis2: Object representing data in the OI_VIS2 tables concatenated over the considered OIFITS files.
    :ivar OIT3 t3: Object representing data in the OI_T3 tables concatenated over the considered OIFITS files.
    :ivar OIFlux flux: Object representing data in the OI_FLUX tables concatenated over the considered OIFITS files.
    """

    def __init__(self) -> None:
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        self.header = None  # Primary header dictionary
        self.target = None  # OITarget instance
        self.vis = None  # OIVis object
        self.vis2 = None  # OIVis2 object
        self.t3 = None  # OIT3 object
        self.flux = None  # OIFlux object

    def read_oifits(self, data_dir: str, data_file: str, ft_only=False):
        """
        Reads in data from OIFits files. Includes the use of wildcards, e.g. *, to read in data from multiple files
        simultaneously.

        NOTE: The read-in metadata on the instrument setup, targer and array (i.e. the primary table header,
        OI_TARGET, OI_ARRAY) and choice of observables (absolute, differential, coherent flux, etc) are read in only
        from the first considered file. Make sure the data in simultaneously read-in files are thus consistent in terms
        of setup. This allows the consistent merging of OI_VIS, OI_VIS2, OI_T3 and OI_FLUX tables into single
        consistent Python objects. If you wish to model observations taken with different instrument setup
        simultaneously, different OIData objects should be created for each setup.
        """

        file_paths = sorted(glob.glob(f"{data_dir}/{data_file}"))  # OIFITS files

        # check if any files were found at all.
        if len(file_paths) == 0:
            Exception(f"No files found at {data_dir}/{data_file}.")

        for file_idx, filepath in enumerate(file_paths):
            hdul = fits.open(filepath)  # open oifits file

            ### from first file, read in and create primary header and OITarget object
            if file_idx == 0:
                ### primary header
                prim_hdu = hdul["Primary"]
                prim_hdr_keywords = list(prim_hdu.header.keys())  # get keywords
                prim_hdr_values = list(prim_hdu.header.values())  # get values
                self.header = dict(zip(prim_hdr_keywords, prim_hdr_values))  # assign primary header
                ###

                ### Initialize OITarget object
                for hdu in hdul:
                    if hdu.name == "OI_TARGET" and self.target is None:
                        self.target = OITarget()  # initialize empty OITarget object
                        # assign header
                        target_hdr_keywords = list(hdu.header.keys())
                        target_hdr_values = list(hdu.header.values())
                        self.target.header = dict(zip(target_hdr_keywords, target_hdr_values))  # assign primary header
                        # assign other table data
                        self.target.target_id = hdu.data["TARGET_ID"][0]  # assign target id
                        self.target.target = hdu.data["TARGET"][0]
                        self.target.equinox = hdu.data["EQUINOX"][0]
                        self.target.raep0 = hdu.data["RAEP0"][0]
                        self.target.decep0 = hdu.data["DECEP0"][0]
                        self.target.ra_err = hdu.data["RA_ERR"][0]
                        self.target.dec_err = hdu.data["DEC_ERR"][0]
                        self.target.sysvel = hdu.data["SYSVEL"][0]
                        self.target.veltyp = hdu.data["VELTYP"][0]
                        self.target.veldef = hdu.data["VELDEF"][0]
                        self.target.pmra = hdu.data["PMRA"][0]
                        self.target.pmdec = hdu.data["PMDEC"][0]
                        self.target.pmra_err = hdu.data["PMRA_ERR"][0]
                        self.target.pmdec_err = hdu.data["PMDEC_ERR"][0]
                        self.target.parallax = hdu.data["PARALLAX"][0]
                        self.target.para_err = hdu.data["PARA_ERR"][0]
                        self.target.spectyp = hdu.data["SPECTYP"][0]
                ###
            ###

            ### Continue for all other files

            ### Create a nested double dictionary based on the ARRNAME keyword and STA_INDEX of the telescope name
            # and stance name
            tel_name_dict = {}
            sta_name_dict = {}
            for hdu in hdul:
                if hdu.name == "OI_ARRAY":
                    tel_name_dict[hdu.header["ARRNAME"]] = dict(zip(hdu.data["STA_INDEX"], hdu.data["TEL_NAME"]))
                    sta_name_dict[hdu.header["ARRNAME"]] = dict(zip(hdu.data["STA_INDEX"], hdu.data["STA_NAME"]))
            ###

            ### Create a dictionary of the effective wavelength and bandwidth solutions based on INSNAME keyword
            eff_wave_dict = {}
            eff_band_dict = {}
            for hdu in hdul:
                if hdu.name == "OI_WAVELENGTH":
                    # skip table if needed (e.g. depending on science or fringe tracker being needed)
                    if (not ft_only and ("ft" in hdu.header["INSNAME"].lower())) or (
                        ft_only and ("ft" not in hdu.header["INSNAME"].lower())
                    ):
                        continue
                    else:
                        eff_wave_dict[hdu.header["INSNAME"]] = hdu.data["EFF_WAVE"]
                        eff_band_dict[hdu.header["INSNAME"]] = hdu.data["EFF_BAND"]
            ###

            # ### Initialize/extend OIVis, OIVis2, OIT3 and OIFlux objects
            # for hdu in hdul:
            #     if hdu.name == "OI_VIS":
            #         # skip table if needed (e.g. depending on science or fringe tracker being needed)
            #         if (not ft_only and ("ft" in hdu.header["INSNAME"].lower())) or (
            #             ft_only and ("ft" not in hdu.header["INSNAME"].lower())
            #         ):
            #             continue
            #         elif self.vis is None:  # if OIVis object not created yet, create it and assign header
            #             self.vis = OIVis()  # initialize empty OIVis object
            #             # assign header
            #             vis_hdr_keywords = list(hdu.header.keys())
            #             vis_hdr_values = list(hdu.header.values())
            #             self.vis.header = dict(zip(vis_hdr_keywords, vis_hdr_values))  # assign primary header

            #             # retrieve wavelength array of the data
            #             eff_wave_array = eff_wave_dict[hdu.header["INSNAME"]]
            #             eff_band_array = eff_band_dict[hdu.header["INSNAME"]]

            #             # assign other table data
            #             self.vis.time = np.repeat(hdu.data["TIME"], len(eff_wave_array))
            #             self.vis.mjd = np.repeat(hdu.data["MJD"], len(eff_wave_array))
            #             self.vis.int_time = np.repeat(hdu.data["INT_TIME"], len(eff_wave_array))
            #             self.vis.visamp = hdu.data["VISAMP"].flatten()
            #             self.vis.visamperr = hdu.data["VISAMPERR"].flatten()
            #             self.vis.visphi = hdu.data["VISPHIERR"].flatten()
            #             self.vis.visphierr = hdu.data["VISPHIERR"].flatten()
            #             if "RVIS" in hdu.columns.names:  # RVIS and IVIS are optional
            #                 self.vis.rvis = hdu.data["RVIS"].flatten()
            #                 self.vis.rviserr = hdu.data["RVISERR"].flatten()
            #             if "IVIS" in hdu.columns.names:
            #                 self.vis.ivis = hdu.data["IVIS"].flatten()
            #                 self.vis.iviserr = hdu.data["IVISERR"].flatten()
            #             self.vis.ucoord = np.repeat(hdu.data["UCOORD"], len(eff_wave_array))
            #             self.vis.vcoord = np.repeat(hdu.data["VCOORD"], len(eff_wave_array))
            #             self.vis.flag = hdu.data["FLAG"].flatten()

            #             print(filepath)
            #             print((self.vis.vcoord))

            #                 # assign added properties
            #         else:  # extend already existing OIVis object
            #             pass

            ###

                # print(
                #     self.target.header,
                #     "\n",
                #     self.target.target_id,
                #     self.target.target,
                #     self.target.equinox,
                #     self.target.raep0,
                #     self.target.decep0,
                #     self.target.ra_err,
                #     self.target.dec_err,
                #     self.target.sysvel,
                #     self.target.veltyp,
                #     self.target.veldef,
                #     self.target.pmra,
                #     self.target.pmdec,
                #     self.target.pmra_err,
                #     self.target.pmdec_err,
                #     self.target.parallax,
                #     self.target.para_err,
                #     self.target.spectyp,
                # )
                # print(
                #     type(self.target.target_id),
                #     type(self.target.target),
                #     type(self.target.equinox),
                #     type(self.target.raep0),
                #     type(self.target.decep0),
                # )

            hdul.close()  # close file

        pass


class OITarget:
    """
    Class to describe the OI_TARGET table of an OIFITS file.

    :ivar dict header: Header dictionary.
    :ivar int target_id: Target ID number.
    :ivar str target: Target name.
    :ivar float equinox: Equinox in years.
    :ivar float raep0: Target right ascension at equinox in deg.
    :ivar float decep0: Target declination at equinox in deg.
    :ivar float ra_err:  Error on right ascension.
    :ivar float dec_err: Error on declination.
    :ivar float sysvel:  Systemic radial velocity in m/s.
    :ivar str veltyp: Reference for radial velocity (e.g. "LSR").
    :ivar str veldef: Definition for radial velocity (e.g. "OPTICAL").
    :ivar float pmra: Right ascension proper motion in deg/yr.
    :ivar float pmdec: Declination proper motion in deg/yr.
    :ivar float pmra_err: Error on right ascension proper motion.
    :ivar float pmdec_err: Error on declination proper motion.
    :ivar float parallax: Parallax in degrees.
    :ivar float para_err: Error on parallax.
    :ivar str spectyp: Target spectral type.
    """

    def __init__(self) -> None:
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        self.header = None  # header dictionary
        self.target_id = None  # target ID
        self.target = None  # target name
        self.equinox = None  # equinox in years
        self.raep0 = None  # target raep0 right ascension in deg
        self.decep0 = None  # target decep0 declination in deg
        self.ra_err = None  # right ascension error
        self.dec_err = None  # declination error
        self.sysvel = None  # systemic radial velocity in m/s
        self.veltyp = None  # reference for radial velocity (e.g. "LSR")
        self.veldef = None  # definition for radial velocity (e.g. "OPTICAL")
        self.pmra = None  # proper motion right ascension in deg/yr
        self.pmdec = None  # proper motion declination in deg/yr
        self.pmra_err = None  # proper motion error right ascension
        self.pmdec_err = None  # proper motion error declination
        self.parallax = None  # parallax in deg
        self.para_err = None  # parallax error
        self.spectyp = None  # spectral type


class OIVis:
    """
    Class to describe the concatenated data contained in OI_VIS tables of multiple OIFITS files.

    NOTE: There correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).

    :ivar dict header:
    """

    def __init__(self):
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        ### OIFITS-based properties
        self.header = None  # header dictionary
        # properties with dimension equal to amount of considered data points
        self.time = None  # np.ndarray with time of observations in seconds
        self.mjd = None  # np.ndarray with Modified Julian Date of observations
        self.int_time = None  # np.ndarray with integration time in seconds
        self.visamp = None  # np.ndarray with visibility amplitude
        self.visamperr = None  # np.ndarray with visibility amplitude error
        self.visphi = None  # np.ndarray with visibility phase in degrees
        self.visphierr = None  # np.ndarray with visibility phase error in degrees
        self.rvis = None  # np.ndarray with real part of complex coherent flux, units defined in header
        self.rviserr = None  # np.ndarray with error on real part of complex coherent flux, units defined in header
        self.ivis = None  # np.ndarray with imaginary part of complex coherent flux, units defined in header
        self.iviserr = None  # np.ndarray with error on imaginary part of complex coherent flux, units defined in header
        self.ucoord = None  # np.ndarray with u coordinate of the data in m
        self.vcoord = None  # np.ndarray with v coordinate of the data in m
        self.flag = None  # boolean np.ndarray with flag status of the data
        ###

        ### Added properties for ease of use (e.g. filtering and calculations)
        self.baseline_dict = None  # dictionary from baseline index to string describing the baseline pair
        # properties with dimension equal to amount of considered data points
        self.baseline_idx = None  # np.ndarray with indices denoting to which basline pair a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        ###

        pass

    def filter():
        """
        Filter data based on user input range on quantity of interest or on if data has been flagged.
        """
        pass


class OIVis2:
    """
    Class to describe the concatenated data contained in OI_VIS2 tables of multiple OIFITS files.

    NOTE: There correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).


    :ivar dict header:
    """

    def __init__(self):
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        ### OIFITS-based properties
        self.header = None  # OIFITS table header dictionary
        # properties with dimension equal to amount of considered data points
        self.time = None  # np.ndarray with time of observations in seconds
        self.mjd = None  # np.ndarray with Modified Julian Date of observations
        self.int_time = None  # np.ndarray with integration time in seconds
        self.vis2data = None  # np.ndarray with squared visibility
        self.vis2data = None  # np.ndarray with squared visibility error
        self.ucoord = None  # np.ndarray with u coordinate of the data in m
        self.vcoord = None  # np.ndarray with v coordinate of the data in m
        self.flag = None  # boolean np.ndarray with flag status of the data
        ###

        ### Added properties for ease of use (e.g. filtering and calculations)
        self.baseline_dict = None  # dictionary from baseline index to string describing the baseline pair
        # properties with dimension equal to amount of considered data points
        self.baseline_idx = None  # np.ndarray with indices denoting to which basline pair a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        ###
        pass

    def filter():
        """
        Filter data based on user input range on quantity of interest or on if data has been flagged.
        """
    pass


class OIT3:
    """
    Class to describe the concatenated data contained in OI_T3 tables of multiple OIFITS files.

    NOTE: There correspondence to the OIFITS standard. For example, all triple product data is flattened into 1D arrays
    for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote the baseline
    triangles. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary. In addition, the wavelength information is stored in the 1D eff_wave array.

    :ivar dict header:
    """

    def __init__(self):
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.

        NOTE: There correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
        into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
        the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
        baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
        (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).
        """
        ### OIFITS-based properties
        self.header = None  # OIFITS table header dictionary
        # properties with dimension equal to amount of considered data points
        self.time = None  # np.ndarray with time of observations in seconds
        self.mjd = None  # np.ndarray with Modified Julian Date of observations
        self.int_time = None  # np.ndarray with integration time in seconds
        self.t3amp = None  # np.ndarray with triple product visibility amplitude
        self.t3amperr = None  # np.ndarray with triple product visibility amplitude error
        self.t3phi = None  # np.ndarray with triple product visibility phase (closure phase) in degrees
        self.t3phierr = None  # np.ndarray with triple product visibility phase (closure phase) error in degrees
        self.u1coord = None  # np.ndarray with u coordinate of the first baseline AB of the triangle in m
        self.v1coord = None  # np.ndarray with v coordinate of the first baseline AB of the triangle in m
        self.u2coord = None  # np.ndarray with u coordinate of the second baseline BC of the triangle in m
        self.v2coord = None  # np.ndarray with v coordinate of the second baseline BC of the triangle in m
        self.flag = None  # boolean np.ndarray with flag status of the data
        ###

        ### Added properties for ease of use (e.g. filtering and calculations)
        self.baseline_dict = None  # dictionary from baseline index to string describing the baseline triangle
        # properties with dimension equal to amount of considered data points
        self.baseline_idx = None  # np.ndarray with indices denoting to which basline triangle a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        self.u3coord = None  # np.ndarray with u coordinate of the third baseline AC of the triangle in m
        self.v3coord = None  # np.ndarray with v coordinate of the third baseline AC of the triangle in m
        self.u1fcoord = None  # u coordinate 1st baseline in frequency space, units of cycle per radian
        self.v1fcoord = None  # v coordinate 1st baseline in frequency space, units of cycle per radian
        self.u2fcoord = None  # u coordinate 2nd baseline in frequency space, units of cycle per radian
        self.v2fcoord = None  # v coordinate 2nd baseline in frequency space, units of cycle per radian
        self.u3fcoord = None  # u coordinate 3d baseline in frequency space, units of cycle per radian
        self.v3fcoord = None  # v coordinate 3d baseline in frequency space, units of cycle per radian
        ###
        pass

    def filter():
        """
        Filter data based on user input range on quantity of interest or on if data has been flagged.
        """
    pass

class OIFlux:
    """
    Class to describe the concatenated data contained in OI_FLUX tables of multiple OIFITS files.

    NOTE: There correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the telescopes. Instead, the baseline pair corresponding to a datapoint is encoded in the telescope_idx array and
    the telescope_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).

    :ivar dict header:
    """

    def __init__(self):
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        ### OIFITS-based properties
        self.header = None  # OIFITS table header dictionary
        # properties with dimension equal to amount of considered data points
        self.time = None  # np.ndarray with time of observations in seconds
        self.mjd = None  # np.ndarray with Modified Julian Date of observations
        self.int_time = None  # np.ndarray with integration time in seconds
        self.fluxdata = None  # np.ndarray (units as specified in header)
        self.fluxerr = None  # np.ndarray with error on flux (units in header)
        self.flag = None  # boolean np.ndarray with flag status of the data
        ###

        ### Added properties for ease of use (e.g. filtering and calculations)
        self.telescope_dict = None  # dictionary from baseline index to string describing the used telescope
        # properties with dimension equal to amount of considered data points
        self.telescope_idx = None  # np.ndarray with indices denoting to which telescope a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        ###
        pass

    def filter():
        """
        Filter data based on user input range on quantity of interest or on if data has been flagged.
        """
    pass


if __name__ == "__main__":
    import numpy as np
    from astropy.io import fits

    data_dir = (
        "/home/toond/Documents/phd/data/IRAS08544-4431/" "radiative_transfer_modelling_corporaal_et_al2023/MATISSE_L"
    )
    data_file = "2019-04-29T012641_IRAS08544-4431_K0G2D0J3_IR-LM_LOW_cal_oifits_0.fits"

    # hdul = fits.open(f"{data_dir}/{data_file}")
    # prim_hdr_keywords = list(hdul["Primary"].header.keys())
    # prim_hdr_values = list(hdul["Primary"].header.values())
    # print(dict(zip(prim_hdr_keywords, prim_hdr_values)))

    oidata = OIData()
    oidata.read_oifits(data_dir=data_dir, data_file=data_file)
