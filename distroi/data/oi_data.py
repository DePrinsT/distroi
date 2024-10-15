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

# TODO: add units everywhere


class OIData:
    """
    Class to describe optical interferometry data, allowing to represent simultaneously the data contained in multiple
    OIFits files observed using the same array and instrument setup.

    Make sure the data in different files is consistent in terms of target and setup. E.g. if first file read in uses
    differential instead of absolute visibilities in the OI_VIS table, all other files should also contain differential
    visibilities. If you wish to model observations taken with different instrument setup simultaneously, different
    OIData objects should be created for each setup.
    """

    def __init__(self) -> None:
        self.header = None  # Header dictionary
        self.target = None  # OITarget instance
        self.vis = None  # OIVis object
        self.vis2 = None  # OIVis2 object
        self.t3 = None  # OIT3 object
        pass

    def read_oifits(data_dir: str, data_file: str):
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
        pass


class OITarget:
    """
    :ivar dict header:
    :ivar int target_id:
    :ivar str target:
    :ivar int equinox:
    :ivar float raep0:
    :ivar float decep0:
    :ivar float ra_err:
    :ivar float dec_err:
    :ivar float sysvel:
    :ivar str veltyp:
    :ivar str veldef:
    :ivar float pmra:
    :ivar float pmdec:
    :ivar float pmra_err:
    :ivar float pmdec_err:
    :ivar float parallax:
    :ivar float para_err:
    :ivar str spectyp:
    """

    def __init__(self) -> None:
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.
        """
        self.header = None  # header dictionary
        self.target_id = None  # target ID
        self.target = None  # target name
        self.equinox = None  # equinox in years
        self.raep0 = None  # taregt raep0 right ascension in deg
        self.decep0 = None  # target decep0 declination in deg
        self.ra_err = None  # right ascension error
        self.dec_err = None  # declination error
        self.sysvel = None  # systematic velocity
        self.veltyp = None  # velocity type
        self.veldef = None  # velcoity definition
        self.pmra = None  # proper motion right ascension
        self.pmdec = None  # proper motion declination
        self.pmra_err = None  # proper motion error right ascension
        self.pmdec_err = None  # proper motion error declination
        self.parallax = None  # parallax
        self.para_err = None  # parallax error
        self.spectyp = None  # spectral type


class OIVis:
    """
    Constructor method. See class docstring for information on initialization parameters and instance properties.
    """

    def __init__(self):
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
        ###

        ### Added properties for ease of use
        # properties with dimension equal to amount of considered data points
        self.wavelength = None  # np.ndarray with wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.baseline = None  # string np.ndarray with the baseline pair used;  replaces link with OI_ARRAY tables
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        ###

        pass

    def filter_flagged():
        """
        Filter out flagged data.
        """
        pass

    def filter():
        """
        Filter data based on user input range on quantity of interest.
        """
        pass


class OIVis2:
    def __init__(self):
        ### OIFITS-based properties
        self.header = None  # header dictionary
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

        ### Added properties for ease of use
        # properties with dimension equal to amount of considered data points
        self.wavelength = None  # np.ndarray with wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.baseline = None  # string np.ndarray with the baseline pair used; replaces link with OI_ARRAY tables
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        ###
        pass

    def filter_flagged():
        """
        Filter out flagged data.
        """

    def filter():
        """
        Filter data based on user input range on quantity of interest or baseline.
        """


class OIT3:
    def __init__(self):
        ### OIFITS-based properties
        self.header = None  # header dictionary
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

        ### Added properties for ease of use
        # properties with dimension equal to amount of considered data points
        self.wavelength = None  # np.ndarray with wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.baseline = None  # string np.ndarray with the baseline trianlge used; replaces link with OI_ARRAY tables
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

    def filter_flagged():
        """
        Filter out flagged data.
        """

    def filter():
        """
        Filter data based on user input range on quantity of interest or baseline.
        """


class OIFlux:
    def __init__(self):
        pass


if __name__ == "__main__":
    import numpy as np

    a = np.array([0, 1])
    print(a)
