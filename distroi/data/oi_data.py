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

from distroi.auxiliary import constants

import os

import glob
from astropy.io import fits
import matplotlib.pyplot as plt

from typing import Literal


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

    def read_oifits(self, data_dir: str, data_file: str, ft_only: bool = False) -> None:
        """
        Reads in data from OIFits files. Includes the use of wildcards to read in data from multiple files
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
            # TODO: add checks here to see if the data format is consistent between files

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

            ### Initialize/extend OIVis, OIVis2, OIT3 and OIFlux objects
            for hdu in hdul:
                # ignore fringe tracker/science data based on selected option
                if hdu.name in ["OI_VIS", "OI_VIS2", "OI_T3"]:
                    # skip table if needed (e.g. depending on science or fringe tracker being needed)
                    if (not ft_only and ("ft" in hdu.header["INSNAME"].lower())) or (
                        ft_only and ("ft" not in hdu.header["INSNAME"].lower())
                    ):
                        continue  # skip to next hdu in for loop

                # read OIVis data
                if hdu.name == "OI_VIS":
                    if self.vis is None:  # if OIVis object not created yet, create it and assign header
                        self.vis = OIVis()  # initialize empty OIVis object
                        # assign header
                        vis_hdr_keywords = list(hdu.header.keys())
                        vis_hdr_values = list(hdu.header.values())
                        self.vis.header = dict(zip(vis_hdr_keywords, vis_hdr_values))  # assign primary header

                    # retrieve wavelength array of the data
                    eff_wave_array = eff_wave_dict[hdu.header["INSNAME"]]
                    eff_band_array = eff_band_dict[hdu.header["INSNAME"]]

                    # number of wavelength channels
                    num_wave = len(eff_wave_array)

                    # assign table data
                    if self.vis.time is None:  # check if the property already exists or needs to be created first
                        self.vis.time = np.repeat(hdu.data["TIME"], num_wave)
                    else:
                        self.vis.time = np.append(self.vis.time, np.repeat(hdu.data["TIME"], num_wave))
                    if self.vis.mjd is None:
                        self.vis.mjd = np.repeat(hdu.data["MJD"], num_wave)
                    else:
                        self.vis.mjd = np.append(self.vis.mjd, np.repeat(hdu.data["MJD"], num_wave))
                    if self.vis.int_time is None:
                        self.vis.int_time = np.repeat(hdu.data["INT_TIME"], num_wave)
                    else:
                        self.vis.int_time = np.append(self.vis.int_time, np.repeat(hdu.data["INT_TIME"], num_wave))
                    if self.vis.visamp is None:
                        self.vis.visamp = hdu.data["VISAMP"].flatten()
                    else:
                        self.vis.visamp = np.append(self.vis.visamp, hdu.data["VISAMP"].flatten())
                    if self.vis.visamperr is None:
                        self.vis.visamperr = hdu.data["VISAMPERR"].flatten()
                    else:
                        self.vis.visamperr = np.append(self.vis.visamperr, hdu.data["VISAMPERR"].flatten())
                    if self.vis.visphi is None:
                        self.vis.visphi = hdu.data["VISPHI"].flatten()
                    else:
                        self.vis.visphi = np.append(self.vis.visphi, hdu.data["VISPHI"].flatten())
                    if self.vis.visphierr is None:
                        self.vis.visphierr = hdu.data["VISPHIERR"].flatten()
                    else:
                        self.vis.visphierr = np.append(self.vis.visphierr, hdu.data["VISPHIERR"].flatten())
                    if "RVIS" in hdu.columns.names:  # RVIS and IVIS are optional according to the OIFITS 2 standard
                        if self.vis.rvis is None:
                            self.vis.rvis = hdu.data["RVIS"].flatten()
                        else:
                            self.vis.rvis = np.append(self.vis.rvis, hdu.data["RVIS"].flatten())
                        if self.vis.rviserr is None:
                            self.vis.rviserr = hdu.data["RVISERR"].flatten()
                        else:
                            self.vis.rviserr = np.append(self.vis.rviserr, hdu.data["RVISERR"].flatten())
                    if "IVIS" in hdu.columns.names:
                        if self.vis.ivis is None:
                            self.vis.ivis = hdu.data["IVIS"].flatten()
                        else:
                            self.vis.ivis = np.append(self.vis.ivis, hdu.data["IVIS"].flatten())
                        if self.vis.iviserr is None:
                            self.vis.iviserr = hdu.data["IVISERR"].flatten()
                        else:
                            self.vis.iviserr = np.append(self.vis.iviserr, hdu.data["IVISERR"].flatten())
                    if self.vis.ucoord is None:
                        self.vis.ucoord = np.repeat(hdu.data["UCOORD"], num_wave)
                    else:
                        self.vis.ucoord = np.append(self.vis.ucoord, np.repeat(hdu.data["UCOORD"], num_wave))
                    if self.vis.vcoord is None:
                        self.vis.vcoord = np.repeat(hdu.data["VCOORD"], num_wave)
                    else:
                        self.vis.vcoord = np.append(self.vis.vcoord, np.repeat(hdu.data["VCOORD"], num_wave))
                    if self.vis.flag is None:
                        self.vis.flag = hdu.data["FLAG"].flatten()
                    else:
                        self.vis.flag = np.append(self.vis.flag, hdu.data["FLAG"].flatten())

                    # assign added properties
                    # retrieve baseline info
                    if self.vis.baseline_dict is None:
                        self.vis.baseline_dict = {}  # baseline dictionary
                        self.vis.baseline_idx = np.array([])  # baseline index array
                        baseline_indx = 0  # baseline index, initialized to zero
                    else:
                        baseline_indx = int(np.max(self.vis.baseline_idx)) + 1  # if already exists, start new index

                    num_baselines = 0  # number of considered baselines
                    sta_idx = hdu.data["STA_INDEX"]
                    for sta_indx_pair in sta_idx:
                        sta1 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_pair[0]]
                        sta2 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_pair[1]]
                        sta_string = sta1 + "-" + sta2
                        self.vis.baseline_dict[baseline_indx] = sta_string  # extend baseline dict
                        print(baseline_indx)

                        self.vis.baseline_idx = np.append(
                            self.vis.baseline_idx, np.repeat(baseline_indx, num_wave)
                        ).astype(int)  # extend baseline index array

                        baseline_indx += 1  # increment baseline index
                        num_baselines += 1  # increment number of baselines

                    # retrieve wavelength info
                    if self.vis.eff_wave is None:
                        self.vis.eff_wave = np.array([])  # effective wavelength array
                        self.vis.eff_wave = np.tile(eff_wave_array, num_baselines)
                    else:
                        self.vis.eff_wave = np.append(self.vis.eff_wave, np.tile(eff_wave_array, num_baselines))
                    if self.vis.eff_band is None:
                        self.vis.eff_band = np.array([])  # effective spectral bandwidth array
                        self.vis.eff_band = np.tile(eff_band_array, num_baselines)
                    else:
                        self.vis.eff_band = np.append(self.vis.eff_band, np.tile(eff_band_array, num_baselines))

                    # calculate spatial frequencies and other variables
                    self.vis.ufcoord = self.vis.ucoord / self.vis.eff_wave
                    self.vis.vfcoord = self.vis.vcoord / self.vis.eff_wave
                    self.vis.base = np.sqrt(self.vis.ucoord**2 + self.vis.vcoord)
                    self.vis.spat_freq = np.sqrt(self.vis.ufcoord**2 + self.vis.vfcoord)
                    #

                # read OIVis2 data
                if hdu.name == "OI_VIS2":
                    if self.vis2 is None:  # if OIVis object not created yet, create it and assign header
                        self.vis2 = OIVis2()  # initialize empty OIVis object
                        # assign header
                        vis2_hdr_keywords = list(hdu.header.keys())
                        vis2_hdr_values = list(hdu.header.values())
                        self.vis2.header = dict(zip(vis2_hdr_keywords, vis2_hdr_values))  # assign primary header

                    # retrieve wavelength array of the data
                    eff_wave_array = eff_wave_dict[hdu.header["INSNAME"]]
                    eff_band_array = eff_band_dict[hdu.header["INSNAME"]]

                    # number of wavelength channels
                    num_wave = len(eff_wave_array)

                    # assign table data
                    if self.vis2.time is None:  # check if the property already exists or needs to be created first
                        self.vis2.time = np.repeat(hdu.data["TIME"], num_wave)
                    else:
                        self.vis2.time = np.append(self.vis2.time, np.repeat(hdu.data["TIME"], num_wave))
                    if self.vis2.mjd is None:
                        self.vis2.mjd = np.repeat(hdu.data["MJD"], num_wave)
                    else:
                        self.vis2.mjd = np.append(self.vis2.mjd, np.repeat(hdu.data["MJD"], num_wave))
                    if self.vis2.int_time is None:
                        self.vis2.int_time = np.repeat(hdu.data["INT_TIME"], num_wave)
                    else:
                        self.vis2.int_time = np.append(self.vis2.int_time, np.repeat(hdu.data["INT_TIME"], num_wave))
                    if self.vis2.vis2data is None:
                        self.vis2.vis2data = hdu.data["VIS2DATA"].flatten()
                    else:
                        self.vis2.vis2data = np.append(self.vis2.vis2data, hdu.data["VIS2DATA"].flatten())
                    if self.vis2.vis2err is None:
                        self.vis2.vis2err = hdu.data["VIS2ERR"].flatten()
                    else:
                        self.vis2.vis2err = np.append(self.vis2.vis2err, hdu.data["VIS2ERR"].flatten())
                    if self.vis2.ucoord is None:
                        self.vis2.ucoord = np.repeat(hdu.data["UCOORD"], num_wave)
                    else:
                        self.vis2.ucoord = np.append(self.vis2.ucoord, np.repeat(hdu.data["UCOORD"], num_wave))
                    if self.vis2.vcoord is None:
                        self.vis2.vcoord = np.repeat(hdu.data["VCOORD"], num_wave)
                    else:
                        self.vis2.vcoord = np.append(self.vis2.vcoord, np.repeat(hdu.data["VCOORD"], num_wave))
                    if self.vis2.flag is None:
                        self.vis2.flag = hdu.data["FLAG"].flatten()
                    else:
                        self.vis2.flag = np.append(self.vis2.flag, hdu.data["FLAG"].flatten())

                    # assign added properties
                    # retrieve baseline info
                    if self.vis2.baseline_dict is None:
                        self.vis2.baseline_dict = {}  # baseline dictionary
                        self.vis2.baseline_idx = np.array([])  # baseline index array
                        baseline_indx = 0  # baseline index, initialized to zero
                    else:
                        baseline_indx = int(np.max(self.vis2.baseline_idx)) + 1  # if already exists, start new index

                    num_baselines = 0  # number of considered baselines
                    sta_idx = hdu.data["STA_INDEX"]
                    for sta_indx_pair in sta_idx:
                        sta1 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_pair[0]]
                        sta2 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_pair[1]]
                        sta_string = sta1 + "-" + sta2
                        self.vis2.baseline_dict[baseline_indx] = sta_string  # extend baseline dict
                        print(baseline_indx)

                        self.vis2.baseline_idx = np.append(
                            self.vis2.baseline_idx, np.repeat(baseline_indx, num_wave)
                        ).astype(int)  # extend baseline index array

                        baseline_indx += 1  # increment baseline index
                        num_baselines += 1  # increment number of baselines

                    # retrieve wavelength info
                    if self.vis2.eff_wave is None:
                        self.vis2.eff_wave = np.array([])  # effective wavelength array
                        self.vis2.eff_wave = np.tile(eff_wave_array, num_baselines)
                    else:
                        self.vis2.eff_wave = np.append(self.vis2.eff_wave, np.tile(eff_wave_array, num_baselines))
                    if self.vis2.eff_band is None:
                        self.vis2.eff_band = np.array([])  # effective spectral bandwidth array
                        self.vis2.eff_band = np.tile(eff_band_array, num_baselines)
                    else:
                        self.vis2.eff_band = np.append(self.vis2.eff_band, np.tile(eff_band_array, num_baselines))

                    # calculate spatial frequencies and other variables
                    self.vis2.ufcoord = self.vis2.ucoord / self.vis2.eff_wave
                    self.vis2.vfcoord = self.vis2.vcoord / self.vis2.eff_wave
                    self.vis2.base = np.sqrt(self.vis2.ucoord**2 + self.vis2.vcoord)
                    self.vis2.spat_freq = np.sqrt(self.vis2.ufcoord**2 + self.vis2.vfcoord)
                    #

                # read OIT3 data
                if hdu.name == "OI_T3":
                    if self.t3 is None:  # if OIVis object not created yet, create it and assign header
                        self.t3 = OIT3()  # initialize empty OIVis object
                        # assign header
                        t3_hdr_keywords = list(hdu.header.keys())
                        t3_hdr_values = list(hdu.header.values())
                        self.t3.header = dict(zip(t3_hdr_keywords, t3_hdr_values))  # assign prim header

                    # retrieve wavelength array of the data
                    eff_wave_array = eff_wave_dict[hdu.header["INSNAME"]]
                    eff_band_array = eff_band_dict[hdu.header["INSNAME"]]

                    # number of wavelength channels
                    num_wave = len(eff_wave_array)

                    # assign table data
                    if self.t3.time is None:  # check if the property already exists or needs to be created first
                        self.t3.time = np.repeat(hdu.data["TIME"], num_wave)
                    else:
                        self.t3.time = np.append(self.t3.time, np.repeat(hdu.data["TIME"], num_wave))
                    if self.t3.mjd is None:
                        self.t3.mjd = np.repeat(hdu.data["MJD"], num_wave)
                    else:
                        self.t3.mjd = np.append(self.t3.mjd, np.repeat(hdu.data["MJD"], num_wave))
                    if self.t3.int_time is None:
                        self.t3.int_time = np.repeat(hdu.data["INT_TIME"], num_wave)
                    else:
                        self.t3.int_time = np.append(self.t3.int_time, np.repeat(hdu.data["INT_TIME"], num_wave))
                    if self.t3.t3amp is None:
                        self.t3.t3amp = hdu.data["T3AMP"].flatten()
                    else:
                        self.t3.t3amp = np.append(self.t3.t3amp, hdu.data["T3AMP"].flatten())
                    if self.t3.t3amperr is None:
                        self.t3.t3amperr = hdu.data["T3AMPERR"].flatten()
                    else:
                        self.t3.t3amperr = np.append(self.t3.t3amperr, hdu.data["T3AMPERR"].flatten())
                    if self.t3.t3phi is None:
                        self.t3.t3phi = hdu.data["T3PHI"].flatten()
                    else:
                        self.t3.t3phi = np.append(self.t3.t3phi, hdu.data["T3PHI"].flatten())
                    if self.t3.t3phierr is None:
                        self.t3.t3phierr = hdu.data["T3PHIERR"].flatten()
                    else:
                        self.t3.t3phierr = np.append(self.t3.t3phierr, hdu.data["T3PHIERR"].flatten())
                    if self.t3.u1coord is None:
                        self.t3.u1coord = np.repeat(hdu.data["U1COORD"], num_wave)
                    else:
                        self.t3.u1coord = np.append(self.t3.u1coord, np.repeat(hdu.data["U1COORD"], num_wave))
                    if self.t3.v1coord is None:
                        self.t3.v1coord = np.repeat(hdu.data["V1COORD"], num_wave)
                    else:
                        self.t3.v1coord = np.append(self.t3.v1coord, np.repeat(hdu.data["V1COORD"], num_wave))
                    if self.t3.u2coord is None:
                        self.t3.u2coord = np.repeat(hdu.data["U2COORD"], num_wave)
                    else:
                        self.t3.u2coord = np.append(self.t3.u2coord, np.repeat(hdu.data["U2COORD"], num_wave))
                    if self.t3.v2coord is None:
                        self.t3.v2coord = np.repeat(hdu.data["V2COORD"], num_wave)
                    else:
                        self.t3.v2coord = np.append(self.t3.v2coord, np.repeat(hdu.data["V2COORD"], num_wave))
                    if self.t3.flag is None:
                        self.t3.flag = hdu.data["FLAG"].flatten()
                    else:
                        self.t3.flag = np.append(self.t3.flag, hdu.data["FLAG"].flatten())

                    # assign added properties
                    # retrieve baseline info
                    if self.t3.baseline_dict is None:
                        self.t3.baseline_dict = {}  # baseline dictionary
                        self.t3.baseline_idx = np.array([])  # baseline index array
                        baseline_indx = 0  # baseline index, initialized to zero
                    else:
                        baseline_indx = int(np.max(self.t3.baseline_idx)) + 1  # if already exists, start new index

                    num_baselines = 0  # number of considered baselines
                    sta_idx = hdu.data["STA_INDEX"]
                    for sta_indx_triple in sta_idx:
                        sta1 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_triple[0]]
                        sta2 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_triple[1]]
                        sta3 = sta_name_dict[hdu.header["ARRNAME"]][sta_indx_triple[2]]
                        sta_string = sta1 + "-" + sta2 + "-" + sta3
                        self.t3.baseline_dict[baseline_indx] = sta_string  # extend baseline dict
                        print(baseline_indx)

                        self.t3.baseline_idx = np.append(
                            self.t3.baseline_idx, np.repeat(baseline_indx, num_wave)
                        ).astype(int)  # extend baseline index array

                        baseline_indx += 1  # increment baseline index
                        num_baselines += 1  # increment number of baseline triangles

                    # caculate third baseline
                    self.t3.u3coord = self.t3.u1coord + self.t3.u2coord
                    self.t3.v3coord = self.t3.v1coord + self.t3.v2coord

                    # retrieve wavelength info
                    if self.t3.eff_wave is None:
                        self.t3.eff_wave = np.array([])  # effective wavelength array
                        self.t3.eff_wave = np.tile(eff_wave_array, num_baselines)
                    else:
                        self.t3.eff_wave = np.append(self.t3.eff_wave, np.tile(eff_wave_array, num_baselines))
                    if self.t3.eff_band is None:
                        self.t3.eff_band = np.array([])  # effective spectral bandwidth array
                        self.t3.eff_band = np.tile(eff_band_array, num_baselines)
                    else:
                        self.t3.eff_band = np.append(self.t3.eff_band, np.tile(eff_band_array, num_baselines))

                    # calculate spatial frequencies
                    self.t3.u1fcoord = self.t3.u1coord / self.t3.eff_wave
                    self.t3.v1fcoord = self.t3.v1coord / self.t3.eff_wave
                    self.t3.u2fcoord = self.t3.u2coord / self.t3.eff_wave
                    self.t3.v2fcoord = self.t3.v2coord / self.t3.eff_wave
                    self.t3.u3fcoord = self.t3.u3coord / self.t3.eff_wave
                    self.t3.v3fcoord = self.t3.v3coord / self.t3.eff_wave
                    #

                    # info for maximum baseline in the triangle
                    self.t3.base_max = np.maximum(
                        np.sqrt(self.t3.u3coord**2 + self.t3.v3coord**2),
                        np.maximum(
                            np.sqrt(self.t3.u1coord**2 + self.t3.v1coord**2),
                            np.sqrt(self.t3.u2coord**2 + self.t3.v2coord**2),
                        ),
                    )
                    self.t3.spat_freq_max = self.t3.base_max / self.t3.eff_wave

                # read OIFlux data
                if hdu.name == "OI_FLUX":
                    if self.flux is None:  # if OIVis object not created yet, create it and assign header
                        self.flux = OIFlux()  # initialize empty OIVis object
                        # assign header
                        flux_hdr_keywords = list(hdu.header.keys())
                        flux_hdr_values = list(hdu.header.values())
                        self.flux.header = dict(zip(flux_hdr_keywords, flux_hdr_values))  # assign primary header

                    # retrieve wavelength array of the data
                    eff_wave_array = eff_wave_dict[hdu.header["INSNAME"]]
                    eff_band_array = eff_band_dict[hdu.header["INSNAME"]]

                    # number of wavelength channels
                    num_wave = len(eff_wave_array)

                    # Retrieve flux data column name (GRAVITY does not follow the OIFITS 2 convention of using
                    # 'FLUXDATA', and instead just use 'FLUX')
                    if "FLUXDATA" in hdu.columns.names:
                        flux_data_col_name = "FLUXDATA"
                    else:
                        flux_data_col_name = "FLUX"  # case for GRAVITY data files

                    # assign table data
                    if self.flux.mjd is None:
                        self.flux.mjd = np.repeat(hdu.data["MJD"], num_wave)
                    else:
                        self.flux.mjd = np.append(self.flux.mjd, np.repeat(hdu.data["MJD"], num_wave))
                    if self.flux.int_time is None:
                        self.flux.int_time = np.repeat(hdu.data["INT_TIME"], num_wave)
                    else:
                        self.flux.int_time = np.append(self.flux.int_time, np.repeat(hdu.data["INT_TIME"], num_wave))
                    if self.flux.fluxdata is None:
                        self.flux.fluxdata = hdu.data[flux_data_col_name].flatten()
                    else:
                        self.flux.fluxdata = np.append(self.flux.fluxdata, hdu.data[flux_data_col_name].flatten())
                    if self.flux.fluxerr is None:
                        self.flux.fluxerr = hdu.data["FLUXERR"].flatten()
                    else:
                        self.flux.fluxerr = np.append(self.flux.fluxerr, hdu.data["FLUXERR"].flatten())
                    if self.flux.flag is None:
                        self.flux.flag = hdu.data["FLAG"].flatten()
                    else:
                        self.flux.flag = np.append(self.flux.flag, hdu.data["FLAG"].flatten())

                    # assign added properties
                    # retrieve telescope info (denotes between 'CALSTAT = U/C' in the header)
                    if hdu.header["CALSTAT"] == "C":  # Calibrated spectrum case
                        self.flux.telescope_dict = None  # Set telescope dict and index array to None
                        self.flux.telescope_idx = None

                        # retrieve wavelength info
                        if self.flux.eff_wave is None:
                            self.flux.eff_wave = np.array([])  # effective wavelength array
                            self.flux.eff_wave = eff_wave_array
                        else:
                            self.flux.eff_wave = np.append(self.flux.eff_wave, eff_wave_array)
                        if self.flux.eff_band is None:
                            self.flux.eff_band = np.array([])  # effective spectral bandwidth array
                            self.flux.eff_band = eff_band_array
                        else:
                            self.flux.eff_band = np.append(self.flux.eff_band, eff_band_array)
                    elif hdu.header["CALSTAT"] == "U":  # Case for uncalibrated spectra (i.e. provided per telescope)
                        if self.flux.telescope_dict is None:
                            self.flux.telescope_dict = {}  # baseline dictionary
                            self.flux.telescope_idx = np.array([])  # baseline index array
                            telescope_indx = 0  # baseline index, initialized to zero
                        else:
                            telescope_indx = (
                                int(np.max(self.flux.telescope_idx)) + 1
                            )  # if already exists, start new index

                        num_telescopes = 0  # number of considered telescopes
                        sta_idx = hdu.data["STA_INDEX"]
                        for sta in sta_idx:
                            sta_string = tel_name_dict[hdu.header["ARRNAME"]][sta]
                            self.flux.telescope_dict[telescope_indx] = sta_string  # extend baseline dict
                            self.flux.telescope_idx = np.append(
                                self.flux.telescope_idx, np.repeat(telescope_indx, num_wave)
                            ).astype(int)  # extend baseline index array

                            telescope_indx += 1  # increment telescope index
                            num_telescopes += 1  # increment number of considered telescopes

                        # retrieve wavelength info
                        if self.flux.eff_wave is None:
                            self.flux.eff_wave = np.array([])  # effective wavelength array
                            self.flux.eff_wave = np.tile(eff_wave_array, num_telescopes)
                        else:
                            self.flux.eff_wave = np.append(self.flux.eff_wave, np.tile(eff_wave_array, num_telescopes))
                        if self.flux.eff_band is None:
                            self.flux.eff_band = np.array([])  # effective spectral bandwidth array
                            self.flux.eff_band = np.tile(eff_band_array, num_telescopes)
                        else:
                            self.flux.eff_band = np.append(self.flux.eff_band, np.tile(eff_band_array, num_telescopes))

            ###

            hdul.close()  # close file

            return

    # TODO: fully finnish this function
    def plot_data(
        self,
        dname: str,
        x: str,
        y: str,
        error: str | None = None,
        cname: str | None = None,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
        per_baseline: bool = False,
        ptype: str = "line",
        show_plots: bool = True,
        fig_dir: str | None = None,
        logx: bool = False,
        logy: bool = False,
        logc: bool = False,
    ) -> plt.Figure:
        """
        Make a plot of the data contained in the object.

        :param str dname: Name of the type of OI data table to plot. Can be `vis`, `vis2`, `t3` and `flux`.
        :param str x: Name of the data to be plotted on the x axis. Can be any of the names of the np.ndarray
            object attributes corresponding to dtype, except `flag`.
        :param str y: Name of the data to be plotted on the y axis. Can be any of the names of the np.ndarray
            object attributes corresponding to dtype, except `flag`.
        :param str error: Name of the data to be used to plot errorbars. None by default -> i.e. no errorbars.
        :param str cname: Name of the data used to colour the plot. Can be any of the names of the np.ndarray
            object attributes corresponding to dtype, except `flag`, but also `baseline` in order to colour data
            by the baseline pair/triangle/telescope. None by default, in which case all datapoints will have the
            same color.
        :param bool per_baseline: Whether to make a separate plot per baseline pair/triangle/telescope. By default
            set to False, meaning all data is put into a separate plot.
        :param str ptype: Plotting type, by default set to 'line' to plot a line for each baseline pair/triangle
            /telescope. Can be set to 'scatter' to instead plot point separately in a scatterplot.
        :param bool show_plots: Whether to show the plots.
        :param str fig_dir: Directory of where to save the figure. None by default, in which case no figure will be
            saved.
        :param bool logx: Whether to plot x axis in log scale.
        :param bool logy: Whether to plot y axis in log scale.
        :param bool logc: Whether to plot the color scale in log scale.
        :return fig: The matplotlib figure, which can then be further manipulated.
        :rtype: plt.Figure
        """

        # TODO: finnish this dictionary
        # dictionary of plotting label strings for the data quantities
        label_dict = {
            "time": r"Time $\,(s)$",
            "mjd": "MJD",
            "int_time": r"Integration time $\, (s)$",
            "visamp": r"$V$",
            "visphi": r"$\phi \, (^\circ)$",
            "eff_wave": r"$\lambda \, (\mathrm{m})$",
            "base": r"$B \, (\mathrm{m})$",
            "spat_freq": r"Spatial frequency $(\, \mathrm{cycle \, rad^{-1}})$",
            "t3phi": r"$\phi_{clos} \, (^\circ)$",
            "base_max": r"$B_{max} \, (\mathrm{m})$",
            "spat_freq_max": r"Maximum spatial frequency $(\, \mathrm{cycle \, rad^{-1}})$",
        }

        data = getattr(self, dname)  # get main data attribute, e.g. vis, vis2, t3 or flux
        xdata = getattr(data, x)  # get x axis data
        ydata = getattr(data, y)  # get y axis data
        if error is not None:
            error_data = getattr(data, error)  # get error
        else:
            error_data = np.zeros_like(xdata)  # set to 0s otherwise
        if cname is not None:
            cdata = getattr(data, cname)  # get colour name
        else:
            cdata = None

        if per_baseline:
            # get number of baseline pairs/telescopes/triangles, which will be the number of plots
            # TODO: make this work by relying on the dictionary and the index array
            if dname == "flux":
                if data.telescope_idx is not None:
                    num_plots = np.size(np.unique(data.telescope_idx))
                else:
                    num_plots = 1  # in case of a calibrateds flux spectrum we can just use a single plot
            else:
                num_plots = np.size(np.unique(data.baseline_idx))
            num_plot_rows = np.ceil(num_plots // 2)

            # create the figure
            fig, ax = plt.subplot(num_plot_rows, 2, figsize=(12, 4 * num_plot_rows))

            if num_plot_rows % 2 == 1:  # remove last plot if needed
                ax[:-1, :-1].remove()

            # for idx in np.unique

        else:
            # create the figure
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            if ptype == "scatter":
                plot = ax.scatter(xdata, ydata, c=cdata, s=2, cmap=constants.PLOT_CMAP)  # make scatter plot
                ax.errorbar(
                    xdata,
                    ydata,
                    error_data,
                    ecolor="grey",
                    fmt="",
                    zorder=-1000,
                    capsize=0,
                    ls="",
                    alpha=0.5,
                    elinewidth=0.5,
                )  # plot errorbars

                # set colorbar
                if cdata is not None:
                    cb = fig.colorbar(plot, ax=ax, pad=0.01, aspect=40, label=label_dict[cname])

                # set axis limits
                if xlim is not None:
                    ax.set_xlim(xlim[0], xlim[1])
                # set axis limits
                if ylim is not None:
                    ax.set_ylim(ylim[0], ylim[1])

                # set axis labels
                ax.set_xlabel(label_dict[x])
                ax.set_ylabel(label_dict[y])

        plt.tight_layout()

        if fig_dir is not None:
            plt.savefig(
                os.path.join(fig_dir, f"closure_phases.{constants.FIG_OUTPUT_TYPE}"),
                dpi=constants.FIG_DPI,
                bbox_inches="tight",
            )
        if show_plots:
            plt.show()

        return

    def plot_uv(self, dname):
        """
        Make a plot of the uv coverage for the data.

        :param str dname: Name of the type of OI data table whose uv coverage is plotted.
            Can be `vis`, `vis2` or `t3`.
        """
        pass

    # TODO: finnish this filtering function
    def filter(self, name: str | list[str], range: tuple[float, float] | list[tuple[float, float]]) -> None:
        """
        Filter data based on user input range on quantity of interest or on if data has been flagged.

        NOTE: This function applies to all relevant types of stored data. I.e. if the OIData object contains both
        squared visibilities and tiple products, and you filter based on 'eff_wave', filtering will be applied to
        both types of observables. If you want to filter differently for different data types, e.g. on wavelength of
        the triple product only, call the filter function of the OIT3 class directly instead.

        :param str | list[str] names: Name(s) of the variables to be filtered (e.g. 'visamp'). Can also pass 'flag' in
            order to filter out flagged data.
        :param tuple[float, float] | list[tuple[float, float]]: Range(s) of the variables which we want to cut.
            E.g. name='visamp' & range=(0.01, 0.1) -> will filter out all data with visibility between 0.01 and 0.1.
        :rtype: None
        """
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
    :ivar float ra_err: Error on right ascension.
    :ivar float dec_err: Error on declination.
    :ivar float sysvel: Systemic radial velocity in m/s.
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

    NOTE: The correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).

    :ivar dict header: Header dictionary.
    :ivar np.ndarray time: Time of observations in seconds.
    :ivar np.ndarray mjd: Modified Julian Date of observations.
    :ivar np.ndarray int_time: Integration time in seconds.
    :ivar np.ndarray visamp: Visibility amplitude. Units depend on type of visibility (e.g. coherent flux vs.
        normalized visibility). In case 'correlated flux' and TUNIT keyword not specified in header -> assumed to be
        in Jansky.
    :ivar np.ndarray visamperr: Error on visibility amplitude.
    :ivar np.ndarray visphi: Visibility phase in degrees.
    :ivar np.ndarray visphierr: Error on visibility phase.
    :ivar np.ndarray rvis: Real part of complex coherent flux; units defined in header
    :ivar np.ndarray rviserr: Error on real part of complex coherent flux.
    :ivar np.ndarray ivis: Imaginary part of complex coherent flux; units defined in header
    :ivar np.ndarray iviserr: Error on imaginary part of complex coherent flux.
    :ivar np.ndarray ucoord: u coordinate of the data in unit m.
    :ivar np.ndarray vcoord: v coordinate of the data in unit m.
    :ivar np.ndarray flag: Array of booleans denoting whether data has been flagged by the reduction or not.
    :ivar dict baseline_dict: Dictionary mapping baseline index integers to the strings describing the baseline.
    :ivar np.ndarray baseline_idx: Baseline index of the data (to be used with baseline_dict).
    :ivar np.ndarray eff_wave: Effective wavelength of the data; units m.
    :ivar np.ndarray eff_band: Effective wavelength bandwidth of data; units m.
    :ivar np.ndarray ufcoord: u coordinate spatial frequency; units cycle rad^-1.
    :ivar np.ndarray vfcoord: v coordinate spatial frequency; units cycle rad^-1.
    :ivar np.ndarray base: Size of the baseline in m.
    :ivar np.ndarray spat_freq: Spatial frequency in cycles per radian.
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

        ### Added properties for ease of use (e.g. filtering, plotting and caclulations)
        self.baseline_dict = None  # dictionary from baseline index to string describing the baseline pair
        # properties with dimension equal to amount of considered data points
        self.baseline_idx = None  # np.ndarray with indices denoting to which basline pair a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.base = None  # euclidian sum of the u and v coord, i.e. the size of the baseline
        self.spat_freq = None  # spatial frequency of the data in cycles per rad
        ###


class OIVis2:
    """
    Class to describe the concatenated data contained in OI_VIS2 tables of multiple OIFITS files.

    NOTE: The correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).


    :ivar dict header: Header dictionary.
    :ivar np.ndarray time: Time of observations in seconds.
    :ivar np.ndarray mjd: Modified Julian Date of observations.
    :ivar np.ndarray int_time: Integration time in seconds.
    :ivar np.ndarray vis2data: Squared visibility amplitude. Always normalized.
    :ivar np.ndarray vis2err: Error on squared visibility amplitude.
    :ivar np.ndarray ucoord: u coordinate of the data in unit m.
    :ivar np.ndarray vcoord: v coordinate of the data in unit m.
    :ivar np.ndarray flag: Array of booleans denoting whether data has been flagged by the reduction or not.
    :ivar dict baseline_dict: Dictionary mapping baseline index integers to the strings describing the baseline.
    :ivar np.ndarray baseline_idx: Baseline index of the data (to be used with baseline_dict).
    :ivar np.ndarray eff_wave: Effective wavelength of the data; units m.
    :ivar np.ndarray eff_band: Effective wavelength bandwidth of data; units m.
    :ivar np.ndarray ufcoord: u coordinate spatial frequency; units cycle rad^-1.
    :ivar np.ndarray vfcoord: v coordinate spatial frequency; units cycle rad^-1.
    :ivar np.ndarray base: Size of the baseline in m.
    :ivar np.ndarray spat_freq: Spatial frequency in cycles per radian.
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
        self.vis2err = None  # np.ndarray with squared visibility error
        self.ucoord = None  # np.ndarray with u coordinate of the data in m
        self.vcoord = None  # np.ndarray with v coordinate of the data in m
        self.flag = None  # boolean np.ndarray with flag status of the data
        ###

        ### Added properties for ease of use (e.g. filtering, plotting and caclulations)
        self.baseline_dict = None  # dictionary from baseline index to string describing the baseline pair
        # properties with dimension equal to amount of considered data points
        self.baseline_idx = None  # np.ndarray with indices denoting to which basline pair a datapoint belongs to
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        self.ufcoord = None  # u coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.vfcoord = None  # v coordinate in frequency space, units of cycle per radian; for ease of calculations
        self.base = None  # euclidian sum of the u and v coord, i.e. the size of the baseline
        self.spat_freq = None  # spatial frequency of the data in cycles per rad
        ###


class OIT3:
    """
    Class to describe the concatenated data contained in OI_T3 tables of multiple OIFITS files.

    NOTE: The correspondence to the OIFITS standard. For example, all triple product data is flattened into 1D arrays
    for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote the baseline
    triangles. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx array and the
    baseline_dict dictionary. In addition, the wavelength information is stored in the 1D eff_wave array.

    NOTE: The third baseline is chosen following the convention that the triple product is calculated as V1 x V2 x V3*,
    with V3* being the conjugate of the third baseline's visibility.

    :ivar dict header: Header dictionary.
    :ivar np.ndarray time: Time of observations in seconds.
    :ivar np.ndarray mjd: Modified Julian Date of observations.
    :ivar np.ndarray int_time: Integration time in seconds.
    :ivar np.ndarray t3amp: Triple product amplitude.
    :ivar np.ndarray t3amperr: Error on tiple product amplitude.
    :ivar np.ndarray t3phi: Triple product phase (closure phase) in degrees.
    :ivar np.ndarray t3phierr: Error on tiple product phase (closure phase).
    :ivar np.ndarray u1coord: u coordinate of the data's first baseline in unit m.
    :ivar np.ndarray v1coord: v coordinate of the data's first baseline in unit m.
    :ivar np.ndarray u2coord: u coordinate of the data's second baseline in unit m.
    :ivar np.ndarray v2coord: v coordinate of the data's second baseline in unit m.
    :ivar np.ndarray flag: Array of booleans denoting whether data has been flagged by the reduction or not.
    :ivar dict baseline_dict: Dictionary mapping baseline index integers to the strings describing the baseline.
    :ivar np.ndarray baseline_idx: Baseline index of the data (to be used with baseline_dict).
    :ivar np.ndarray eff_wave: Effective wavelength of the data; units m.
    :ivar np.ndarray eff_band: Effective wavelength bandwidth of data; units m.
    :ivar np.ndarray u3coord: u coordinate of the data's third baseline in unit m.
    :ivar np.ndarray v3coord: v coordinate of the data's third baseline in unit m.
    :ivar np.ndarray u1fcoord: u coordinate spatial frequency of first baseline; units cycle rad^-1.
    :ivar np.ndarray v1fcoord: v coordinate spatial frequency of first baseline; units cycle rad^-1.
    :ivar np.ndarray u2fcoord: u coordinate spatial frequency of second baseline; units cycle rad^-1.
    :ivar np.ndarray v2fcoord: v coordinate spatial frequency of second baseline; units cycle rad^-1.
    :ivar np.ndarray u3fcoord: u coordinate spatial frequency of third baseline; units cycle rad^-1.
    :ivar np.ndarray v3fcoord: v coordinate spatial frequency of third baseline; units cycle rad^-1.
    :ivar np.ndarray base_max: Size of the longest baseline in m.
    :ivar np.ndarray spat_freq_max: Spatial frequency in of the longest baseline cycles per radian.
    """

    def __init__(self):
        """
        Constructor method. See class docstring for information on initialization parameters and instance properties.

        NOTE: The correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
        into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows
        denote the baselines. Instead, the baseline pair corresponding to a datapoint is encoded in the baseline_idx
        array and the baseline_dict dictionary, and, in addition, the wavelength information is stored in the 1D
        eff_wave array (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).
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

        ### Added properties for ease of use (e.g. filtering, plotting and caclulations)
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
        self.base_max = None  # Size of the longest baseline in the triangle in m.
        self.spat_freq_max = None  # Spatial frequency of the longest baseline in the triangle.
        ###


class OIFlux:
    """
    Class to describe the concatenated data contained in OI_FLUX tables of multiple OIFITS files.

    NOTE: The correspondence to the OIFITS standard is not exact. For example, all visibility data is flattened
    into 1D arrays for ease of use, instead of keeping the OIFITS 2D array structures, where the different rows denote
    the telescopes. Instead, the baseline pair corresponding to a datapoint is encoded in the telescope_idx array and
    the telescope_dict dictionary, and, in addition, the wavelength information is stored in the 1D eff_wave array
    (replacing the usual link with the OI_ARRAY and OI_WAVELENGTH tables).

    :ivar dict header: Header dictionary.
    :ivar np.ndarray time: Time of observations in seconds.
    :ivar np.ndarray mjd: Modified Julian Date of observations.
    :ivar np.ndarray int_time: Integration time in seconds.
    :ivar np.ndarray fluxdata: Measured flux; units specified in header.
    :ivar np.ndarray fluxerr: Data on measured flux.
    :ivar np.ndarray flag: Array of booleans denoting whether data has been flagged by the reduction or not.
    :ivar dict telescope_dict: Dictionary mapping index integers to the strings describing the telescope used for the
        measurement. Only used if 'CALSTAT = U', in which the uncalibrated flux spectrum is stored per telescope
        (see OIFITS 2 standard). If 'CALSTAT = C', will instead remain None.
    :ivar np.ndarray telescope_idx: Telescope index of the data (to be used with telescope_dict). Only used if
        'CALSTAT = U', in which case the uncalibrated flux spectrum is stored per telescope (see OIFITS 2 standard).
        If 'CALSTAT = C', will instead remain None.
    :ivar np.ndarray eff_wave: Effective wavelength of the data; units m.
    :ivar np.ndarray eff_band: Effective wavelength bandwidth of data; units m.
    """

    # TODO: account for the calibrated/uncalibrated cases described in the OIFITS V2 paper

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

        ### Added properties for ease of use (e.g. filtering, plotting and caclulations)
        self.telescope_dict = (
            None  # dictionary from baseline index to string describing the used telescope. Only used if 'CALSTAT = U'
        )
        # properties with dimension equal to amount of considered data points.
        self.telescope_idx = None  # np.ndarray with indices denoting to which telescope a datapoint belongs to.
        # Only used if 'CALSTAT = U'.
        self.eff_wave = None  # np.ndarray with effective  wavelengths in m; replaces link with OI_WAVELENGTH tables
        self.eff_band = None  # np.ndarray with effective wavelength band widths in m
        ###


if __name__ == "__main__":
    import numpy as np
    from astropy.io import fits
    import matplotlib.pyplot as plt

    data_dir = (
        "/home/toond/Documents/phd/data/IRAS08544-4431/" "radiative_transfer_modelling_corporaal_et_al2023/MATISSE_L"
    )
    data_file = "2019-04-29T012641_IRAS08544-4431_K0G2D0J3_IR-LM_LOW_cal_oifits_0.fits"

    oidata = OIData()
    oidata.read_oifits(data_dir=data_dir, data_file=data_file)
    oidata.plot_data(
        dname="vis",
        x="spat_freq",
        y="visphi",
        error="visphierr",
        cname="eff_wave",
        ptype="scatter",
    )
    # print((oidata.vis.baseline_dict))
    # print((oidata.t3.baseline_dict))
    # print((oidata.flux.telescope_dict))
    # print((oidata.flux.telescope_idx))

    # flux = oidata.flux.fluxdata
    # wave = oidata.flux.eff_wave

    # indices = np.argwhere(oidata.flux.telescope_idx == 0)
    # flux_t1 = oidata.flux.fluxdata[indices]
    # wave_t1 = oidata.flux.eff_wave[indices]

    # ptype = "line"

    # if ptype == "scatter":
    #     plt.scatter(wave, flux)
    # if ptype == "line":
    #     for indx in np.unique(oidata.flux.telescope_idx):
    #         indices = np.argwhere(oidata.flux.telescope_idx == indx)
    #         flux_tel = oidata.flux.fluxdata[indices]
    #         wave_tel = oidata.flux.eff_wave[indices]
    #         plt.plot(wave_tel, flux_tel)
    # plt.show()
