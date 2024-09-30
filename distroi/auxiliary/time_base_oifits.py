"""
Provides function to interactively plot multiple OIFITS files as a time-series, allowing the user to specify and adapt
a certain time window. In other words, it allows the user to investigate the timebase of the observations.
Functionalities are provided to plot the uv coverage of data within the time window, and retrieve
the paths of OIFITS files with observations within the time window (and optionally copy them to a specified directory).

NOTE: Because of the spawning of interactive plots, this functionalities in this module cannot be properly used
in jupyter notebooks, instead use them within a python script.
"""

import os
import shutil

import numpy as np
import pandas as pd
import glob

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import astropy.io.fits as fits

from distroi.auxiliary import constants

constants.set_matplotlib_params()  # set project matplotlib parameters


def timestamp_to_plt_float(date_time: pd.Timestamp) -> float:
    """
    Pass along a pandas Timestamp object and convert it to the amount of days passed since
    midnight 1970-01-01 UTC as a float. This is also how numpy converts Timestamp instances
    to floats for plotting purposes.

    :param pd.Timestamp date_time: Pandas Timestamp instance.
    :return date_plt_float: The Timestamp converted to a float value to be used in matplotlib when specifying e.g.
        position along the axes.
    :rtype: float
    """
    date_ref = pd.to_datetime(0, unit="D")  # reference date at midnight 1970-01-01 UTC
    time_dif = date_time - date_ref  # time difference in nanoseconds
    date_plt_float = time_dif.value / 8.64e13  # convert .value (nanoseconds) to days

    return date_plt_float


def oifits_time_window_plot(
    data_dir: str, data_file: str, init_window_width: float, copy_dir: str = None
) -> list[str] | None:
    """
    Produces an interactive matplotlib plot showing the specified OIFITS file as a time-series.
    The specified time window is shown as a red-shaded area. Sliders are provided to change the position and width
    of this time window. The 'uv coverage' button can then be clicked in order to plot the uv coverage of the data
    included within the time window. If the copy_dir argument is passed along, the 'Copy files' button can be
    pressed to copy over the OIFITS file which contain observations within the interactively set time window.
    After all plots are closed, returns a list with the filepaths of these OIFITS files.

    NOTE: Because the uv coverage button spawns another plot,
    this function cannot be properly used in jupyter notebooks, instead use this method within a python script.

    :param str data_dir: Path to the directory where the OIFITS files are stored.
    :param str data_file: Data filename. Use wildcards to read in multiple files at once.
    :param float init_window_width: Initial width of the time window in days. Needs to be larger or equal to 1 day or
        function will not plot and return None instead.
    :param str copy_dir: If specified, the OIFITS files that have observations within the interactively set time window
        will be copied to this directory upon using the 'Copy files' button.
    :return files_within_window: List of filepaths corresponding to the OIFITS files that have observations within the
        interactively set time window.
    :rtype: list[str]
    """

    # create copy directory if it doesn't exist yet
    if copy_dir is not None:
        if not os.path.exists(copy_dir):
            os.makedirs(copy_dir)

    if init_window_width < 1:
        print("Initial window width must be larger than or equal to 1 day! Function will return None!")
        return

    obs_mjd = []  # list to hold MJD dates extracted from OIFITS files
    file_names = []  # list to assign filenames to the observations
    wavelengths = []  # list to assign wavelengths in meter
    u_coordinates = []  # list to assign u-direction baseline lengths in meter
    v_coordinates = []  # list to assign v-direction baseline lengths in meter

    # retrieve date and uv coverage information from the OIFITS files
    filepaths = sorted(glob.glob(f"{data_dir}/{data_file}", recursive=True))  # get filenames
    for filepath in filepaths:
        hdul = fits.open(filepath)  # open oifits file
        insname_to_eff_wave_dict = {}  # dictionary to associate the OIFITS INSNAME keywoard to wavelength values

        for hdu in hdul:
            if hdu.name == "OI_WAVELENGTH":
                insname = hdu.header["INSNAME"]  # INSNAME keyword
                eff_wave_col_index = hdu.columns.names.index("EFF_WAVE")  # column index for effective wavelength
                hdu_wavelengths = []  # list to keep wavelength values in
                for row in hdu.data:
                    hdu_wavelengths.append(row[eff_wave_col_index])
                insname_to_eff_wave_dict[insname] = hdu_wavelengths  # add to dictionary linking INSNAME and wavelengths

            if hdu.name == "OI_VIS2":  # for squared visibility tables
                insname = hdu.header["INSNAME"]  # INSNAME keyword
                mjd_col_index = hdu.columns.names.index("MJD")  # column index for modified julian date (MJD)
                ucoord_col_index = hdu.columns.names.index("UCOORD")  # column index for projected u-baseline (in meter)
                vcoord_col_index = hdu.columns.names.index("VCOORD")  # column index for projected v-baseline (in meter)
                for row in hdu.data:
                    eff_wavelengths = insname_to_eff_wave_dict[insname]

                    wavelengths.extend(eff_wavelengths)  # add wavelengths to the corresponding list
                    file_names.extend([filepath] * len(eff_wavelengths))  # add filename to the list
                    obs_mjd.extend(
                        [row[mjd_col_index]] * len(eff_wavelengths)
                    )  # add rounded off MJD date to the set of dates
                    u_coordinates.extend([row[ucoord_col_index]] * len(eff_wavelengths))  # add uv coordinates
                    v_coordinates.extend([row[vcoord_col_index]] * len(eff_wavelengths))

            if hdu.name == "OI_VIS":  # for visibility tables
                insname = hdu.header["INSNAME"]  # INSNAME keyword
                mjd_col_index = hdu.columns.names.index("MJD")  # column index for modified julian date (MJD)
                ucoord_col_index = hdu.columns.names.index("UCOORD")  # column index for projected u-baseline (in meter)
                vcoord_col_index = hdu.columns.names.index("VCOORD")  # column index for projected v-baseline (in meter)
                for row in hdu.data:
                    eff_wavelengths = insname_to_eff_wave_dict[insname]

                    wavelengths.extend(eff_wavelengths)  # add wavelengths to the corresponding list
                    file_names.extend([filepath] * len(eff_wavelengths))  # add filename to the list
                    obs_mjd.extend(
                        [row[mjd_col_index]] * len(eff_wavelengths)
                    )  # add rounded off MJD date to the set of dates
                    u_coordinates.extend([row[ucoord_col_index]] * len(eff_wavelengths))  # add uv coordinates
                    v_coordinates.extend([row[vcoord_col_index]] * len(eff_wavelengths))

            if hdu.name == "OI_T3":  # for closure phase tables
                insname = hdu.header["INSNAME"]  # INSNAME keyword
                mjd_col_index = hdu.columns.names.index("MJD")  # column index for modified julian date (MJD)
                u1coord_col_index = hdu.columns.names.index(
                    "U1COORD"
                )  # column index for 1st projected u-baseline (in meter)
                v1coord_col_index = hdu.columns.names.index(
                    "V1COORD"
                )  # column index for 1st projected v-baseline (in meter)
                u2coord_col_index = hdu.columns.names.index(
                    "U2COORD"
                )  # column index for 2nd projected u-baseline (in meter)
                v2coord_col_index = hdu.columns.names.index(
                    "V2COORD"
                )  # column index for 2nd projected v-baseline (in meter)
                for row in hdu.data:
                    eff_wavelengths = insname_to_eff_wave_dict[insname]

                    wavelengths.extend(eff_wavelengths)  # add wavelengths to the corresponding list
                    file_names.extend([filepath] * len(eff_wavelengths))  # add filename to the list
                    obs_mjd.extend(
                        [row[mjd_col_index]] * len(eff_wavelengths)
                    )  # add rounded off MJD date to the set of dates
                    u_coordinates.extend([row[u1coord_col_index]] * len(eff_wavelengths))  # add uv coordinates
                    v_coordinates.extend([row[v1coord_col_index]] * len(eff_wavelengths))

                    wavelengths.extend(eff_wavelengths)  # add wavelengths to the corresponding list
                    file_names.extend([filepath] * len(eff_wavelengths))  # add filename to the list
                    obs_mjd.extend([row[mjd_col_index]] * len(eff_wavelengths))
                    u_coordinates.extend([row[u2coord_col_index]] * len(eff_wavelengths))
                    v_coordinates.extend([row[v2coord_col_index]] * len(eff_wavelengths))

                    # also for 3rd baseline in the closure triangle
                    wavelengths.extend(eff_wavelengths)  # add wavelengths to the corresponding list
                    file_names.extend([filepath] * len(eff_wavelengths))  # add filename to the list
                    obs_mjd.extend([row[mjd_col_index]] * len(eff_wavelengths))
                    u_coordinates.extend([row[u1coord_col_index] + row[u2coord_col_index]] * len(eff_wavelengths))
                    v_coordinates.extend([row[v1coord_col_index] + row[v2coord_col_index]] * len(eff_wavelengths))

    # cast to numpy arrays
    wavelengths, u_coordinates, v_coordinates = (
        np.array(wavelengths),
        np.array(u_coordinates),
        np.array(v_coordinates),
    )
    # get probed spatial frequencies in 1/rad
    uf, vf = (u_coordinates / wavelengths), (v_coordinates / wavelengths)

    # (Modified) Julian Date float formats
    obs_mjd = np.array(obs_mjd)  # array of rounded modified julian dates
    obs_timespan = abs(np.max(obs_mjd) - np.min(obs_mjd))  # total time spanned accross all observations in days

    # pandas Timestamp format (for plotting in matplotlib with date labels)
    obs_jd = obs_mjd + 2400000.5  # convert MJD to julian dates first
    date_times = pd.to_datetime(
        np.array(list(set(obs_jd))), origin="julian", unit="D"
    )  # no duplicates +  cast to Timestamp
    date_timespan = np.max(date_times) - np.min(date_times)  # timespan in Timestamp format
    date_init_time_window = pd.to_datetime(init_window_width, unit="D") - pd.to_datetime(
        0, unit="D"
    )  # window in Timestamp
    # format

    # interactive plot with sliders to adapt time window
    # and with plot to show the uv coverage
    fig = plt.figure(figsize=(10, 7))
    main_ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.30, left=0.10)
    main_ax.set_title("Observations across UTC date")

    # plot vertical lines for observation dates
    for i, time in enumerate(date_times):
        if i == 0:  # set axis legend label once
            main_ax.axvline(
                timestamp_to_plt_float(time),
                color="b",
                linestyle="-",
                label="observations",
                alpha=0.3,
                lw=1.2,
            )
        else:
            main_ax.axvline(
                timestamp_to_plt_float(time),
                color="b",
                linestyle="-",
                alpha=0.3,
                lw=1.2,
            )

    # set axis layout
    main_ax.set_ylim(0, 1)
    main_ax.get_yaxis().set_visible(False)
    main_ax.set_xlim(
        np.min(date_times) - 0.1 * date_timespan,
        np.max(date_times) + 0.1 * date_timespan,
    )

    # initial vspan
    vspan = main_ax.axvspan(
        np.min(date_times),
        np.min(date_times) + date_init_time_window,
        alpha=0.3,
        color="r",
        label="time window",
    )

    # add sliders to move the time window across the plot and adapt its width
    # also a button to plot and print info for the observations' uv coverage within the time window
    # and one to copy files to copy_dir if specified
    window_t0_slider_ax = fig.add_axes((0.25, 0.10, 0.65, 0.03))
    window_width_slider_ax = fig.add_axes((0.25, 0.05, 0.65, 0.03))
    uv_coverage_plot_button_ax = fig.add_axes((0.8, 0.15, 0.12, 0.04))

    window_t0_slider = Slider(
        window_t0_slider_ax,
        r"$t_{0,window} - t_{0,data}$ (days)",
        -0.1 * obs_timespan,
        1.1 * obs_timespan,
        valinit=0,
        color="r",
        alpha=0.3,
    )
    window_width_slider = Slider(
        window_width_slider_ax,
        r"$\Delta t_{window}$ (days)",
        1,
        1.1 * obs_timespan,
        valinit=min(1.0 * init_window_width, obs_timespan),
        color="r",
        alpha=0.3,
    )
    uv_coverage_plot_button = Button(uv_coverage_plot_button_ax, "uv coverage", color="white")

    # create initial list containing the files with observations within the specified time window
    files_within_window = []
    for j in range(0, len(obs_mjd)):
        if (
            np.min(obs_mjd) + window_t0_slider.val
            <= obs_mjd[j]
            <= np.min(obs_mjd) + window_t0_slider.val + window_width_slider.val
        ):
            files_within_window.append(file_names[j])
    files_within_window = list(sorted(set(files_within_window)))  # remove duplicates and sort

    def window_t0_slider_on_change(val):  # update function for time window t0 slider
        files_within_window.clear()  # clear the within window filelist
        # select datapoints within timewindow and the associated filenames
        mjd_t0_slider = np.min(obs_mjd) + window_t0_slider.val  # beginning of time window t0 slider in MJD

        file_name_selection = []
        for j in range(0, len(obs_mjd)):  # append values only within time window
            if mjd_t0_slider <= obs_mjd[j] <= (mjd_t0_slider + window_width_slider.val):
                file_name_selection.append(file_names[j])
        files_within_window.extend(sorted(set(file_name_selection)))  # remake the files within window list

        t_begin_span = timestamp_to_plt_float(np.min(date_times)) + val  # set beginning of vspan
        # adapt polygon vertices accordingly
        vspan.set_xy(
            np.array(
                [
                    [t_begin_span, 0],
                    [t_begin_span + window_width_slider.val, 0],
                    [t_begin_span + window_width_slider.val, 1],
                    [t_begin_span, 1],
                ]
            )
        )
        # fig.canvas.draw_idle()  # redraw matplotlib figure
        # main_ax.redraw_in_frame()
        return

    def window_width_slider_on_change(
        val,
    ):  # update function for time window width slider
        files_within_window.clear()  # clear the within window filelist
        # select datapoints within timewindow and the associated filenames
        mjd_t0_slider = np.min(obs_mjd) + window_t0_slider.val  # beginning of time window t0 slider in MJD

        file_name_selection = []
        for j in range(0, len(obs_mjd)):  # append values only within time window
            if mjd_t0_slider <= obs_mjd[j] <= (mjd_t0_slider + window_width_slider.val):
                file_name_selection.append(file_names[j])
        files_within_window.extend(sorted(set(file_name_selection)))  # remake the files within window list
        window_t0_slider_on_change(window_t0_slider.val)

        return

    def uv_coverage_plot_button_on_click(mouse_event):
        files_within_window.clear()  # clear the within window filelist

        # select datapoints within timewindow and the associated filenames
        mjd_t0_slider = np.min(obs_mjd) + window_t0_slider.val  # beginning of time window t0 slider in MJD

        wavelength_selection, uf_selection, vf_selection, file_name_selection = (
            [],
            [],
            [],
            [],
        )
        for j in range(0, len(obs_mjd)):  # append values only within time window
            if mjd_t0_slider <= obs_mjd[j] <= (mjd_t0_slider + window_width_slider.val):
                wavelength_selection.append(wavelengths[j])
                uf_selection.append(uf[j])
                vf_selection.append(vf[j])
                file_name_selection.append(file_names[j])
        # cast to numpy arrays
        wavelength_selection, uf_selection, vf_selection = (
            np.array(wavelength_selection),
            np.array(uf_selection),
            np.array(vf_selection),
        )

        plt.ion()  # enable interactive mode (not necessary but avoids some errors)

        # plot uv coverage
        uv_cov_fig, uv_cov_ax = plt.subplots(1, 1, figsize=(8, 8))  # extra figure to plot uv coverage in
        uv_cov_fig.subplots_adjust(right=0.8)
        uv_cov_cax = uv_cov_fig.add_axes((0.82, 0.15, 0.02, 0.7))
        uv_cov_ax.set_aspect("equal", adjustable="datalim")  # make plot axes have the same scale

        uv_cov_ax.scatter(
            uf_selection / 1e6,
            vf_selection / 1e6,
            c=wavelength_selection * constants.M2MICRON,
            s=1,
            cmap="gist_rainbow_r",
        )
        sc = uv_cov_ax.scatter(
            -uf_selection / 1e6,
            -vf_selection / 1e6,
            c=wavelength_selection * constants.M2MICRON,
            s=1,
            cmap="gist_rainbow_r",
        )
        clb = fig.colorbar(sc, cax=uv_cov_cax)
        clb.set_label(r"$\lambda$ ($\mu$m)", labelpad=5)

        uv_cov_ax.set_xlim(uv_cov_ax.get_xlim()[::-1])  # switch x-axis direction
        uv_cov_ax.set_title("uv coverage within time window")
        uv_cov_ax.set_xlabel(r"$\leftarrow B_u$ ($\mathrm{M \lambda}$)")
        uv_cov_ax.set_ylabel(r"$B_v \rightarrow$ ($\mathrm{M \lambda}$)")

        files_within_window.extend(sorted(set(file_name_selection)))  # remake the files within window list

        plt.show()
        return

    # assign update functions to sliders and button
    window_t0_slider.on_changed(window_t0_slider_on_change)
    window_width_slider.on_changed(window_width_slider_on_change)
    uv_coverage_plot_button.on_clicked(uv_coverage_plot_button_on_click)

    # additional axis and button for copying
    if copy_dir is not None:
        copy_files_button_ax = fig.add_axes((0.68, 0.15, 0.10, 0.04))
        copy_files_button = Button(copy_files_button_ax, "copy files", color="white")

        def copy_files_button_on_click(mouse_event):
            if not os.path.exists(copy_dir):  # make directory if it doesn't exist yet
                os.makedirs(copy_dir)
            for filepath in files_within_window:
                shutil.copy(filepath, copy_dir)  # copy over the files within the time window
            return

        copy_files_button.on_clicked(copy_files_button_on_click)  # assign to button

    # format the date x-axis date labels nicely
    for label in main_ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(30)
    main_ax.legend(loc="upper left")  # set legend

    plt.show()

    return files_within_window


if __name__ == "__main__":
    data_dir = "/home/toond/Documents/phd/data/IRAS15469-5311/inspiring/PIONIER/all_data"
    data_file = "*.fits"
    oifits_time_window_plot(data_dir=data_dir, data_file=data_file, init_window_width=10)

    data_dir = "/home/toond/Documents/phd/writing/proposals/ESO_P115/jmmc_tools"
    data_file = "*.fits"
    oifits_time_window_plot(data_dir=data_dir, data_file=data_file, init_window_width=10)
