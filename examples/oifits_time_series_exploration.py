"""
Testcase showing how to create an explorative plot of a time-series of OIFITS files. This plot can be used to explore
the uv coverage of observations within a given time window, which can be moved and changed in size using the provided
sliders. Uses VLTI/PIONIER observational data of the IRAS08544-4431 system.

Data taken from Corporaal et al. 2023 (A&A 671, A15: https://doi.org/10.1051/0004-6361/202245689).
"""

import os
import distroi
import distroi.auxiliary

import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s: %(message)s", stream=sys.stdout)


# set properties for run
# data_dir = "./data/IW_Car/PIONIER/"
data_dir = os.path.expandvars("$MATISSE_DR/reduced/HR4049/non_chopped_selected_filtered/")
copy_dir = os.path.expandvars("$MATISSE_DR/reduced/HR4049/non_chopped_selected_filtered_time_windowed/")
#data_dir = os.path.expandvars("$PHD/umon_modelling/data/PIONIER_ALL/")
#copy_dir = os.path.expandvars("$DOWNLOADS/data_test/")
data_file = "*.fits"
init_time_window = 80 # initial time window width in days

# alternative call where I copy the resulting OIFITS files withim the plot's time window to a folder in my downloads
filenames = distroi.auxiliary.oifits_time_window_plot(data_dir, data_file, init_time_window, copy_dir=copy_dir)
