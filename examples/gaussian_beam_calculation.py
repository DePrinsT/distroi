"""Testcase showing the calculation of the Gaussian resolution ellipse from the uv coverage of observations. This ellipse
, calculated from a Gaussian fit to the interferometric point-spread-function, represents the smallest element that can
be resolved using the given uv coverage. Uses VLTI/PIONIER observational data of the IRAS08544-4431 system.

Data taken from Corporaal et al. 2023 (A&A 671, A15: https://doi.org/10.1051/0004-6361/202245689).

NOTE: the astrometric precision can actually be much lower than the resolution resulting from this calculation,
but a reconstructed image cannot reliably achieve much greater resolution.
"""

import distroi
from distroi.auxiliary.beam import oi_container_calc_gaussian_beam
import os
import numpy as np
import matplotlib.pyplot as plt

# Read in the data to an OIContainer object
# data_dir, data_file = "./data/IRAS08544-4431/PIONIER/", "*.fits"

## target names
# targets = [
#    "AI_Sco",
#    "EN_TrA",
#    "HD95767",
#    "HD108015",
#    "HR4049",
#    "IRAS15469-5311",
#    "IW_Car",
#    "PS_Gem",
# ]
# for target in targets:
#    data_dir, data_file = (
#        os.path.expandvars(f"$DATA/{target}/PIONIER_FINAL/"),
#        "*.fits",
#    )
#
#    container_data = distroi.read_oi_container_from_oifits(data_dir, data_file)
#
#    # Make plots of the data, including the uv coverage
#    fig_dir = "./figures/gaussian_beam_calculation/"
#    container_data.plot_data(fig_dir=fig_dir)
#
#    # Calculate the resolution ellipse from a Gaussian fit to the interferometric point-spread function.
#    # This method is available in the auxiliary subpackage.
#    beam = oi_container_calc_gaussian_beam(
#        container_data,
#        vistype="vis2",
#        make_plots=True,
#        show_plots=True,
#        fig_dir=fig_dir,
#        num_res=3,
#        pix_per_res=64,
#        save_uv_coverage_dict=True,
#        uv_coverage_dict_path=os.path.expandvars(f"$IMG_REC/{target}/uv_coverage.pkl"),
#    )

data_dir, data_file = (
    os.path.expandvars(f"$MATISSE_PAPER/data/reduced/HR4049/non_chopped_selected_filtered_time_windowed/"),
    "*.fits",
)

#data_dir, data_file = (
#    os.path.expandvars(f"$DATA_ARCHIVE/IRAS08544-4431/radiative_transfer_modelling_corporaal_et_al2023/GRAVITY/"),
#    "*.fits",
#)




#data_dir, data_file = (
#    os.path.expandvars(f"$DATA/IW_Car/PIONIER_FINAL/"),
#    "*.fits",
#)


container_data = distroi.read_oi_container_from_oifits(data_dir, data_file)

# Make plots of the data, including the uv coverage
container_data.plot_data(fig_dir=None, log_plotv=True, data_figsize=(5,5))

# Make some histograms of VIS2 and T3PHI errors
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
v2_snr = container_data.v2 / container_data.v2_err
snr_stepsize = 5
bins = np.arange(snr_stepsize, ((int(np.max(v2_snr)) // snr_stepsize) + 1) * snr_stepsize, snr_stepsize)
print(bins)
ax[0].hist(v2_snr, edgecolor="k", bins=bins)
ax[0].axvline(x=5, ls="--", c="r")
ax[0].set_title("VIS2 SNR")
ax[1].hist(container_data.t3_phierr, edgecolor="k")
ax[1].axvline(x=10, ls="--", c="r")
ax[1].set_title("T3PHI ERR (deg)")

print(f"Size of data set: V2: {np.size(container_data.v2)}, t3phi: {np.size(container_data.t3_phi)}")
print(f"Unique points: V2: {np.size(np.unique(container_data.v2))}, t3phi: {np.size(np.unique(container_data.t3_phi))}")

# Calculate the resolution ellipse from a Gaussian fit to the interferometric point-spread function.
# This method is available in the auxiliary subpackage.
beam = oi_container_calc_gaussian_beam(
    container_data,
    vistype="vis2",
    make_plots=True,
    show_plots=True,
    fig_dir=None,
    num_res=48,
    pix_per_res=4,
    save_uv_coverage_dict=False,
    uv_coverage_dict_path=None,
)
