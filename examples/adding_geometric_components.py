"""
Testcase showing the FFT calculations and plots produced for a single MCFOST disk model. Results are
compared to PIONIER observations of the IRAS 08544-4431 system, which hosts a circumbinary disk.

Data taken from Corporaal et al. 2023 (A&A 671, A15: https://doi.org/10.1051/0004-6361/202245689).
"""

import distroi

# Read in the data to an OIContainer object
data_dir, data_file = "./data/IRAS08544-4431/PIONIER/", "*.fits"
container_data = distroi.read_oi_container_from_oifits(data_dir, data_file)
container_data.plot_data()

# set path to mcfost model (including where to find model images) and output plotted figures
mod_dir = "./models/IRAS08544-4431_test_model/"
img_dir = "PIONIER/"
fig_dir = "./figures/adding_geometric_components/PIONIER"

# Chromatic model observables test without geometric component
img_ffts = distroi.read_image_list(mod_dir, img_dir)
container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
distroi.oi_container_plot_data_vs_model(container_data, container_model, fig_dir=f"{fig_dir}", show_plots=True)

# Adding a grey background
background = distroi.Overresolved(sp_dep=distroi.FlatSpecDep(flux_form="flam"))

container_model = distroi.oi_container_calc_image_fft_observables(
    container_data, img_ffts, geom_comps=[background], geom_comp_flux_fracs=[0.70], ref_wavelength=1.65
)
distroi.oi_container_plot_data_vs_model(container_data, container_model, fig_dir=f"{fig_dir}", show_plots=True)

# Adding a secondary star
secondary = distroi.UniformDisk(diameter=0.1, coords=(0, 0), sp_dep=distroi.FlatSpecDep(flux_form="flam"))

container_model = distroi.oi_container_calc_image_fft_observables(
    container_data, img_ffts, geom_comps=[background, secondary], geom_comp_flux_fracs=[0.01, 0.98],
    ref_wavelength=1.65
)
distroi.oi_container_plot_data_vs_model(container_data, container_model, fig_dir=f"{fig_dir}", show_plots=True)
