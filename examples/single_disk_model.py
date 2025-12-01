"""
Testcase showing the FFT calculations and plots produced for a single MCFOST disk model. Results are
compared to the multi-band VLTI data of the IRAS 08544-4431 system, which hosts a circumbinary disk.
Includes observations taken with the PIONIER, GRAVITY and MATISSE (L and N band) VLTI instruments.

Data taken from Corporaal et al. 2023 (A&A 671, A15: https://doi.org/10.1051/0004-6361/202245689).
"""

import distroi

# PIONIER tests
# ------------------------
print("TEST ON VLTI/PIONIER DATA")
data_dir, data_file = "./data/IRAS08544-4431/PIONIER/", "*.fits"
mod_dir = "./models/IRAS08544-4431_test_model/"
fig_dir = "./figures/single_disk_model/PIONIER"

# FFT test + output info on frequencies
img_dir = "PIONIER/data_1.65/"
img = distroi.read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz", disk_only=True)
print("Printing frequency info: \n", img.freq_info())
img.diagnostic_plot(f"{fig_dir}/fft", log_plotv=True, show_plots=True)

# Monochromatic model observables test
img_dir = "PIONIER/data_1.65/"
container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(1.63, 1.65))
img_ffts = distroi.read_image_list(mod_dir, img_dir)
container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
distroi.oi_container_plot_data_vs_model(
    container_data, container_model, fig_dir=f"{fig_dir}/monochromatic", show_plots=True
)

# Chromatic model observables test
img_dir = "PIONIER/"
container_data = distroi.read_oi_container_from_oifits(data_dir, data_file)
img_ffts = distroi.read_image_list(mod_dir, img_dir)
container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
distroi.oi_container_plot_data_vs_model(
    container_data, container_model, fig_dir=f"{fig_dir}/chromatic", show_plots=True
)
#
## GRAVITY tests
## ------------------------
#print("TEST ON VLTI/GRAVITY DATA")
#data_dir, data_file = "./data/IRAS08544-4431/GRAVITY/", "*1.fits"
#mod_dir = "./models/IRAS08544-4431_test_model/"
#fig_dir = "./figures/single_disk_model/GRAVITY"
#
## FFT test
#img_dir = "GRAVITY/data_2.2/"
#img = distroi.read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz", disk_only=True)  # load in img
#img.diagnostic_plot(f"{fig_dir}/fft", plot_vistype="vis", log_plotv=True, show_plots=True)
#print(img.freq_info())
#
## Monochromatic model observables test
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(2.1, 2.3))
#img_ffts = distroi.read_image_list(mod_dir, img_dir)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data, container_model, fig_dir=f"{fig_dir}/monochromatic", show_plots=True
#)
#
## Chromatic model observables test
#img_dir = "GRAVITY/"
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file)
#img_ffts = distroi.read_image_list(mod_dir, img_dir)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data, container_model, fig_dir=f"{fig_dir}/chromatic", show_plots=True
#)
#
## MATISSE L-BAND tests
## ------------------------
#print("TEST ON VLTI/MATISSE L-BAND DATA")
#data_dir, data_file = "./data/IRAS08544-4431/MATISSE_L/", "*.fits"
#mod_dir = "./models/IRAS08544-4431_test_model/"
#fig_dir = "./figures/single_disk_model/MATISSE_L"
#
## FFT test
#img_dir = "MATISSE_L/data_3.5/"
#img = distroi.read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz")  # load in img
#img.diagnostic_plot(f"{fig_dir}/fft", log_plotv=True, show_plots=True)
#print(img.freq_info())
#
## Monochromatic model observables test
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(3.48, 3.55))
#img_ffts = distroi.read_image_list(mod_dir, img_dir)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data, container_model, fig_dir=f"{fig_dir}/monochromatic", log_plotv=True, show_plots=True
#)
#
## Chromatic model observables test
#img_dir = "MATISSE_L/"
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(2.95, 3.95), v2lim=1e-8)
#img_ffts = distroi.read_image_list(mod_dir, img_dir)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data, container_model, fig_dir=f"{fig_dir}/chromatic", log_plotv=True, show_plots=True
#)
#
## MATISSE N-BAND tests
## ------------------------
#print("TEST ON VLTI/MATISSE N-BAND DATA")
#data_dir, data_file = "./data/IRAS08544-4431/MATISSE_N/", "*.fits"
#mod_dir = "./models/IRAS08544-4431_test_model/"
#fig_dir = "./figures/single_disk_model/MATISSE_N"
#
## FFT test
#img_dir = "MATISSE_N/data_10.0/"
#img = distroi.read_image_mcfost(img_path=f"{mod_dir}{img_dir}/RT.fits.gz")  # load in img
#img.redden(ebminv=1.4)
#img.diagnostic_plot(f"{fig_dir}/fft", plot_vistype="fcorr", log_plotv=True, show_plots=True)
#print(img.freq_info())
#
## Monochromatic model observables test
#img_dir = "MATISSE_N/data_10.0/"
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(9.75, 10.20), fcorr=True)
#img_ffts = distroi.read_image_list(mod_dir, img_dir, ebminv=1.4)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data,
#    container_model,
#    fig_dir=f"{fig_dir}/monochromatic",
#    log_plotv=True,
#    show_plots=True,
#    plot_vistype="vis",
#)
#
## Chromatic model observables test
#img_dir = "MATISSE_N/"
#container_data = distroi.read_oi_container_from_oifits(data_dir, data_file, wave_lims=(8.5, 12.0), fcorr=True)
#img_ffts = distroi.read_image_list(mod_dir, img_dir, ebminv=1.4)
#container_model = distroi.oi_container_calc_image_fft_observables(container_data, img_ffts)
#distroi.oi_container_plot_data_vs_model(
#    container_data, container_model, fig_dir=f"{fig_dir}/chromatic", log_plotv=True, show_plots=True, plot_vistype="vis"
#)
