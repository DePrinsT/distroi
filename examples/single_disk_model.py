"""
Testcase showing the calculations and plots produced for a single disk model. Applied to the multi-band VLTI data of
the IRAS 08544-4431 system, which hosts a circumbinary disk.
"""

from modules import image_fft
import modules.oi_observables as oi_observables
import matplotlib.pyplot as plt
import modules.matplotlib_settings as matplotlib_settings

matplotlib_settings.set_matplotlib_params()  # set project matplotlib parameters

# MATISSE N-BAND tests
# ------------------------
print('TEST ON VLTI/MATISSE N-BAND DATA')
data_dir, data_file = './data/IRAS0844-4431/MATISSE_N/', '*.fits'
mod_dir = './models/IRAS08544-4431_test_model/'
fig_dir = './figures/single_disk_model/MATISSE_N'

# FFT test
img_dir = 'MATISSE_N/data_10.0/'
transform = image_fft.ImageFFT(f'{mod_dir}{img_dir}/RT.fits.gz', read_method='mcfost',
                               disk_only=True)  # load in image
transform.redden(ebminv=0.0)
transform.diagnostic_plot(fig_dir, plot_vistype='fcorr', log_plotv=True, log_ploti=False)
plt.show()
print(transform.freq_info())

# Monochromatic model observables test
img_dir = 'MATISSE_N/data_10.0/'
container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(9.75, 10.20), v2lim=None)
container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=True, fcorr=True,
                                                        ebminv=0.0, read_method='mcfost', disk_only=False)
oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                  plot_vistype='fcorr')
plt.show()

# Chromatic model observables test
img_dir = 'MATISSE_N/'
container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(8.5, 12.0), v2lim=None)
container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=False, fcorr=True,
                                                        ebminv=0.0, read_method='mcfost', disk_only=False)
oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                  plot_vistype='fcorr')
plt.show()
