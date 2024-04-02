"""
Testcase showing the calculations and plots produced for a single MCFOST disk model. Applied to the multi-band
VLTI data of the IRAS 08544-4431 system, which hosts a circumbinary disk.
"""

if __name__ == '__main__':
    from distroi import image
    from distroi import oi_observables

    # PIONIER tests
    # ------------------------
    print('TEST ON VLTI/PIONIER DATA')
    data_dir, data_file = './data/IRAS0844-4431/PIONIER/', '*.fits'
    mod_dir = './models/IRAS08544-4431_test_model/'
    fig_dir = './figures/single_disk_model/PIONIER'

    # FFT test
    img_dir = 'PIONIER/data_1.65/'
    img = image.Image(f'{mod_dir}{img_dir}/RT.fits.gz', read_method='mcfost',
                      disk_only=True)  # load in img
    img.fft_diagnostic_plot(fig_dir, log_plotv=True, show_plots=True)
    print(img.freq_info())

    # Monochromatic model observables test
    img_dir = 'PIONIER/data_1.65/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(1.63, 1.65))
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=True)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, show_plots=True)

    # Chromatic model observables test
    img_dir = 'PIONIER/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file)
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=False)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, show_plots=True)

    # GRAVITY tests
    # ------------------------
    print('TEST ON VLTI/GRAVITY DATA')
    data_dir, data_file = './data/IRAS0844-4431/GRAVITY/', '*1.fits'
    mod_dir = './models/IRAS08544-4431_test_model/'
    fig_dir = './figures/single_disk_model/GRAVITY'

    # FFT test
    img_dir = 'GRAVITY/data_2.2/'
    img = image.Image(f'{mod_dir}{img_dir}/RT.fits.gz', read_method='mcfost',
                      disk_only=True)  # load in img
    img.fft_diagnostic_plot(fig_dir, log_plotv=True, show_plots=True)
    print(img.freq_info())

    # Monochromatic model observables test
    img_dir = 'GRAVITY/data_2.2/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(2.1, 2.3))
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=True)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, show_plots=True)

    # Chromatic model observables test
    img_dir = 'GRAVITY/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file)
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=False)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, show_plots=True)

    # MATISSE L-BAND tests
    # ------------------------
    print('TEST ON VLTI/MATISSE L-BAND DATA')
    data_dir, data_file = './data/IRAS0844-4431/MATISSE_L/', '*.fits'
    mod_dir = './models/IRAS08544-4431_test_model/'
    fig_dir = './figures/single_disk_model/MATISSE_L'

    # FFT test
    img_dir = 'MATISSE_L/data_3.5/'
    img = image.Image(f'{mod_dir}{img_dir}/RT.fits.gz', read_method='mcfost',
                      disk_only=True)  # load in img
    img.fft_diagnostic_plot(fig_dir, log_plotv=True, show_plots=True)
    print(img.freq_info())

    # Monochromatic model observables test
    img_dir = 'MATISSE_L/data_3.5/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(3.48, 3.55))
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=True)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                      show_plots=True)

    # Chromatic model observables test
    img_dir = 'MATISSE_L/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(2.95, 3.95), v2lim=1e-8)
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=False)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                      show_plots=True)

    # MATISSE N-BAND tests
    # ------------------------
    print('TEST ON VLTI/MATISSE N-BAND DATA')
    data_dir, data_file = './data/IRAS0844-4431/MATISSE_N/', '*.fits'
    mod_dir = './models/IRAS08544-4431_test_model/'
    fig_dir = './figures/single_disk_model/MATISSE_N'

    # FFT test
    img_dir = 'MATISSE_N/data_10.0/'
    img = image.Image(f'{mod_dir}{img_dir}/RT.fits.gz', read_method='mcfost',
                      disk_only=True)  # load in img
    img.redden(ebminv=1.4)
    img.fft_diagnostic_plot(fig_dir, plot_vistype='fcorr', log_plotv=True, show_plots=True)
    print(img.freq_info())

    # Monochromatic model observables test
    img_dir = 'MATISSE_N/data_10.0/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(9.75, 10.20), v2lim=None,
                                                          fcorr=True)
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=True,
                                                            ebminv=1.4, read_method='mcfost', disk_only=False)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                      show_plots=True, plot_vistype='vis')

    # Chromatic model observables test
    img_dir = 'MATISSE_N/'
    container_data = oi_observables.container_from_oifits(data_dir, data_file, wave_lims=(8.5, 12.0), v2lim=None,
                                                          fcorr=True)
    container_model = oi_observables.calc_model_observables(container_data, mod_dir, img_dir, monochr=False,
                                                            ebminv=1.4, read_method='mcfost', disk_only=False)
    oi_observables.plot_data_vs_model(container_data, container_model, fig_dir=fig_dir, log_plotv=True,
                                      show_plots=True, plot_vistype='vis')
