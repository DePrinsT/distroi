"""
Testcase showing the calculation of the Gaussian resolution ellipse from the uv coverage of observations. This ellipse
, calculated from a Gaussian fit to the interferometric point-spread-function, represents the smallest element that can
be resolved using the given uv coverage. Uses VLTI/PIONIER observational data of the IRAS08544-4431 system.
Data taken from Corporaal et al. 2023 (A&A 671, A15: https://doi.org/10.1051/0004-6361/202245689).

NOTE: the astrometric precision can actually be much lower than the resolution resulting from this calculation,
but a reconstructed image cannot reliably achieve much greater resolution.
"""

from distroi import oi_container
from distroi.auxiliary import beam

# Read in the data to an OIContainer object
data_dir, data_file = "./data/IRAS08544-4431/PIONIER/", "*.fits"
container_data = oi_container.read_oicontainer_oifits(data_dir, data_file)

# Make plots of the data, including the uv coverage
fig_dir = "./figures/gaussian_beam_calculation/"
container_data.plot_data(fig_dir=fig_dir)

# Calculate the resolution ellipse from a Gaussian fit to the interferometric point-spread function.
beam = beam.calc_gaussian_beam(
    container_data,
    vistype="vis2",
    make_plots=True,
    show_plots=True,
    fig_dir=fig_dir,
    num_res=2,
    pix_per_res=64,
)