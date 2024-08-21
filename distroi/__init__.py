"""
This package allows for the calculation of optical interferometry (OI) observables from radiative transfer images.
"""

# Use imports in the __init__.py files to 'expose' classes & methods defined in the (sub)package modules (.py files) to
# the user at a higher level . For example using 'from .image_fft import ImageFFT' in this file, the user can just
# access the ImageFFT class, after importing distroi, by using simply 'distroi.ImageFFT'. Otherwise they would've had
# to use distroi.image_fft.ImageFFT. We thus remove the need for the user to always have to type out the module name
# in which the class/method has been defined (which, if you have lots of modules, becomes very tiresome very quickly).

# IMPORT SUBPACKAGES (these need to be imported first if you have dependency on distroi main package modules
# in some of the subpackage modules, otherwise you will get circular import issues)
import distroi.auxiliary

# IMPORTS FROM DISTROI MAIN PACKAGE

# imports from image_fft module
from distroi.image_fft import ImageFFT
from distroi.image_fft import read_image_fft_list
from distroi.image_fft import read_image_fft_mcfost
from distroi.image_fft import image_fft_comp_vis_interpolator

# imports from oi_container module
from distroi.oi_container import OIContainer
from distroi.oi_container import read_oi_container_from_oifits
from distroi.oi_container import oi_container_calc_image_fft_observables
from distroi.oi_container import oi_container_plot_data_vs_model

# imports from sed module
from distroi.sed import SED
from distroi.sed import read_sed_mcfost
from distroi.sed import read_sed_repo_phot
from distroi.sed import sed_chi2reddened
from distroi.sed import sed_reddening_fit
from distroi.sed import sed_plot_data_vs_model

# imports from geom_comp module
from distroi.geom_comp import GeomComp
from distroi.geom_comp import UniformDisk
from distroi.geom_comp import Gaussian
from distroi.geom_comp import PointSource
from distroi.geom_comp import Overresolved

from distroi.geom_comp import SpecDep
from distroi.geom_comp import FlatSpecDep
from distroi.geom_comp import BlackBodySpecDep
from distroi.geom_comp import PowerLawSpecDep
