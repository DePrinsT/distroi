"""
A package allowing for the calculation of optical interferometry (OI) and other observables from 
models including a radiative transfer component.
"""

# Use imports in the __init__.py files to 'expose' classes & methods defined in the (sub)package modules (.py files) to
# the user at a higher level . For example using 'from .image_fft import Image' in this file, the user can just
# access the Image class, after importing distroi, by using simply 'distroi.Image'. Otherwise they would've had
# to use distroi.image.Image. We thus remove the need for the user to always have to type out the module name
# in which the class/method has been defined (which, if you have lots of modules, becomes very tiresome very quickly).

# IMPORT SUBPACKAGES (these need to be imported first if you have dependency on distroi main package modules
# in some of the subpackage modules, otherwise you will get circular import issues)
import distroi.auxiliary
import distroi.data
import distroi.model
import distroi.model.rt_comp
import distroi.model.geom_comp
import distroi.model.dep
import distroi.fitting

### imports from data subpackage
# imports from image module
from distroi.data.image import Image
from distroi.data.image import read_image_list
from distroi.data.image import read_image_mcfost
from distroi.data.image import _image_fft_comp_vis_interpolator
# imports from oi_container module
from distroi.data.oi_container import OIContainer
from distroi.data.oi_container import read_oi_container_from_oifits
from distroi.data.oi_container import oi_container_calc_image_fft_observables
from distroi.data.oi_container import oi_container_plot_data_vs_model
# imports from sed module
from distroi.data.sed import SED
from distroi.data.sed import read_sed_mcfost
from distroi.data.sed import read_sed_repo_phot
from distroi.data.sed import sed_chi2reddened
from distroi.data.sed import sed_reddening_fit
from distroi.data.sed import sed_plot_data_vs_model
###

### imports from model subpackage
## imports from component dependency subpackage 
# imports from spec_dep module
from distroi.model.dep.spec_dep import SpecDep
from distroi.model.dep.spec_dep import FlatSpecDep
from distroi.model.dep.spec_dep import BlackBodySpecDep
from distroi.model.dep.spec_dep import PowerLawSpecDep
## imports from geometric component subpackage
# imports from geom_comp module
from distroi.model.geom_comp.geom_comp import GeomComp
from distroi.model.geom_comp.geom_comp import UniformDisk
from distroi.model.geom_comp.geom_comp import Gaussian
from distroi.model.geom_comp.geom_comp import PointSource
from distroi.model.geom_comp.geom_comp import Overresolved
###
