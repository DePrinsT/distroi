"""
Subpackage defining interface classes to calculate models using supported RT codes and retrieve the resulting output
for comparison data.
"""

# Use imports in the __init__.py files to 'expose' classes & methods defined in the (sub)package modules (.py files) to
# the user at a higher level . For example using 'from .image_fft import ImageFFT' in this file, the user can just
# access the ImageFFT class, after importing distroi, by using simply 'distroi.ImageFFT'. Otherwise they would've had
# to use distroi.image_fft.ImageFFT. We thus remove the need for the user to always have to type out the module name
# in which the class/method has been defined (which, if you have lots of modules, becomes very tiresome very quickly).

# TODO: add the proper imports to expose these classes and methods to the user.
