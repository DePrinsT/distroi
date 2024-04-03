# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Path info
import os
import sys
# add project root to path, needs to be added so ReadTheDocs can find the modules when building documentation
sys.path.append('../..')  
# add all submodules to the path just to be sure, needs to be added to sphinx autodoc can find all files 
for x in os.walk('../../distroi'):
  sys.path.insert(0, x[0])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DISTROI'
copyright = '2024, Toon De Prins'
author = 'Toon De Prins'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
