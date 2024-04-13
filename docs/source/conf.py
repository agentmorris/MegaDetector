# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MegaDetector'
copyright = '2024, Your friendly neighborhood MegaDetector team'
author = 'Your friendly neighborhood MegaDetector team'
# release = '5.0.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "autoapi.extension",
    #"sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.apidoc",
    "sphinx_mdinclude",
]

templates_path = ['_templates']
exclude_patterns = []

#autoapi opts
autoapi_type = "python"
autoapi_dirs = ["../.."]
autoapi_ignore = [
    "*/archive/*",
    "*/sanbox/*",
    "*/images/*",
]

autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "special-members",
    "imported-members",
]

#myst opts
myst_enable_extensions = [
    "colon_fence",
]

#apidoc opts
apidoc_module_dir = '../../'
apidoc_output_dir = 'modules'
apidoc_excluded_paths = ['archive', 'envs', 'images', 'sandbox']
apidoc_separate_modules = True
apidocs_module_fist = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']