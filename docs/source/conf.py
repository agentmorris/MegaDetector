import sys
import os

project = 'MegaDetector'
copyright = '2024, Your friendly neighborhood MegaDetector team'
author = 'Your friendly neighborhood MegaDetector team'

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx_mdinclude",
]

autodoc_mock_imports = ["azure", "deepdiff", "magic", "tensorflow", "pytesseract"]

myst_enable_extensions = [
    "colon_fence",
]

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']