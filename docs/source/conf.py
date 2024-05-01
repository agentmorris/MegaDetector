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
    "sphinx_argparse_cli"
]

autodoc_mock_imports = ["azure", "deepdiff", "magic", "tensorflow", "pytesseract"]

myst_enable_extensions = [
    "colon_fence",
]

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# Hide "bases: object" from all classes that don't define a base class
from sphinx.ext import autodoc

class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter