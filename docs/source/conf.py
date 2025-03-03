import sys
import os
import shutil

# Define paths
repository_root = os.path.abspath('../..')

# Add the repository root to Python's path so Sphinx can find the package
sys.path.insert(0, repository_root)

# Add repository root to MyST's image path
myst_url_schemes = ["http", "https", "mailto", ""]
myst_image_path = [repository_root]

project = 'MegaDetector'
copyright = '2024, Your friendly neighborhood MegaDetector team'
author = 'Your friendly neighborhood MegaDetector team'

import builtins
builtins.__sphinx_build__ = True

# Set up the base URL for HTML images
html_baseurl = repository_root
html_extra_path = [os.path.join(repository_root, 'images')]

# Make sure images get copied to the build directory
html_copy_source = True

# Copy images to the correct subdirectory
def setup(app):
    # Ensure the images directory exists in the build directory
    build_images_dir = os.path.join(app.outdir, 'images')
    if not os.path.exists(build_images_dir):
        os.makedirs(build_images_dir)
    
    # Copy all images from the source images directory to the build images directory
    source_images_dir = os.path.join(repository_root, 'images')
    for image in os.listdir(source_images_dir):
        if image.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            shutil.copy2(
                os.path.join(source_images_dir, image),
                os.path.join(build_images_dir, image)
            )
            
# Suppress specific warnings
suppress_warnings = [
    'image.not_readable',
]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_argparse_cli"
]

autodoc_mock_imports = ["azure", "deepdiff", "magic", "tensorflow", "pytesseract"]

# Configure myst-parser to handle GitHub-flavored markdown
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

# Enable anchor links
myst_heading_anchors = 3

html_theme = 'sphinx_rtd_theme'

# collapse_navigation doesn't actually work
html_theme_options = {'navigation_depth': 2, 'collapse_navigation': False}

# html_theme = 'sphinx_book_theme'
# html_theme_options['show_navbar_depth'] = 2

# html_static_path = ['_static']

# Hide "bases: object" from all classes that don't define a base class
from sphinx.ext import autodoc

class MockedClassDocumenter(autodoc.ClassDocumenter):
    def add_line(self, line: str, source: str, *lineno: int) -> None:
        if line == "   Bases: :py:class:`object`":
            return
        super().add_line(line, source, *lineno)

autodoc.ClassDocumenter = MockedClassDocumenter
