# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import subprocess
#sys.path.insert(0, os.path.abspath('../code/pipeline'))
#sys.path.insert(0, os.path.abspath('../code/labelCNN'))
#sys.path.insert(0, os.path.abspath('../code'))

#sys.path.insert(0, os.path.abspath('../classification/pipeline'))
#sys.path.insert(0, os.path.abspath('../classification/supervised/labelCNN'))

sys.path.insert(0, os.path.abspath('../preprocessing/'))
sys.path.insert(0, os.path.abspath('../labeling/'))
sys.path.insert(0, os.path.abspath('../classification/'))

# Build api docs first
# api is a folder, with .rst files telling how to display files in code?!?
subprocess.run(['sphinx-apidoc',
                '-f',
                '-o',
                'api',
                # os.path.abspath('../code')])
                os.path.abspath('../preprocessing')])


# -- Project information -----------------------------------------------------

project = 'asparagus'
copyright = '2020, study project'
author = 'study project'

# The full version, including alpha/beta/rc tags
release = '0.1'

# making index to master_doc so that readthedocs does not search for contents.rst
master_doc = 'index'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'recommonmark',
    'sphinx.ext.autosummary',
    'sphinx_autopackagesummary'
]

# Napoleon settings
napoleon_google_docstring = True
# Include autosummary
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']
