# Configuration file for the Sphinx documentation builder.
import os
import sys

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'Two-Time-Scale Transfer Learning (TTSTL)'
copyright = '2026, Juan Diego Mejia Becerra, Gareth W. Peters, Ruimeng Hu'
author = 'Juan Diego Mejia Becerra, Gareth W. Peters, Ruimeng Hu'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.viewcode',  # Add links to highlighted source code
    'sphinx.ext.mathjax',   # Render math via MathJax
]

# Mock rpy2 to avoid GHA/Environment linkage errors during doc build
autodoc_mock_imports = [
    'rpy2',
    'rpy2.robjects',
    'rpy2.robjects.packages',
    'rpy2.rinterface', 
    'rpy2.rinterface_lib',
    'rpy2.rinterface_lib.callbacks',
]

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
