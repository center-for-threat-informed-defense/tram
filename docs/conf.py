import os
import sys

import sphinx.ext.apidoc as apidoc

sys.path.insert(0, os.path.abspath('..'))
apidocs_argv = ['-o', '_generated', '--implicit-namespaces', '--force', '../app/']
apidoc.main(apidocs_argv)

# -- Project information -----------------------------------------------------

project = 'tram'
copyright = '2020, The MITRE Corporation'
author = 'The MITRE Corporation'
master_doc = 'index'


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'recommonmark',
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
