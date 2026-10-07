"""Configuration of the documentation (Sphinx with the PyData theme).

Build the documentation with ``make html`` in this folder (``pip install fibomat[docs]`` installs the requirements).
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent / '_ext'))

import fibomat  # noqa: E402

# -- Project information -----------------------------------------------------------------------------------------------

project = 'fib-o-mat'
author = 'fib-o-mat contributors'
copyright = '2020-2026 fib-o-mat contributors'  # pylint: disable=redefined-builtin

release = fibomat.__version__
version = '.'.join(release.split('.')[:2])

# -- General configuration ---------------------------------------------------------------------------------------------

needs_sphinx = '7.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinxemoji.sphinxemoji',
    'fibomat_plot',
]

exclude_patterns = ['_build', 'LICENSE_DOCS.txt']
source_suffix = {'.rst': 'restructuredtext', '.md': 'markdown'}
master_doc = 'index'

todo_include_todos = True
sphinxemoji_style = 'twemoji'

# the plots in the documentation link to the scripts in the repository
fibomat_source_url = 'https://github.com/fib-o-mat/fib-o-mat/blob/main/examples'

# -- API documentation (autodoc, autosummary, napoleon) ----------------------------------------------------------------

autosummary_generate = True
autosummary_imported_members = False

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'show-inheritance': True,
    'inherited-members': True,
    'undoc-members': False,
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'
autodoc_inherit_docstrings = True

# The docstrings are written in the Google style. The custom section "Access" documents the access of properties.
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_custom_sections = ['Access']
napoleon_use_ivar = False

# Optional dependencies which are not needed to build the API documentation.
autodoc_mock_imports = ['numba', 'PyQt5', 'vispy', 'svgelements', 'ezdxf']

# -- Intersphinx -------------------------------------------------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'sympy': ('https://docs.sympy.org/latest', None),
    'bokeh': ('https://docs.bokeh.org/en/latest', None),
    'pint': ('https://pint.readthedocs.io/en/stable', None),
}

# -- MyST (markdown) ---------------------------------------------------------------------------------------------------

# Sections can be referenced with :ref:`document/path:section title`.
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 3
# The changelog repeats the headings 'Added', 'Changed', ... for every release.
suppress_warnings = ['autosectionlabel.changelog']

myst_enable_extensions = ['colon_fence', 'deflist']

# -- HTML output -------------------------------------------------------------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_title = f'fib-o-mat {release}'
html_logo = 'logo/fibomat.png'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_context = {
    'github_user': 'fib-o-mat',
    'github_repo': 'fib-o-mat',
    'github_version': 'main',
    'doc_path': 'docs',
}

# Version switcher: `_static/switcher.json` lists the published versions of the documentation.
# On Read the Docs, the version which is built is known from the environment, otherwise the version of the package is
# used ("dev" versions are matched with the latest documentation).
_DEPLOYED_SWITCHER = 'https://fib-o-mat.readthedocs.io/en/latest/_static/switcher.json'
_version_match = os.environ.get('READTHEDOCS_VERSION') or ('latest' if 'dev' in release else release)

html_theme_options = {
    'logo': {'text': 'fib-o-mat'},
    'header_links_before_dropdown': 6,
    'navbar_start': ['navbar-logo', 'version-switcher'],
    'navbar_center': ['navbar-nav'],
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    'navbar_persistent': ['search-button'],
    'search_bar_text': 'Search the docs ...',
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/fib-o-mat/fib-o-mat',
            'icon': 'fa-brands fa-github',
        },
        {
            'name': 'PyPI',
            'url': 'https://pypi.org/project/fibomat/',
            'icon': 'fa-brands fa-python',
        },
    ],
    'use_edit_page_button': True,
    'show_toc_level': 2,
    # the remaining top-level pages (Changelog, Contributors, License) go into the "More" dropdown
    'header_links_before_dropdown': 4,
    'navigation_with_keys': False,
    'show_version_warning_banner': True,
    'switcher': {
        'json_url': os.environ.get('FIBOMAT_DOCS_SWITCHER_URL', _DEPLOYED_SWITCHER),
        'version_match': _version_match,
    },
    # the url of the switcher is not reachable for local builds
    'check_switcher': False,
}


def _api_page_context(app, pagename, templatename, context, doctree):
    """Show only the classes and functions (no methods) in the right sidebar of the API reference pages."""
    if pagename.startswith('api/'):
        context['theme_show_toc_level'] = 1


def setup(app):
    """Sphinx hook."""
    app.connect('html-page-context', _api_page_context, priority=100)
