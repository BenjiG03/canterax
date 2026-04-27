from __future__ import annotations

project = "Canterax"
author = "Canterax contributors"
copyright = "2026, Canterax contributors"
release = "0.3.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "alabaster"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_title = "Canterax documentation"
html_theme_options = {
    "description": "Differentiable chemical kinetics with a Cantera-like API.",
    "fixed_sidebar": True,
    "page_width": "1200px",
    "sidebar_width": "280px",
    "show_relbars": True,
}

autosectionlabel_prefix_document = True
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "jax": ("https://docs.jax.dev/en/latest/", None),
}
