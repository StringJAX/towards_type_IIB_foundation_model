"""Sphinx configuration for the documentation of *Towards a foundation model for Type IIB flux vacua*.

Purpose
-------
Configure paths, project metadata, Sphinx extensions, Markdown handling and HTML output for the documentation
build. The layout follows JAXVacua's documentation (``documentation/source`` with a sibling ``requirements.txt``).

Design notes
------------
The repository is not an installable package. The API reference imports the code the way the notebooks do:
the transformer modules from ``transformer/`` (``jax_transformer``, ``evaluation_scripts``, ``data_scripts``) and
the diffusion modules as ``diffusion.src`` from the repository root. Building the API pages therefore needs the
runtime dependencies (``requirements-cpu.txt``).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "transformer"))
sys.path.insert(0, str(REPO_ROOT))

project = "Towards a foundation model for Type IIB flux vacua"
copyright = "2026, S. Krippendorf, M. Larfors, Z. Liu, A. Schachner, M. Walden"
author = "S. Krippendorf, M. Larfors, Z. Liu, A. Schachner, M. Walden"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
]

templates_path = ["_templates"]
source_suffix = [".rst", ".md"]
exclude_patterns = ["_build"]
# myst.header: the README's notebook section goes from its H2 straight to H3 subsections, which the included pages
# start below their own H1. ref.python: shape expressions in docstring types (e.g. ``np.ndarray([max_len])``) are
# parsed as cross-references, and names such as ``max_len`` are attributes of several classes.
suppress_warnings = ["myst.header", "ref.python"]

autosummary_generate = True
autodoc_default_options = {"members": True, "show-inheritance": True}
autodoc_member_order = "bysource"
napoleon_use_rtype = False
napoleon_custom_sections = [("Returns", "params_style")]
add_module_names = False
toc_object_entries_show_parents = "hide"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/StringJAX/towards_type_IIB_foundation_model",
    "repository_branch": "main",
    "path_to_docs": "documentation/source",
    "use_repository_button": True,
    "use_edit_page_button": True,
}
html_static_path = ["_static"]

nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
myst_heading_anchors = 4
mathjax3_config = {
    "tex": {"inlineMath": [["$", "$"], ["\\(", "\\)"]]},
    "svg": {"fontCache": "global"},
}
