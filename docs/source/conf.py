# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "blanket"
copyright = "2023, Adam Coogan & Pablo Lemos"
author = "Adam Coogan & Pablo Lemos"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


def linkcode_resolve(domain, info):
    if domain != "py":
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/Ciela-Institute/blanket/blob/main/src/{filename}.py"


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []
source_suffix = [".md"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
