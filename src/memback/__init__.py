"""MemBack — neural backmapping of Martini 3 membranes to CHARMM36 all-atom."""

from importlib.metadata import PackageNotFoundError, version

try:
    # Set from the git tag at build time (setuptools-scm, see pyproject.toml).
    __version__ = version("memback")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = ["__version__"]
