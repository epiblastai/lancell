"""Differential expression analysis for lancell atlases.

Requires the ``numba`` optional dependency::

    pip install 'lancell[dex]'
"""

try:
    import numba  # noqa: F401
except ImportError as e:
    raise ImportError("lancell.dex requires numba. Install with: pip install 'lancell[dex]'") from e

from lancell.dex._dex import dex

__all__ = ["dex"]
