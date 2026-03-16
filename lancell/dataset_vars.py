"""Deprecated: per-dataset feature mapping helpers.

This module is kept for backward compatibility. New code should use
:mod:`lancell.feature_layouts` instead.
"""

# Re-export everything from feature_layouts for backward compatibility
from lancell.feature_layouts import (
    reindex_registry,
    resolve_feature_uids_to_global_indices,
)

__all__ = [
    "reindex_registry",
    "resolve_feature_uids_to_global_indices",
]
