"""Protocols for extensible lancell components."""

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    import anndata as ad
    import polars as pl

    from lancell.atlas import PointerFieldInfo, RaggedAtlas
    from lancell.group_specs import ZarrGroupSpec


@runtime_checkable
class Reconstructor(Protocol):
    """Protocol for feature-space reconstruction strategies.

    Implementations must provide an ``as_anndata`` method that reads zarr data
    for a single feature space and assembles an AnnData object.
    """

    def as_anndata(
        self,
        atlas: "RaggedAtlas",
        cells_pl: "pl.DataFrame",
        pf: "PointerFieldInfo",
        spec: "ZarrGroupSpec",
        layer_overrides: "list[str] | None" = None,
        feature_join: "Literal['union', 'intersection']" = "union",
    ) -> "ad.AnnData": ...
