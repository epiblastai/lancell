"""AtlasQuery: fluent query builder for reading cells from a RaggedAtlas."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import mudata as mu

    from lancell.dataloader import CellDataset, MultimodalCellDataset

import anndata as ad
import lancedb
import numpy as np
import polars as pl

from lancell.atlas import RaggedAtlas
from lancell.group_specs import get_spec
from lancell.obs_alignment import PointerFieldInfo
from lancell.reconstruction import (
    _build_obs_only_anndata,
    _get_pointer_columns,
)


class AtlasQuery:
    """Fluent query builder for reading cells from a RaggedAtlas."""

    def __init__(self, atlas: RaggedAtlas) -> None:
        self._atlas = atlas
        self._search_query: np.ndarray | list[float] | str | None = None
        self._search_kwargs: dict = {}
        self._where_clause: str | None = None
        self._limit_n: int | None = None
        self._select_columns: list[str] | None = None
        self._feature_spaces: list[str] | None = None
        self._layer_overrides: dict[str, list[str]] = {}
        self._feature_join: Literal["union", "intersection"] = "union"
        self._feature_filter: dict[str, list[str]] = {}

    def search(
        self,
        query: "np.ndarray | list[float] | str | None" = None,
        *,
        vector_column_name: str | None = None,
        query_type: str = "auto",
        fts_columns: str | list[str] | None = None,
    ) -> "AtlasQuery":
        """Add a vector or full-text search to the query.

        Parameters are forwarded to ``lancedb.Table.search()``.

        Parameters
        ----------
        query:
            A vector (ndarray / list), full-text search string, or ``None``
            for a full scan.
        vector_column_name:
            Which vector column to search against.
        query_type:
            One of ``"auto"``, ``"vector"``, ``"fts"``, or ``"hybrid"``.
        fts_columns:
            Column(s) to search for full-text queries.
        """
        self._search_query = query
        self._search_kwargs = {
            "vector_column_name": vector_column_name,
            "query_type": query_type,
            "fts_columns": fts_columns,
        }
        return self

    def select(self, columns: list[str]) -> "AtlasQuery":
        """Select specific metadata columns to return.

        Pointer columns required for AnnData reconstruction are always
        loaded internally, even if not listed here.

        Parameters
        ----------
        columns:
            Column names to include in the results.
        """
        if not isinstance(columns, list):
            raise ValueError("Columns must be a list")
        self._select_columns = columns
        return self

    def where(self, condition: str) -> "AtlasQuery":
        """Add a SQL WHERE filter (LanceDB syntax)."""
        self._where_clause = condition
        return self

    def limit(self, n: int) -> "AtlasQuery":
        """Limit the number of cells returned."""
        self._limit_n = n
        return self

    def feature_spaces(self, *spaces: str) -> "AtlasQuery":
        """Restrict reconstruction to specific feature spaces."""
        self._feature_spaces = list(spaces)
        return self

    def layers(self, feature_space: str, names: list[str]) -> "AtlasQuery":
        """Specify which layers to read for a given feature space."""
        self._layer_overrides[feature_space] = names
        return self

    def features(self, uids: list[str], feature_space: str) -> "AtlasQuery":
        """Filter output to specific features by global UID.

        When set, reconstruction for this feature space returns only the
        requested features. The ``feature_join`` setting is ignored for
        filtered feature spaces; intersection semantics are used.
        """
        self._feature_filter[feature_space] = list(uids)
        return self

    def feature_join(self, join: Literal["union", "intersection"]) -> "AtlasQuery":
        """Set how features are joined across zarr groups.

        ``"union"`` (default) includes all features from any group.
        ``"intersection"`` includes only features present in every group.
        """
        self._feature_join = join
        return self

    # -- Execution ----------------------------------------------------------

    def _build_base_query(self) -> lancedb.table.Table:
        """Build a query with search, where, and limit applied (no column selection)."""
        q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        return q

    def _build_scanner(self) -> lancedb.table.Table:
        """Build a LanceDB query from the current state."""
        q = self._build_base_query()
        if self._select_columns is not None:
            pointer_cols = list(self._atlas._pointer_fields.keys())
            columns = list(dict.fromkeys(self._select_columns + pointer_cols))
            q = q.select(columns)
        return q

    def _active_pointer_fields(self) -> dict[str, PointerFieldInfo]:
        """Return pointer fields filtered by requested feature spaces."""
        pfs = self._atlas._pointer_fields
        if self._feature_spaces is None:
            return pfs
        return {k: v for k, v in pfs.items() if v.feature_space in self._feature_spaces}

    def count(self, group_by: str | list[str] | None = None) -> "pl.DataFrame | int":
        """Count cells, optionally grouped by metadata columns.

        Only the grouping columns are fetched from LanceDB, so this is much
        cheaper than ``to_polars()`` for large atlases.

        Parameters
        ----------
        group_by:
            Column name(s) to group by.  If ``None``, returns a scalar count.

        Returns
        -------
        int
            Total cell count when ``group_by`` is ``None``.
        pl.DataFrame
            DataFrame with one row per group and a ``count`` column otherwise.
        """
        q = self._build_base_query()

        if group_by is None:
            # Fetch only a single cheap column to count rows
            any_col = self._atlas.cell_table.schema.names[0]
            return len(q.select([any_col]).to_arrow())

        cols = [group_by] if isinstance(group_by, str) else list(group_by)
        result = q.select(cols).to_polars()
        return result.group_by(cols).agg(pl.len().alias("count")).sort(cols)

    def to_polars(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame of cell metadata."""
        result = self._build_scanner().to_polars()
        if self._select_columns is not None:
            pointer_cols = _get_pointer_columns(result)
            keep = [c for c in result.columns if c not in pointer_cols]
            result = result.select(keep)
        return result

    def to_anndata(self) -> ad.AnnData:
        """Execute the query and reconstruct an AnnData.

        If multiple feature spaces are active, only the first sparse feature
        space is used for X. Use :meth:`to_mudata` for multi-modal.
        """
        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return _build_obs_only_anndata(cells_pl)

        active_pfs = self._active_pointer_fields()
        # Pick the first feature space for X
        if not active_pfs:
            return _build_obs_only_anndata(cells_pl)

        # Use first pointer field
        pf = next(iter(active_pfs.values()))
        return self._reconstruct_single_space_anndata(cells_pl, pf)

    def to_mudata(self) -> "mu.MuData":
        """Execute the query and return a MuData with one modality per feature space."""
        import mudata as mu

        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return mu.MuData({})

        active_pfs = self._active_pointer_fields()
        modalities: dict[str, ad.AnnData] = {}
        for pf in active_pfs.values():
            adata = self._reconstruct_single_space_anndata(cells_pl, pf)
            if adata.n_obs > 0:
                modalities[pf.feature_space] = adata

        return mu.MuData(modalities)

    def to_batches(self, batch_size: int = 1024) -> Iterator[ad.AnnData]:
        """Stream results as AnnData batches.

        Each batch contains up to ``batch_size`` cells. BatchArray readers
        and remap arrays are cached on the atlas for reuse across batches.
        """
        q = self._build_scanner()
        reader = q.to_batches(batch_size=batch_size)

        active_pfs = self._active_pointer_fields()
        if not active_pfs:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                yield _build_obs_only_anndata(pl.from_arrow(batch))
            return

        pf = next(iter(active_pfs.values()))
        for batch in reader:
            if batch.num_rows == 0:
                continue
            yield self._reconstruct_single_space_anndata(pl.from_arrow(batch), pf)

    def to_cell_dataset(
        self,
        feature_space: str,
        layer: str,
        metadata_columns: list[str] | None = None,
    ) -> "CellDataset":
        """Create a CellDataset for fast ML training iteration.

        Unlike :meth:`to_batches` (which reconstructs full AnnData per batch),
        this returns a :class:`~lancell.dataloader.CellDataset` that yields
        lightweight :class:`~lancell.dataloader.SparseBatch` objects via
        :meth:`~lancell.dataloader.CellDataset.__getitems__`.

        Pair with a :class:`~lancell.sampler.CellSampler` or
        :class:`~lancell.sampler.BalancedCellSampler` for batch planning, then
        use :func:`~lancell.dataloader.make_loader` to create the DataLoader.

        Parameters
        ----------
        feature_space:
            Which feature space to read.
        layer:
            Which layer to read within the feature space.
        metadata_columns:
            Obs column names to include as metadata on each SparseBatch.

        Notes
        -----
        If a feature filter was set on this query (via
        :meth:`~lancell.query.AtlasQuery.feature_spaces`), the returned
        dataset's feature space is automatically restricted to those features
        (``wanted_globals`` is derived from the filter; ``n_features`` reflects
        the filtered count).
        """
        from lancell.dataloader import CellDataset

        cells_pl = self._build_scanner().to_polars()

        wanted_globals = None
        if feature_space in self._feature_filter:
            from lancell.dataset_vars import resolve_feature_uids_to_global_indices

            wanted_globals = resolve_feature_uids_to_global_indices(
                self._atlas._registry_tables[feature_space],
                self._feature_filter[feature_space],
            )

        return CellDataset(
            atlas=self._atlas,
            cells_pl=cells_pl,
            feature_space=feature_space,
            layer=layer,
            metadata_columns=metadata_columns,
            wanted_globals=wanted_globals,
        )

    def to_multimodal_dataset(
        self,
        feature_spaces: list[str],
        layers: dict[str, str] | None = None,
        metadata_columns: list[str] | None = None,
    ) -> "MultimodalCellDataset":
        """Create a MultimodalCellDataset for within-cell multimodal training.

        Each yielded :class:`~lancell.dataloader.MultimodalBatch` contains
        one sub-batch per modality with only the cells that have that
        modality present. A ``present`` mask tracks membership. No fill
        values are added.

        Pair with :class:`~lancell.sampler.CellSampler` (using
        ``dataset.groups_np``) and :func:`~lancell.dataloader.make_loader`
        for the standard training loop.

        Parameters
        ----------
        feature_spaces:
            Ordered list of feature spaces to include.  The first is
            the "primary" space used to derive ``groups_np``.
        layers:
            ``{feature_space: layer_name}`` mapping.  Defaults to
            ``"counts"`` for each space when omitted.
        metadata_columns:
            Obs column names to include as metadata on each batch.
        """
        from lancell.dataloader import MultimodalCellDataset

        cells_pl = self._build_scanner().to_polars()

        if layers is None:
            layers = {fs: "counts" for fs in feature_spaces}

        wanted_globals: dict[str, np.ndarray] | None = None
        for fs in feature_spaces:
            if fs in self._feature_filter:
                from lancell.dataset_vars import resolve_feature_uids_to_global_indices

                wg = resolve_feature_uids_to_global_indices(
                    self._atlas._registry_tables[fs],
                    self._feature_filter[fs],
                )
                if wanted_globals is None:
                    wanted_globals = {}
                wanted_globals[fs] = wg

        return MultimodalCellDataset(
            atlas=self._atlas,
            cells_pl=cells_pl,
            feature_spaces=feature_spaces,
            layers=layers,
            metadata_columns=metadata_columns,
            wanted_globals=wanted_globals,
        )

    # -- Reconstruction internals -------------------------------------------

    def _reconstruct_single_space_anndata(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
    ) -> ad.AnnData:
        """Reconstruct an AnnData for a single feature space."""
        spec = get_spec(pf.feature_space)
        if spec.reconstructor is None:
            raise ValueError(f"No reconstructor registered for feature space '{pf.feature_space}'")

        wanted_globals = None
        if pf.feature_space in self._feature_filter:
            from lancell.dataset_vars import resolve_feature_uids_to_global_indices

            wanted_globals = resolve_feature_uids_to_global_indices(
                self._atlas._registry_tables[pf.feature_space],
                self._feature_filter[pf.feature_space],
            )

        return spec.reconstructor.as_anndata(
            self._atlas,
            cells_pl,
            pf,
            spec,
            layer_overrides=self._layer_overrides.get(pf.feature_space),
            feature_join=self._feature_join,
            wanted_globals=wanted_globals,
        )
