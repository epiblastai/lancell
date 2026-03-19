"""AtlasQuery: fluent query builder for reading cells from a RaggedAtlas."""

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import mudata as mu
    from lancedb.query import LanceQueryBuilder

    from lancell.dataloader import CellDataset, MultimodalCellDataset

import anndata as ad
import numpy as np
import polars as pl

from lancell.atlas import RaggedAtlas
from lancell.group_specs import get_spec
from lancell.obs_alignment import PointerFieldInfo
from lancell.reconstruction import (
    _build_obs_only_anndata,
    _get_pointer_columns,
)
from lancell.util import sql_escape


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
        self._balanced_limit_n: int | None = None
        self._balanced_limit_column: str | None = None

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
        if self._balanced_limit_n is not None:
            raise ValueError("Cannot use both limit() and balanced_limit() on the same query")
        self._limit_n = n
        return self

    def balanced_limit(self, n: int, column: str) -> "AtlasQuery":
        """Limit cells, drawing equally from each unique value of *column*.

        The result contains at most *n* cells, split evenly across each
        unique value of *column* that passes any ``.where()`` filter.

        Cannot be combined with ``.limit()``.
        """
        if self._limit_n is not None:
            raise ValueError("Cannot use both limit() and balanced_limit() on the same query")
        self._balanced_limit_n = n
        self._balanced_limit_column = column
        return self

    def feature_spaces(self, *spaces: str) -> "AtlasQuery":
        """Restrict reconstruction to specific feature spaces."""
        known = {pf.feature_space for pf in self._atlas._pointer_fields.values()}
        unknown = set(spaces) - known
        if unknown:
            raise ValueError(
                f"Unknown feature space(s): {sorted(unknown)}. Available: {sorted(known)}"
            )
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
        if feature_space not in self._atlas._registry_tables:
            known = sorted(self._atlas._registry_tables.keys())
            raise ValueError(f"No registry for feature space '{feature_space}'. Available: {known}")
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

    def _build_base_query(self) -> "LanceQueryBuilder":
        """Build a query with search, where, and limit applied (no column selection)."""
        q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        return q

    def _build_scanner(self) -> "LanceQueryBuilder":
        """Build a LanceDB query from the current state."""
        q = self._build_base_query()
        if self._select_columns is not None:
            pointer_cols = list(self._atlas._pointer_fields.keys())
            columns = list(dict.fromkeys(self._select_columns + pointer_cols))
            q = q.select(columns)
        return q

    def _materialize_cells(self) -> pl.DataFrame:
        """Materialise the cell DataFrame, respecting balanced_limit if set."""
        if self._balanced_limit_n is not None:
            return self._materialize_balanced()
        return self._build_scanner().to_polars()

    def _materialize_cells_for_dataset(self) -> pl.DataFrame:
        """Materialise a lightweight cell DataFrame with row IDs for CellDataset.

        Returns only pointer columns + ``_rowid`` (lance's built-in row ID).
        Metadata is loaded lazily per batch via ``take_row_ids``.
        """
        if self._balanced_limit_n is not None:
            return self._materialize_balanced_for_dataset()

        q = self._build_base_query().with_row_id(True)
        pointer_cols = list(self._atlas._pointer_fields.keys())
        q = q.select(pointer_cols)
        return q.to_polars()

    def _materialize_balanced_for_dataset(self) -> pl.DataFrame:
        """Two-phase balanced materialisation returning only pointers + _rowid."""
        column = self._balanced_limit_column
        assert column is not None

        discovery_q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            discovery_q = discovery_q.where(self._where_clause)
        unique_values = discovery_q.select([column]).to_polars()[column].unique().to_list()
        n_groups = len(unique_values)
        if n_groups == 0:
            pointer_cols = list(self._atlas._pointer_fields.keys())
            q = self._build_base_query().with_row_id(True).select(pointer_cols)
            return q.to_polars().head(0)

        per_group = self._balanced_limit_n // n_groups
        pointer_cols = list(self._atlas._pointer_fields.keys())

        frames: list[pl.DataFrame] = []
        for val in unique_values:
            escaped = sql_escape(str(val))
            group_filter = f"{column} = '{escaped}'"
            if self._where_clause is not None:
                combined = f"({self._where_clause}) AND ({group_filter})"
            else:
                combined = group_filter

            q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
            q = q.where(combined).limit(per_group).with_row_id(True)
            q = q.select(pointer_cols)
            frames.append(q.to_polars())

        return pl.concat(frames)

    def _materialize_balanced(self) -> pl.DataFrame:
        """Two-phase balanced materialisation."""
        column = self._balanced_limit_column
        assert column is not None

        # Phase 1: discover unique values (fetch only the balance column)
        discovery_q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._where_clause is not None:
            discovery_q = discovery_q.where(self._where_clause)
        unique_values = discovery_q.select([column]).to_polars()[column].unique().to_list()
        n_groups = len(unique_values)
        if n_groups == 0:
            return self._build_scanner().to_polars().head(0)

        per_group = self._balanced_limit_n // n_groups

        # Phase 2: one sub-query per group
        frames: list[pl.DataFrame] = []
        for val in unique_values:
            escaped = sql_escape(str(val))
            group_filter = f"{column} = '{escaped}'"
            if self._where_clause is not None:
                combined = f"({self._where_clause}) AND ({group_filter})"
            else:
                combined = group_filter

            q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
            q = q.where(combined).limit(per_group)
            if self._select_columns is not None:
                pointer_cols = list(self._atlas._pointer_fields.keys())
                columns = list(dict.fromkeys(self._select_columns + pointer_cols))
                q = q.select(columns)
            frames.append(q.to_polars())

        return pl.concat(frames)

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
        result = self._materialize_cells()
        pointer_cols = _get_pointer_columns(result)
        if pointer_cols:
            keep = [c for c in result.columns if c not in pointer_cols]
            result = result.select(keep)
        return result

    def to_anndata(self) -> ad.AnnData:
        """Execute the query and reconstruct an AnnData.

        If multiple feature spaces are active, only the first sparse feature
        space is used for X. Use :meth:`to_mudata` for multi-modal.
        """
        cells_pl = self._materialize_cells()
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

        cells_pl = self._materialize_cells()
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

        When ``balanced_limit`` is active, the full balanced result is
        materialised first and then chunked in Python.
        """
        active_pfs = self._active_pointer_fields()
        pf = next(iter(active_pfs.values())) if active_pfs else None

        if self._balanced_limit_n is not None:
            cells_pl = self._materialize_cells()
            for offset in range(0, len(cells_pl), batch_size):
                chunk = cells_pl.slice(offset, batch_size)
                if chunk.is_empty():
                    continue
                if pf is None:
                    yield _build_obs_only_anndata(chunk)
                else:
                    yield self._reconstruct_single_space_anndata(chunk, pf)
            return

        q = self._build_scanner()
        reader = q.to_batches(batch_size=batch_size)

        if pf is None:
            for batch in reader:
                if batch.num_rows == 0:
                    continue
                yield _build_obs_only_anndata(pl.from_arrow(batch))
            return

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

        Pair with a :class:`~lancell.sampler.CellSampler` for batch planning,
        then use :func:`~lancell.dataloader.make_loader` to create the
        DataLoader.

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

        cells_pl = self._materialize_cells_for_dataset()

        wanted_globals = None
        if feature_space in self._feature_filter:
            from lancell.feature_layouts import resolve_feature_uids_to_global_indices

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

        cells_pl = self._materialize_cells_for_dataset()

        if layers is None:
            layers = {fs: "counts" for fs in feature_spaces}

        wanted_globals: dict[str, np.ndarray] | None = None
        for fs in feature_spaces:
            if fs in self._feature_filter:
                from lancell.feature_layouts import resolve_feature_uids_to_global_indices

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
            from lancell.feature_layouts import resolve_feature_uids_to_global_indices

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
