"""AtlasQuery: fluent query builder for reading cells from a RaggedAtlas."""

from collections.abc import Iterator

import anndata as ad
import lancedb
import numpy as np
import polars as pl

from lancell.atlas import PointerFieldInfo, RaggedAtlas
from lancell.group_specs import get_spec
from lancell.reconstruction import _build_obs_only_anndata, _get_pointer_columns


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
        assert isinstance(columns, list), "Columns must be a list"
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

    # -- Execution ----------------------------------------------------------

    def _build_scanner(self) -> lancedb.table.Table:
        """Build a LanceDB query from the current state."""
        q = self._atlas.cell_table.search(self._search_query, **self._search_kwargs)
        if self._select_columns is not None:
            pointer_cols = list(self._atlas._pointer_fields.keys())
            columns = list(dict.fromkeys(self._select_columns + pointer_cols))
            q = q.select(columns)
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        return q

    def _active_pointer_fields(self) -> dict[str, PointerFieldInfo]:
        """Return pointer fields filtered by requested feature spaces."""
        pfs = self._atlas._pointer_fields
        if self._feature_spaces is None:
            return pfs
        return {k: v for k, v in pfs.items() if v.feature_space in self._feature_spaces}

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
            return ad.AnnData()

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
        feature_space: str = "gene_expression",
        layer: str = "counts",
        batch_size: int = 1024,
        shuffle: bool = True,
        seed: int | None = None,
        drop_last: bool = False,
        metadata_columns: list[str] | None = None,
    ) -> "CellDataset":
        """Create a CellDataset for fast ML training iteration.

        Unlike :meth:`to_batches` (which reconstructs full AnnData per batch),
        this returns a :class:`~lancell.dataloader.CellDataset` that yields
        lightweight :class:`~lancell.dataloader.SparseBatch` objects.

        Parameters
        ----------
        feature_space:
            Which feature space to read.
        layer:
            Which layer to read within the feature space.
        batch_size:
            Number of cells per batch.
        shuffle:
            Whether to shuffle cells each epoch.
        seed:
            Random seed for reproducibility.
        drop_last:
            Whether to drop the last incomplete batch.
        metadata_columns:
            Obs column names to include as metadata on each SparseBatch.
        """
        from lancell.dataloader import CellDataset

        cells_pl = self._build_scanner().to_polars()
        return CellDataset(
            atlas=self._atlas,
            cells_pl=cells_pl,
            feature_space=feature_space,
            layer=layer,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            metadata_columns=metadata_columns,
        )

    def to_dataloader(
        self,
        collate_fn=None,
        **cell_dataset_kwargs,
    ) -> "torch.utils.data.DataLoader":
        """Create a torch DataLoader for fast ML training.

        Wraps :meth:`to_cell_dataset` in a
        ``torch.utils.data.IterableDataset``, configured for single-process
        iteration (``num_workers=0``, ``batch_size=None``).

        Parameters
        ----------
        collate_fn:
            Optional function to transform each :class:`SparseBatch`.
            See :func:`~lancell.dataloader.sparse_to_dense_collate` and
            :func:`~lancell.dataloader.sparse_to_csr_collate`.
        **cell_dataset_kwargs:
            Forwarded to :meth:`to_cell_dataset`.
        """
        from torch.utils.data import DataLoader

        from lancell.dataloader import TorchCellDataset

        dataset = self.to_cell_dataset(**cell_dataset_kwargs)
        torch_dataset = TorchCellDataset(dataset)

        return DataLoader(
            torch_dataset,
            batch_size=None,
            collate_fn=collate_fn,
            num_workers=0,
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
            raise ValueError(
                f"No reconstructor registered for feature space '{pf.feature_space}'"
            )
        return spec.reconstructor.as_anndata(
            self._atlas,
            cells_pl,
            pf,
            spec,
            layer_overrides=self._layer_overrides.get(pf.feature_space),
        )
