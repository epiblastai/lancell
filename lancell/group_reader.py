"""GroupReader: per-(zarr_group, feature_space) read state."""

import lancedb
import numpy as np
import obstore
import polars as pl
import zarr

from lancell.batch_array import BatchAsyncArray
from lancell.dataset_vars import read_dataset_vars


class GroupReader:
    """Encapsulates all per-(zarr_group, feature_space) read state.

    Used by both the reconstruction path (RaggedAtlas) and the ML training
    path (CellDataset worker processes). Zarr handles are opened lazily and
    zeroed on pickling; all other state is picklable.

    Create via the two factories:
    - :meth:`from_atlas_root` — atlas reconstruction path
    - :meth:`for_worker` — DataLoader worker path
    """

    def __init__(
        self,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        dataset_vars_table: lancedb.table.Table | None,
        dataset_uid: str | None,
        remap_cache: tuple[int, np.ndarray] | None = None,
        var_df_cache: tuple[int, pl.DataFrame] | None = None,
        zarr_group_handle: zarr.Group | None = None,
    ) -> None:
        self.zarr_group = zarr_group
        self.feature_space = feature_space
        self._store = store
        self._dataset_vars_table = dataset_vars_table
        self._dataset_uid = dataset_uid
        self._remap_cache = remap_cache
        self._var_df_cache = var_df_cache
        self._zarr_group_handle = zarr_group_handle
        self._array_reader_cache: dict[str, BatchAsyncArray] = {}

    @classmethod
    def from_atlas_root(
        cls,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        dataset_vars_table: lancedb.table.Table | None,
        dataset_uid: str | None,
    ) -> "GroupReader":
        """Create a GroupReader for an atlas.

        The zarr group handle is opened lazily on first array access.
        Used by ``RaggedAtlas._get_group_reader``.
        ``dataset_vars_table`` may be ``None`` for old atlases or feature
        spaces with ``has_var_df=False``.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            dataset_vars_table=dataset_vars_table,
            dataset_uid=dataset_uid,
        )

    @classmethod
    def for_worker(
        cls,
        zarr_group: str,
        feature_space: str,
        store: obstore.store.ObjectStore,
        remap: np.ndarray,
    ) -> "GroupReader":
        """Create a GroupReader for a DataLoader worker.

        Accepts a pre-resolved remap (already version-checked at CellDataset
        init time). Sets ``_dataset_vars_table=None`` — workers never re-check
        table version. The zarr group handle is ``None`` until first use.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            dataset_vars_table=None,
            dataset_uid=None,
            remap_cache=(0, remap),
        )

    def get_remap(self) -> np.ndarray:
        """Return the local-to-global-index remap array.

        If ``_dataset_vars_table`` is ``None`` (worker path), returns the frozen
        remap directly. Otherwise performs a version-aware cache check and
        rebuilds from the Lance table if stale.
        """
        if self._dataset_vars_table is None:
            assert self._remap_cache is not None, (
                f"GroupReader for {self.zarr_group!r} has no remap. "
                "The for_worker path requires a remap at construction time."
            )
            return self._remap_cache[1]

        if self._dataset_uid is None:
            raise ValueError(
                f"GroupReader for {self.zarr_group!r} has no dataset_uid. "
                "Cannot load remap from _dataset_vars."
            )

        current_version = self._dataset_vars_table.version
        if self._remap_cache is not None:
            cached_version, cached_remap = self._remap_cache
            if cached_version == current_version:
                return cached_remap

        # Cache miss or stale — read from Lance table
        rows = read_dataset_vars(self._dataset_vars_table, self._dataset_uid)
        remap = rows["global_index"].to_numpy().astype(np.int32, copy=False)
        self._remap_cache = (current_version, remap)
        return remap

    @property
    def var_df(self) -> pl.DataFrame:
        """Load and cache var_df for this zarr group.

        Returns a DataFrame with columns ``global_feature_uid``, ``csc_start``,
        ``csc_end`` in local feature order (row i = local feature i).
        """
        if self._dataset_vars_table is None or self._dataset_uid is None:
            return pl.DataFrame(
                schema={
                    "global_feature_uid": pl.Utf8,
                    "csc_start": pl.Int64,
                    "csc_end": pl.Int64,
                }
            )

        current_version = self._dataset_vars_table.version
        if self._var_df_cache is not None:
            cached_version, cached_df = self._var_df_cache
            if cached_version == current_version:
                return cached_df

        rows = read_dataset_vars(self._dataset_vars_table, self._dataset_uid)
        df = rows.select(
            [
                pl.col("feature_uid").alias("global_feature_uid"),
                pl.col("csc_start"),
                pl.col("csc_end"),
            ]
        )
        self._var_df_cache = (current_version, df)
        return df

    @property
    def has_csc(self) -> bool:
        """Return True if this zarr group has CSC data."""
        df = self.var_df
        return len(df) > 0 and df["csc_start"].null_count() == 0 and df["csc_end"].null_count() == 0

    def get_array_reader(self, array_name: str) -> BatchAsyncArray:
        """Return a cached BatchAsyncArray reader for a zarr array."""
        self._ensure_initialized()
        reader = self._array_reader_cache.get(array_name)
        if reader is None:
            reader = BatchAsyncArray.from_array(self._zarr_group_handle[array_name])
            self._array_reader_cache[array_name] = reader
        return reader

    def _ensure_initialized(self) -> None:
        """Open the zarr group handle lazily if not yet done."""
        if self._zarr_group_handle is None:
            root = zarr.open_group(zarr.storage.ObjectStore(self._store), mode="r")
            self._zarr_group_handle = root[self.zarr_group]
            self._array_reader_cache = {}

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        # Zero transient zarr state so the object is safely picklable.
        state["_zarr_group_handle"] = None
        state["_array_reader_cache"] = {}
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
