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
    path (CellDataset worker processes).

    ``zarr_group`` is the string path within the object store (e.g.
    ``"datasets/abc123/rna"``).  It is the durable, picklable identity of
    this reader.  ``_zarr_group_handle`` is the live ``zarr.Group`` object
    derived from that path; it is opened lazily on first array access and
    **zeroed out on pickling** (see ``__getstate__``).  The two are kept
    separate because ``zarr.Group`` handles are not safely picklable across
    process boundaries — when a ``GroupReader`` is sent to a DataLoader worker
    the handle is stripped and re-opened fresh inside the worker on first use.

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
        _remap: np.ndarray | None = None,
        _var_df: pl.DataFrame | None = None,
        zarr_group_handle: zarr.Group | None = None,
    ) -> None:
        self.zarr_group = zarr_group
        self.feature_space = feature_space
        self._store = store
        self._dataset_vars_table = dataset_vars_table
        self._dataset_uid = dataset_uid
        self._remap = _remap
        self._var_df = _var_df
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
        ``dataset_vars_table`` may be ``None`` for feature
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
        init time). The zarr group handle is ``None`` until first use.
        """
        return cls(
            zarr_group=zarr_group,
            feature_space=feature_space,
            store=store,
            dataset_vars_table=None,
            dataset_uid=None,
            _remap=remap,
        )

    def get_remap(self) -> np.ndarray:
        """Return the local-to-global-index remap array (load-once)."""
        if self._remap is not None:
            return self._remap
        if self._dataset_vars_table is None or self._dataset_uid is None:
            raise ValueError(
                f"GroupReader for {self.zarr_group!r} has no remap and no table to load from."
            )
        rows = read_dataset_vars(self._dataset_vars_table, self._dataset_uid)
        self._remap = rows["global_index"].to_numpy().astype(np.int32, copy=False)
        return self._remap

    @property
    def var_df(self) -> pl.DataFrame:
        """Load and cache var_df for this zarr group (load-once).

        Returns a DataFrame with columns ``global_feature_uid``, ``csc_start``,
        ``csc_end`` in local feature order (row i = local feature i).
        """
        if self._var_df is not None:
            return self._var_df
        if self._dataset_vars_table is None or self._dataset_uid is None:
            return pl.DataFrame(
                schema={
                    "global_feature_uid": pl.Utf8,
                    "csc_start": pl.Int64,
                    "csc_end": pl.Int64,
                }
            )
        rows = read_dataset_vars(self._dataset_vars_table, self._dataset_uid)
        self._var_df = rows.select(
            [
                pl.col("feature_uid").alias("global_feature_uid"),
                pl.col("csc_start"),
                pl.col("csc_end"),
            ]
        )
        return self._var_df

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
