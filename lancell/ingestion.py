"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pyarrow as pa
import polars as pl
import scipy.sparse as sp
import zarr

from lancell.atlas import (
    RaggedAtlas,
    PointerFieldInfo,
    _schema_obs_fields,
    validate_obs_columns,
)
from lancell.group_specs import PointerKind, get_spec
from lancell.schema import (
    DatasetRecord,
    make_uid,
)
from lancell.var_df import build_remap, write_remap, write_var_df


_INTEGER_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("uint32"), np.dtype("uint64")}

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024
_SHARD_ELEMS = _CHUNKS_PER_SHARD * _CHUNK_ELEMS


def _is_backed_csr(adata: ad.AnnData) -> bool:
    """Return True if adata.X is a backed HDF5 CSR matrix (h5ad format)."""
    return (
        adata.isbacked
        and "X" in adata.file._file
        and "data" in adata.file._file["X"]
    )


def _write_sparse_batched(
    group: zarr.Group,
    adata: ad.AnnData,
    zarr_layer: str,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
    use_bitpacking: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-allocate and stream-write CSR data in shard-sized batches.

    For backed h5ad files, reads directly from the HDF5 CSR datasets without
    materialising the full matrix.  For in-memory AnnData, converts to scipy
    CSR first, then streams the flat arrays.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(starts, ends)`` — per-cell indptr start/end positions.
    """
    batch_size = shard_shape[0]

    indices_kwargs: dict = {}
    layer_kwargs: dict = {}
    if use_bitpacking:
        from lancell.codecs.bitpacking import BitpackingCodec

        indices_kwargs["compressors"] = BitpackingCodec(transform="delta")
        layer_kwargs["compressors"] = BitpackingCodec(transform="none")

    if _is_backed_csr(adata):
        h5x = adata.file._file["X"]
        nnz = int(h5x["data"].shape[0])
        indptr = h5x["indptr"][:]
        src_indices = h5x["indices"]
        src_data = h5x["data"]
        data_dtype = src_data.dtype
    else:
        csr = adata.X if isinstance(adata.X, sp.csr_matrix) else sp.csr_matrix(adata.X)
        nnz = csr.nnz
        indptr = csr.indptr
        src_indices = csr.indices
        src_data = csr.data
        data_dtype = csr.data.dtype

    csr_group = group.create_group("csr")
    zarr_indices = csr_group.create_array(
        "indices", shape=(nnz,), dtype=np.uint32,
        chunks=chunk_shape, shards=shard_shape, **indices_kwargs,
    )
    layers = csr_group.create_group("layers")
    zarr_values = layers.create_array(
        zarr_layer, shape=(nnz,), dtype=data_dtype,
        chunks=chunk_shape, shards=shard_shape, **layer_kwargs,
    )

    written = 0
    while written < nnz:
        end = min(written + batch_size, nnz)
        zarr_indices[written:end] = src_indices[written:end].astype(np.uint32, copy=False)
        zarr_values[written:end] = src_data[written:end]
        written = end

    starts = indptr[:-1].astype(np.int64)
    ends = indptr[1:].astype(np.int64)
    return starts, ends


def _write_dense_batched(
    group: zarr.Group,
    adata: ad.AnnData,
    zarr_layer: str | None,
    chunk_shape: tuple[int, ...],
    shard_shape: tuple[int, ...],
) -> None:
    """Pre-allocate and stream-write dense 2D data in shard-sized cell batches.

    Slices ``adata.X[start:end, :]`` per batch; anndata handles backed vs
    in-memory transparently for dense arrays.
    """
    n_cells, n_vars = adata.shape
    batch_size = shard_shape[0]

    if zarr_layer is not None:
        layers_group = group.create_group("layers")
        zarr_arr = layers_group.create_array(
            zarr_layer, shape=(n_cells, n_vars), dtype=np.float32,
            chunks=chunk_shape, shards=shard_shape,
        )
    else:
        zarr_arr = group.create_array(
            "data", shape=(n_cells, n_vars), dtype=np.float32,
            chunks=chunk_shape, shards=shard_shape,
        )

    written = 0
    while written < n_cells:
        end = min(written + batch_size, n_cells)
        zarr_arr[written:end] = np.asarray(adata.X[written:end], dtype=np.float32)
        written = end


def add_anndata_batch(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    *,
    feature_space: str,
    zarr_layer: str | None,
    dataset_record: DatasetRecord,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
) -> int:
    """Ingest an AnnData into the atlas using batched zarr writes.

    Always ingests ``adata.X``. ``zarr_layer`` is the *destination* layer name
    within the zarr group (e.g. ``"counts"``), not a source AnnData layer.

    Writes zarr arrays, var_df sidecar, remap, and inserts cell records into
    the cell table. Features must already be registered via
    :meth:`RaggedAtlas.register_features`, and ``adata.var`` must contain a
    ``global_feature_uid`` column.

    Parameters
    ----------
    atlas:
        The atlas to ingest into.
    adata:
        The AnnData to ingest. Use ``backed="r"`` for large files to avoid
        materialising the full matrix; see :func:`add_from_anndata` for a
        convenience wrapper that opens h5ad paths automatically.
    feature_space:
        Which feature space this data belongs to.
    zarr_layer:
        Destination layer name within the zarr CSR ``layers/`` group
        (e.g. ``"counts"``). Required for feature spaces with
        ``allowed_layers``; pass ``None`` for feature spaces without layers.
    dataset_record:
        Dataset record to register. ``dataset_record.zarr_group`` is used as
        the zarr group path (relative to the atlas store). Construct with
        :class:`DatasetRecord` or a subclass for richer metadata.
    chunk_shape:
        Zarr chunk shape. For sparse feature spaces this must be a 1-element
        tuple; for dense a 2-element tuple ``(n_cells_per_chunk, n_features)``.
        Defaults to ``(_CHUNK_ELEMS,)`` for sparse and
        ``(max(1, _CHUNK_ELEMS // n_vars), n_vars)`` for dense.
        Values should be multiples of 128 for optimal BP-128 bitpacking.
    shard_shape:
        Zarr shard shape, same dimensionality rules as ``chunk_shape``.
        Defaults to ``(_SHARD_ELEMS,)`` for sparse and
        ``(max(1, _SHARD_ELEMS // n_vars), n_vars)`` for dense.

    Returns
    -------
    int
        Number of cells ingested.
    """
    spec = get_spec(feature_space)

    if spec.allowed_layers and zarr_layer is None:
        raise ValueError(
            f"zarr_layer is required for feature space '{feature_space}'. "
            f"Allowed values: {spec.allowed_layers}"
        )
    if zarr_layer is not None and spec.allowed_layers and zarr_layer not in spec.allowed_layers:
        raise ValueError(
            f"zarr_layer '{zarr_layer}' is not allowed for feature space "
            f"'{feature_space}'. Allowed: {spec.allowed_layers}"
        )

    obs_errors = validate_obs_columns(adata.obs, atlas._cell_schema)
    if obs_errors:
        raise ValueError(
            f"obs columns do not match cell schema: {obs_errors}"
        )

    pointer_field: PointerFieldInfo | None = None
    for pf in atlas._pointer_fields.values():
        if pf.feature_space == feature_space:
            pointer_field = pf
            break
    if pointer_field is None:
        raise ValueError(
            f"Schema {atlas._cell_schema.__name__} has no pointer field "
            f"for feature space '{feature_space}'"
        )

    n_cells = adata.n_obs
    zarr_group = dataset_record.zarr_group

    if spec.pointer_kind is PointerKind.SPARSE:
        chunk_shape = chunk_shape or (_CHUNK_ELEMS,)
        shard_shape = shard_shape or (_SHARD_ELEMS,)
        if len(chunk_shape) != 1 or len(shard_shape) != 1:
            raise ValueError(
                f"Sparse feature space '{feature_space}' requires 1-element chunk_shape "
                f"and shard_shape, got chunk_shape={chunk_shape}, shard_shape={shard_shape}"
            )
        data_dtype = (
            np.dtype(adata.file._file["X"]["data"].dtype)
            if _is_backed_csr(adata)
            else adata.X.dtype
        )
        use_bitpacking = data_dtype in _INTEGER_DTYPES
    else:
        n_vars = adata.n_vars
        chunk_shape = chunk_shape or (max(1, _CHUNK_ELEMS // n_vars), n_vars)
        shard_shape = shard_shape or (max(1, _SHARD_ELEMS // n_vars), n_vars)
        if len(chunk_shape) != 2 or len(shard_shape) != 2:
            raise ValueError(
                f"Dense feature space '{feature_space}' requires 2-element chunk_shape "
                f"and shard_shape, got chunk_shape={chunk_shape}, shard_shape={shard_shape}"
            )
        use_bitpacking = False

    dataset_arrow = pa.Table.from_pylist(
        [dataset_record.model_dump()],
        schema=type(dataset_record).to_arrow_schema(),
    )
    atlas._dataset_table.add(dataset_arrow)

    group = atlas._root.create_group(zarr_group)
    if spec.pointer_kind is PointerKind.SPARSE:
        starts, ends = _write_sparse_batched(
            group, adata, zarr_layer, chunk_shape, shard_shape, use_bitpacking,
        )
    else:
        _write_dense_batched(group, adata, zarr_layer, chunk_shape, shard_shape)

    if spec.has_var_df:
        write_var_sidecar(atlas, adata, feature_space, zarr_group)

    arrow_schema = atlas._cell_schema.to_arrow_schema()
    obs_df = adata.obs
    schema_fields = _schema_obs_fields(atlas._cell_schema)

    if spec.pointer_kind is PointerKind.SPARSE:
        pointer_struct = pa.StructArray.from_arrays(
            [
                pa.array([feature_space] * n_cells, type=pa.string()),
                pa.array([zarr_group] * n_cells, type=pa.string()),
                pa.array(starts.astype(np.int64), type=pa.int64()),
                pa.array(ends.astype(np.int64), type=pa.int64()),
                pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
            ],
            names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
        )
    else:
        pointer_struct = pa.StructArray.from_arrays(
            [
                pa.array([feature_space] * n_cells, type=pa.string()),
                pa.array([zarr_group] * n_cells, type=pa.string()),
                pa.array(np.arange(n_cells, dtype=np.int64), type=pa.int64()),
            ],
            names=["feature_space", "zarr_group", "position"],
        )

    columns = {
        "uid": pa.array([make_uid() for _ in range(n_cells)], type=pa.string()),
        "dataset_uid": pa.array([dataset_record.uid] * n_cells, type=pa.string()),
        pointer_field.field_name: pointer_struct,
    }

    # Zero-fill any other pointer fields in the schema (multi-modal schemas).
    # Lance requires non-null values for non-null struct fields; an empty zarr_group
    # string ("") signals "absent" and is filtered by _prepare_*_cells.
    for other_pf_name, other_pf in atlas._pointer_fields.items():
        if other_pf_name == pointer_field.field_name:
            continue
        if other_pf.pointer_kind is PointerKind.SPARSE:
            columns[other_pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([0] * n_cells, type=pa.int64()),
                    pa.array([0] * n_cells, type=pa.int64()),
                    pa.array([0] * n_cells, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "start", "end", "zarr_row"],
            )
        else:
            columns[other_pf_name] = pa.StructArray.from_arrays(
                [
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([""] * n_cells, type=pa.string()),
                    pa.array([0] * n_cells, type=pa.int64()),
                ],
                names=["feature_space", "zarr_group", "position"],
            )

    for col in schema_fields:
        if col in obs_df.columns:
            columns[col] = pa.array(obs_df[col].values, type=arrow_schema.field(col).type)

    for col in schema_fields:
        if col not in columns:
            columns[col] = pa.nulls(n_cells, type=arrow_schema.field(col).type)

    arrow_table = pa.table(columns, schema=arrow_schema)
    atlas.cell_table.add(arrow_table)
    return n_cells


def add_from_anndata(
    atlas: RaggedAtlas,
    adata: ad.AnnData | str | Path,
    *,
    feature_space: str,
    zarr_layer: str | None,
    dataset_record: DatasetRecord,
    chunk_shape: tuple[int, ...] | None = None,
    shard_shape: tuple[int, ...] | None = None,
) -> int:
    """Convenience wrapper around :func:`add_anndata_batch`.

    Accepts an in-memory :class:`anndata.AnnData` or a path to an ``.h5ad``
    file.  Paths are opened with ``backed="r"`` so the full matrix is never
    materialised into memory.

    All other parameters are forwarded to :func:`add_anndata_batch`; see that
    function for full documentation.
    """
    if not isinstance(adata, ad.AnnData):
        adata = ad.read_h5ad(adata, backed="r")
    return add_anndata_batch(
        atlas, adata,
        feature_space=feature_space,
        zarr_layer=zarr_layer,
        dataset_record=dataset_record,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )


def write_var_sidecar(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    feature_space: str,
    zarr_group: str,
) -> None:
    """Write var.parquet and version-gated remap.parquet for a dataset.

    Requires ``global_feature_uid`` in ``adata.var`` and features to
    already be registered via :meth:`RaggedAtlas.register_features`.  The remap
    is tagged with the current registry table version so readers can detect
    staleness.
    """
    var_df = pl.from_pandas(adata.var.reset_index())
    if "global_feature_uid" not in var_df.columns:
        raise ValueError(
            "adata.var must have a 'global_feature_uid' column. "
            "Set it before calling add_anndata_batch()."
        )

    write_var_df(atlas._store, zarr_group, var_df)

    if feature_space in atlas._registry_tables:
        registry_table = atlas._registry_tables[feature_space]
        remap = build_remap(var_df, registry_table)
        group = atlas._root[zarr_group]
        write_remap(
            atlas._store, group, remap,
            registry_version=registry_table.version,
        )
