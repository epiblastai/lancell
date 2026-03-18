"""Reference ingestion functions for writing AnnData into a RaggedAtlas.

These are extracted from the original ``RaggedAtlas`` write path and serve as a
reference implementation.  Downstream projects can write their own ingestion
that calls the lower-level ``var_df`` helpers directly.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import polars as pl
import pyarrow as pa
import scipy.sparse as sp
import zarr

from lancell.atlas import RaggedAtlas
from lancell.feature_layouts import read_feature_layout
from lancell.group_specs import PointerKind, get_spec
from lancell.obs_alignment import PointerFieldInfo, _schema_obs_fields, validate_obs_columns
from lancell.schema import (
    DatasetRecord,
    make_uid,
)
from lancell.util import sql_escape

_INTEGER_DTYPES = {np.dtype("int32"), np.dtype("int64"), np.dtype("uint32"), np.dtype("uint64")}

_CHUNK_ELEMS = 40_960
_CHUNKS_PER_SHARD = 1024
_SHARD_ELEMS = _CHUNKS_PER_SHARD * _CHUNK_ELEMS


def _is_backed_csr(adata: ad.AnnData) -> bool:
    """Return True if adata.X is a backed HDF5 CSR matrix (h5ad format)."""
    return adata.isbacked and "X" in adata.file._file and "data" in adata.file._file["X"]


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

    from lancell.codecs.bitpacking import BitpackingCodec

    indices_kwargs: dict = {"compressors": BitpackingCodec(transform="delta")}
    layer_kwargs: dict = {}
    if use_bitpacking:
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
        "indices",
        shape=(nnz,),
        dtype=np.uint32,
        chunks=chunk_shape,
        shards=shard_shape,
        **indices_kwargs,
    )
    layers = csr_group.create_group("layers")
    zarr_values = layers.create_array(
        zarr_layer,
        shape=(nnz,),
        dtype=data_dtype,
        chunks=chunk_shape,
        shards=shard_shape,
        **layer_kwargs,
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
    data_dtype = adata.X.dtype

    if zarr_layer is not None:
        layers_group = group.create_group("layers")
        zarr_arr = layers_group.create_array(
            zarr_layer,
            shape=(n_cells, n_vars),
            dtype=data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
        )
    else:
        zarr_arr = group.create_array(
            "data",
            shape=(n_cells, n_vars),
            dtype=data_dtype,
            chunks=chunk_shape,
            shards=shard_shape,
        )

    written = 0
    while written < n_cells:
        end = min(written + batch_size, n_cells)
        zarr_arr[written:end] = np.asarray(adata.X[written:end], dtype=data_dtype)
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
        raise ValueError(f"obs columns do not match cell schema: {obs_errors}")

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
            group,
            adata,
            zarr_layer,
            chunk_shape,
            shard_shape,
            use_bitpacking,
        )
    else:
        _write_dense_batched(group, adata, zarr_layer, chunk_shape, shard_shape)

    if spec.has_var_df:
        write_feature_layout(atlas, adata, feature_space, zarr_group, dataset_record.uid)

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
        atlas,
        adata,
        feature_space=feature_space,
        zarr_layer=zarr_layer,
        dataset_record=dataset_record,
        chunk_shape=chunk_shape,
        shard_shape=shard_shape,
    )


def write_feature_layout(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    feature_space: str,
    zarr_group: str,
    dataset_uid: str,
) -> None:
    """Write feature layout for a dataset into the _feature_layouts Lance table.

    Requires ``global_feature_uid`` in ``adata.var`` and features to
    already be registered via :meth:`RaggedAtlas.register_features`.
    """
    var_df = pl.from_pandas(adata.var.reset_index())
    if "global_feature_uid" not in var_df.columns:
        raise ValueError(
            "adata.var must have a 'global_feature_uid' column. "
            "Set it before calling add_anndata_batch()."
        )

    atlas.add_or_reuse_layout(var_df, dataset_uid, feature_space)


def add_csc(
    atlas: RaggedAtlas,
    zarr_group: str,
    feature_space: str,
    layer_name: str = "counts",
    chunk_size: int = 4096,
    shard_size: int = 65536,
) -> None:
    """Read existing CSR group and write CSC alongside it.

    Reads the full CSR flat arrays from ``{zarr_group}/csr/``, transposes
    to CSC order sorted by feature index, writes ``{zarr_group}/csc/``, and
    stores the CSC ``indptr`` as a zarr array at ``{zarr_group}/csc/indptr``.

    After running, a new ``{zarr_group}/csc/`` subgroup appears alongside the
    existing ``{zarr_group}/csr/``, including an ``indptr`` array. Subsequent
    feature-filtered queries will automatically use the CSC path.

    Parameters
    ----------
    atlas:
        The atlas whose zarr store and cell table to use.
    zarr_group:
        Path of the zarr group to process (relative to atlas store root).
    feature_space:
        The feature space this zarr group belongs to.
    layer_name:
        Which layer to transpose (e.g. ``"counts"``).
    chunk_size:
        Chunk size for the new CSC zarr arrays.
    shard_size:
        Shard size for the new CSC zarr arrays.

    Raises
    ------
    ValueError
        If no cells or no dataset record are found for this group, or if
        ``zarr_row`` is not sequential.
    """
    # Look up dataset_uid and layout_uid for this zarr_group + feature_space
    datasets_df = (
        atlas._dataset_table.search()
        .where(
            f"zarr_group = '{sql_escape(zarr_group)}' AND feature_space = '{sql_escape(feature_space)}'",
            prefilter=True,
        )
        .select(["uid", "layout_uid"])
        .to_polars()
    )
    if datasets_df.is_empty():
        raise ValueError(
            f"No dataset record found for zarr_group='{zarr_group}', "
            f"feature_space='{feature_space}'"
        )
    layout_uid = datasets_df["layout_uid"][0]

    # Query all cells in this zarr group
    cells_df = (
        atlas.cell_table.search()
        .where(f"{feature_space}.zarr_group = '{sql_escape(zarr_group)}'", prefilter=True)
        .select([feature_space])
        .to_polars()
    )
    ptr_struct = cells_df[feature_space].struct.unnest()
    cells_df = pl.DataFrame(
        {
            "_zg": ptr_struct["zarr_group"],
            "_zarr_row": ptr_struct["zarr_row"],
            "_start": ptr_struct["start"],
            "_end": ptr_struct["end"],
        }
    )

    if cells_df.is_empty():
        raise ValueError(
            f"No cells found for zarr_group='{zarr_group}', feature_space='{feature_space}'"
        )

    cells_df = cells_df.sort("_zarr_row")
    zarr_rows = cells_df["_zarr_row"].to_numpy()
    starts = cells_df["_start"].to_numpy()
    ends = cells_df["_end"].to_numpy()
    n_cells = len(zarr_rows)

    if len(zarr_rows) != len(np.unique(zarr_rows)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' contain duplicate values. "
            f"Was zarr_row populated correctly during ingest?"
        )
    if not np.array_equal(zarr_rows, np.arange(n_cells)):
        raise ValueError(
            f"zarr_rows for group '{zarr_group}' are not sequential 0..{n_cells - 1}. "
            f"Was zarr_row populated correctly during ingest?"
        )

    # Get n_features from _feature_layouts
    rows = read_feature_layout(atlas._feature_layouts_table, layout_uid)
    n_features = len(rows)

    _add_csc_scipy(
        atlas,
        zarr_group,
        layer_name,
        starts,
        ends,
        n_cells,
        n_features,
        chunk_size,
        shard_size,
        feature_space,
    )


def _add_csc_scipy(
    atlas: RaggedAtlas,
    zarr_group: str,
    layer_name: str,
    starts: np.ndarray,
    ends: np.ndarray,
    n_cells: int,
    n_features: int,
    chunk_size: int,
    shard_size: int,
    feature_space: str,
) -> None:
    """CSR-to-CSC using scipy (fast, but loads full matrix into RAM)."""
    root = zarr.open_group(zarr.storage.ObjectStore(atlas._store), mode="r")
    csr_indices = root[f"{zarr_group}/csr/indices"][:]
    csr_values = root[f"{zarr_group}/csr/layers/{layer_name}"][:]

    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0
    indptr[1:] = ends
    # starts/ends are absolute offsets; indptr needs to be relative from 0
    # but since starts[0]==0 and ends are cumulative, we can just use them directly
    indptr_csr = np.concatenate([[starts[0]], ends])

    csr = sp.csr_matrix(
        (csr_values, csr_indices.astype(np.int32), indptr_csr),
        shape=(n_cells, n_features),
    )
    csc = csr.tocsc()

    nnz = csc.nnz
    _writable_root = zarr.open_group(zarr.storage.ObjectStore(atlas._store), mode="a")
    csc_group = _writable_root.require_group(f"{zarr_group}/csc")

    csc_indices_zarr = csc_group.create_array(
        "indices",
        shape=(nnz,),
        dtype=np.uint32,
        chunks=(chunk_size,),
        shards=(shard_size,),
    )
    layers_group = csc_group.create_group("layers")
    csc_values_zarr = layers_group.create_array(
        layer_name,
        shape=(nnz,),
        dtype=np.uint32,
        chunks=(chunk_size,),
        shards=(shard_size,),
    )

    # Write in shard-sized batches
    written = 0
    while written < nnz:
        end = min(written + shard_size, nnz)
        csc_indices_zarr[written:end] = csc.indices[written:end].astype(np.uint32)
        csc_values_zarr[written:end] = csc.data[written:end].astype(np.uint32)
        written = end

    csc_group.create_array("indptr", data=csc.indptr.astype(np.int64))

    # Cache invalidation
    atlas._group_readers.pop((zarr_group, feature_space), None)
