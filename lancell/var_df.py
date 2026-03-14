"""Per-dataset var_df sidecar: read, write, validate, and remap utilities.

Convention: for a zarr group at ``{zarr_group}/``, the sidecar lives at
``{zarr_group}/var.parquet`` and the compiled remap at
``{zarr_group}/local_to_global_index.parquet``.

The var_df is a parquet file with one row per local feature.  Row *i*
corresponds to local feature index *i* in the dataset's zarr arrays.
"""

import io

import duckdb
import lancedb
import numpy as np
import obstore
import polars as pl
import zarr
from pydantic import BaseModel

from lancell.group_specs import ZarrGroupSpec

VAR_DF_FILENAME = "var.parquet"
REMAP_FILENAME = "local_to_global_index.parquet"


# ---------------------------------------------------------------------------
# Column schema
# ---------------------------------------------------------------------------


class VarDfColumnSchema(BaseModel):
    """Declares the required columns of a var_df sidecar.

    Subclass to add modality-specific required or optional columns::

        class GeneVarDfSchema(VarDfColumnSchema):
            gene_name: str
            ensembl_gene_id: str | None = None

    Fields without a default are *required* — the DataFrame must contain a
    column with that name.  Fields with a default (typically ``None``) are
    *optional* — validated only if present.
    """

    global_feature_uid: str
    csc_start: int | None = None  # offset into csc/indices where this feature's cells begin
    csc_end: int | None = None    # exclusive end offset (populated by add_csc)

    @classmethod
    def required_columns(cls) -> set[str]:
        """Column names that must be present (fields without defaults)."""
        return {
            name
            for name, field in cls.model_fields.items()
            if field.is_required()
        }

    @classmethod
    def validate_df(cls, df: pl.DataFrame) -> list[str]:
        """Check that *df* satisfies this schema's column requirements.

        Returns a list of error strings (empty means valid).
        """
        missing = cls.required_columns() - set(df.columns)
        if missing:
            return [f"Missing required columns: {missing}"]
        return []


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def var_df_path(zarr_group: str) -> str:
    return f"{zarr_group.rstrip('/')}/{VAR_DF_FILENAME}"


def remap_path(zarr_group: str) -> str:
    return f"{zarr_group.rstrip('/')}/{REMAP_FILENAME}"


def has_csc(var_df: pl.DataFrame) -> bool:
    """Return True if *var_df* has fully populated ``csc_start`` and ``csc_end`` columns."""
    return (
        "csc_start" in var_df.columns
        and "csc_end" in var_df.columns
        and var_df["csc_start"].null_count() == 0
        and var_df["csc_end"].null_count() == 0
    )


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_var_df(
    store: obstore.store.ObjectStore,
    zarr_group: str,
    df: pl.DataFrame,
    *,
    schema: type[VarDfColumnSchema] = VarDfColumnSchema,
) -> None:
    """Write a var_df sidecar to ``{zarr_group}/var.parquet``.

    Parameters
    ----------
    store:
        An obstore ObjectStore (S3, GCS, local, etc.).
    zarr_group:
        Zarr group path prefix (the store's key namespace).
    df:
        One row per local feature, in local feature order.
        Must contain at least ``global_feature_uid``.
    schema:
        Column schema to validate against.  Defaults to the base
        :class:`VarDfColumnSchema`.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    errors = schema.validate_df(df)
    if errors:
        raise ValueError(f"var_df is missing required columns: {schema.required_columns() - set(df.columns)}")

    buf = io.BytesIO()
    df.write_parquet(buf)
    obstore.put(store, var_df_path(zarr_group), buf.getvalue())


def write_remap(
    store: obstore.store.ObjectStore,
    group: zarr.Group,
    remap: np.ndarray,
    *,
    registry_version: int,
) -> None:
    """Write a compiled ``local_to_global_index`` array as a single-column parquet.

    Also stores ``remap_registry_version`` in the group's attrs so readers can
    detect stale remaps.

    Parameters
    ----------
    store:
        An obstore ObjectStore.
    group:
        The dataset zarr group.  The parquet file is written next to it and the
        registry version is stored in ``group.attrs``.
    remap:
        1-D int32/int64 array where ``remap[i]`` is the global_index of local
        feature *i*.
    registry_version:
        The LanceDB table version of the registry used to build this remap.
        Stored in the group's attrs for freshness checks.
    """
    remap = np.asarray(remap)
    if remap.ndim != 1:
        raise ValueError(f"remap must be 1-D, got ndim={remap.ndim}")
    zarr_group = group.store_path.path
    df = pl.DataFrame({"global_index": remap})
    buf = io.BytesIO()
    df.write_parquet(buf)
    obstore.put(store, remap_path(zarr_group), buf.getvalue())
    group.attrs["remap_registry_version"] = registry_version


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_var_df(
    store: obstore.store.ObjectStore,
    zarr_group: str,
) -> pl.DataFrame:
    """Read a var_df sidecar from ``{zarr_group}/var.parquet``."""
    data = obstore.get(store, var_df_path(zarr_group)).bytes()
    return pl.read_parquet(io.BytesIO(data))


def read_remap(
    store: obstore.store.ObjectStore,
    zarr_group: str,
) -> np.ndarray:
    """Read the compiled ``local_to_global_index`` and return as int32 numpy array."""
    data = obstore.get(store, remap_path(zarr_group)).bytes()
    df = pl.read_parquet(io.BytesIO(data))
    return df["global_index"].to_numpy().astype(np.int32, copy=False)


def read_remap_if_fresh(
    store: obstore.store.ObjectStore,
    group: zarr.Group,
    current_registry_version: int,
) -> np.ndarray | None:
    """Read stored remap if its registry version matches *current_registry_version*.

    Returns ``None`` if the stored version is missing, stale, or the remap
    file cannot be read.
    """
    stored_version = group.attrs.get("remap_registry_version")
    if stored_version is None or stored_version != current_registry_version:
        return None
    try:
        return read_remap(store, group.store_path.path)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build remap from var_df + registry
# ---------------------------------------------------------------------------


def build_remap(
    var_df: pl.DataFrame,
    registry_table: lancedb.table.Table,
) -> np.ndarray:
    """Resolve ``global_feature_uid`` in *var_df* against a feature registry.

    Parameters
    ----------
    var_df:
        The dataset-local var_df.  Must have ``global_feature_uid``.
    registry_table:
        A LanceDB table whose rows have ``uid`` and ``global_index`` columns
        (i.e. a table backed by a :class:`FeatureBaseSchema` subclass).

    Returns
    -------
    numpy.ndarray
        int32 array of length ``len(var_df)`` where element *i* is the
        ``global_index`` for local feature *i*.

    Raises
    ------
    ValueError
        If any uid in *var_df* is not found in the registry.
    """
    uids = var_df["global_feature_uid"].to_list()
    registry_df = (
        registry_table.search()
        .select(["uid", "global_index"])
        .to_polars()
    )
    # Filter out rows where global_index hasn't been assigned yet
    indexed_df = registry_df.filter(pl.col("global_index").is_not_null())
    uid_to_idx = dict(
        zip(indexed_df["uid"].to_list(), indexed_df["global_index"].to_list(), strict=False)
    )
    remap = np.empty(len(uids), dtype=np.int32)
    missing: list[str] = []
    unindexed: list[str] = []
    registry_uids = set(registry_df["uid"].to_list())
    for i, uid in enumerate(uids):
        idx = uid_to_idx.get(uid)
        if idx is not None:
            remap[i] = idx
        elif uid in registry_uids:
            unindexed.append(uid)
        else:
            missing.append(uid)

    if missing:
        raise ValueError(
            f"{len(missing)} uid(s) in var_df not found in registry. "
            f"First 5: {missing[:5]}"
        )
    if unindexed:
        raise ValueError(
            f"{len(unindexed)} uid(s) in registry have no global_index "
            f"(run reindex_registry first). First 5: {unindexed[:5]}"
        )
    return remap


# ---------------------------------------------------------------------------
# Feature UID resolution
# ---------------------------------------------------------------------------


def resolve_feature_uids_to_global_indices(
    registry_table: lancedb.table.Table,
    feature_uids: list[str],
) -> np.ndarray:
    """Resolve feature UIDs to sorted global indices.

    Parameters
    ----------
    registry_table:
        A LanceDB table with ``uid`` and ``global_index`` columns.
    feature_uids:
        List of feature UIDs to resolve.

    Returns
    -------
    numpy.ndarray
        Sorted int32 array of global indices.

    Raises
    ------
    ValueError
        If any UID is missing from the registry, or has ``global_index = None``.
    """
    if not feature_uids:
        return np.array([], dtype=np.int32)

    registry_df = (
        registry_table.search()
        .select(["uid", "global_index"])
        .to_polars()
    )
    requested = set(feature_uids)
    registry_uids = set(registry_df["uid"].to_list())

    missing = requested - registry_uids
    if missing:
        raise ValueError(
            f"{len(missing)} UID(s) not found in registry. "
            f"First 5: {sorted(missing)[:5]}"
        )

    matched = registry_df.filter(pl.col("uid").is_in(list(requested)))
    unindexed = matched.filter(pl.col("global_index").is_null())["uid"].to_list()
    if unindexed:
        raise ValueError(
            f"{len(unindexed)} UID(s) have global_index = None "
            f"(run reindex_registry first). First 5: {unindexed[:5]}"
        )

    return matched["global_index"].to_numpy().astype(np.int32, copy=False)[
        np.argsort(matched["global_index"].to_numpy())
    ]


# ---------------------------------------------------------------------------
# Feature search across datasets
# ---------------------------------------------------------------------------


def _store_uri_prefix(store: obstore.store.ObjectStore) -> str | None:
    """Return a URI prefix that DuckDB can use to read files from *store*.

    Returns ``None`` for store types DuckDB cannot reach directly (e.g.
    MemoryStore), in which case callers should fall back to reading via
    obstore.
    """
    name = type(store).__name__
    if name == "LocalStore":
        return str(store.prefix)
    if name == "S3Store":
        bucket = store.config.get("bucket", "")
        prefix = store.prefix
        base = f"s3://{bucket}"
        return f"{base}/{prefix}".rstrip("/") if prefix else base
    if name == "GCSStore":
        bucket = store.config.get("bucket", "")
        prefix = store.prefix
        base = f"gs://{bucket}"
        return f"{base}/{prefix}".rstrip("/") if prefix else base
    return None


def find_datasets_with_features(
    store: obstore.store.ObjectStore,
    dataset_table: lancedb.table.Table,
    feature_uids: str | list[str],
    feature_space: str,
) -> pl.DataFrame:
    """Find datasets that measured specific features.

    Uses DuckDB to query var_df parquet sidecars directly for one or more
    ``global_feature_uid`` values within a given feature space.  When the
    store is local or cloud-backed (S3/GCS), DuckDB reads the parquet files
    natively with predicate pushdown.  For other stores (e.g. MemoryStore)
    it falls back to reading via obstore.

    Parameters
    ----------
    store:
        An obstore ObjectStore for reading var_df sidecars.
    dataset_table:
        The atlas dataset table (rows are :class:`~lancell.schema.DatasetRecord`).
    feature_uids:
        One or more ``global_feature_uid`` values to search for.
    feature_space:
        Which feature space to search within (e.g. ``"gene_expression"``).

    Returns
    -------
    polars.DataFrame
        One row per (zarr_group, feature) match with columns
        ``zarr_group``, ``global_feature_uid``, plus all columns from the
        dataset table.
    """
    if isinstance(feature_uids, str):
        feature_uids = [feature_uids]

    datasets_df = dataset_table.search().to_polars()
    datasets_df = datasets_df.filter(pl.col("feature_space") == feature_space)
    if datasets_df.is_empty():
        return pl.DataFrame(schema={"zarr_group": pl.Utf8, "global_feature_uid": pl.Utf8})

    zarr_groups = datasets_df["zarr_group"].unique().to_list()
    feature_list = pl.DataFrame({"uid": feature_uids})

    uri_prefix = _store_uri_prefix(store)
    if uri_prefix is not None:
        matches = _query_var_dfs_direct(uri_prefix, zarr_groups, feature_list)
    else:
        matches = _query_var_dfs_via_obstore(store, zarr_groups, feature_list)

    if matches.is_empty():
        return matches

    return matches.join(datasets_df, on="zarr_group", how="left")


def _query_var_dfs_direct(
    uri_prefix: str,
    zarr_groups: list[str],
    feature_list: pl.DataFrame,
) -> pl.DataFrame:
    """Query var_df sidecars directly via DuckDB ``read_parquet``.

    DuckDB reads the parquet files natively, applying predicate pushdown on
    ``global_feature_uid``.  The ``filename=true`` option tags each row with
    its source file, which we join against a path→zarr_group lookup table
    to recover dataset identity.
    """
    paths = [f"{uri_prefix}/{var_df_path(zg)}" for zg in zarr_groups]
    path_map = pl.DataFrame({"filepath": paths, "zarr_group": zarr_groups})
    paths_sql = "[" + ", ".join(f"'{p}'" for p in paths) + "]"

    return duckdb.sql(f"""
        SELECT DISTINCT m.zarr_group, v.global_feature_uid
        FROM read_parquet({paths_sql}, filename=true) v
        JOIN path_map m ON v.filename = m.filepath
        SEMI JOIN feature_list f ON v.global_feature_uid = f.uid
    """).pl()


def _query_var_dfs_via_obstore(
    store: obstore.store.ObjectStore,
    zarr_groups: list[str],
    feature_list: pl.DataFrame,
) -> pl.DataFrame:
    """Fallback: read sidecars via obstore, then query with DuckDB."""
    var_dfs: list[pl.DataFrame] = []
    for zg in zarr_groups:
        vdf = read_var_df(store, zg)
        vdf = vdf.select("global_feature_uid").with_columns(
            pl.lit(zg).alias("zarr_group")
        )
        var_dfs.append(vdf)

    all_var = pl.concat(var_dfs)
    return duckdb.sql("""
        SELECT DISTINCT v.zarr_group, v.global_feature_uid
        FROM all_var v
        SEMI JOIN feature_list f ON v.global_feature_uid = f.uid
    """).pl()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _get_local_feature_count(
    group: zarr.Group,
    spec: ZarrGroupSpec,
) -> int | None:
    """Derive the expected number of local features from the zarr group.

    For sparse feature spaces the feature count is not directly stored in the
    zarr arrays (the arrays hold ragged per-cell data).  In that case we
    return None to signal that the count cannot be validated from the arrays
    alone — the caller should supply it from external metadata (e.g. the
    max index value + 1).

    For dense feature spaces the second axis of the ``data`` array gives the
    feature count.
    """
    from lancell.group_specs import PointerKind

    if spec.pointer_kind is PointerKind.DENSE:
        if "layers" in group and isinstance(group["layers"], zarr.Group):
            for _, arr in group["layers"].arrays():
                return arr.shape[1]
        if "data" in group and isinstance(group["data"], zarr.Array):
            return group["data"].shape[1]
    return None


def validate_var_df(
    var_df: pl.DataFrame,
    *,
    spec: ZarrGroupSpec,
    group: zarr.Group | None = None,
    expected_feature_count: int | None = None,
    registry_table: lancedb.table.Table | None = None,
    schema: type[VarDfColumnSchema] = VarDfColumnSchema,
) -> list[str]:
    """Validate a var_df sidecar against its zarr group and/or feature registry.

    Parameters
    ----------
    var_df:
        The sidecar DataFrame to validate.
    spec:
        The ``ZarrGroupSpec`` for the feature space.
    group:
        Optional zarr group.  Used to cross-check row count for dense spaces.
    expected_feature_count:
        Explicit expected feature count.  For sparse feature spaces the count
        is not derivable from the zarr arrays, so pass it here (e.g.
        ``indices.max() + 1`` from the full dataset).  Overrides the zarr
        group-derived count if both are provided.
    registry_table:
        Optional LanceDB table.  If provided, validates that every
        ``global_feature_uid`` resolves and that ``global_index`` values
        (if present in the var_df) agree with the registry.
    schema:
        Column schema to validate against.  Defaults to the base
        :class:`VarDfColumnSchema`.

    Returns
    -------
    list[str]
        List of validation error messages.  Empty means valid.
    """
    errors: list[str] = []

    # Required columns (from schema)
    schema_errors = schema.validate_df(var_df)
    if schema_errors:
        errors.extend(schema_errors)
        return errors  # can't do further checks

    # Row count vs zarr group
    n_local = expected_feature_count
    if n_local is None and group is not None:
        n_local = _get_local_feature_count(group, spec)
    if n_local is not None and len(var_df) != n_local:
        errors.append(
            f"var_df has {len(var_df)} rows but expected {n_local} features"
        )

    # Null uids
    null_count = var_df["global_feature_uid"].null_count()
    if null_count > 0:
        errors.append(f"global_feature_uid has {null_count} null(s)")

    # Duplicate uids
    n_unique = var_df["global_feature_uid"].n_unique()
    if n_unique != len(var_df):
        errors.append(
            f"global_feature_uid has duplicates: {len(var_df)} rows but "
            f"{n_unique} unique values"
        )

    # Registry checks
    if registry_table is not None:
        registry_df = (
            registry_table.search()
            .select(["uid", "global_index"])
            .to_polars()
        )
        registry_uids = set(registry_df["uid"].to_list())
        uid_to_idx = dict(
            zip(
                registry_df["uid"].to_list(),
                registry_df["global_index"].to_list(),
                strict=False,
            )
        )
        var_uids = var_df["global_feature_uid"].to_list()

        # Check all uids resolve
        unresolved = [u for u in var_uids if u not in registry_uids]
        if unresolved:
            errors.append(
                f"{len(unresolved)} uid(s) not found in registry. "
                f"First 5: {unresolved[:5]}"
            )

        # If var_df carries global_index, check agreement
        if "global_index" in var_df.columns and not unresolved:
            var_indices = var_df["global_index"].to_list()
            mismatches: list[str] = []
            for i, (uid, local_gi) in enumerate(zip(var_uids, var_indices, strict=True)):
                expected_gi = uid_to_idx.get(uid)
                if expected_gi is not None and local_gi != expected_gi:
                    mismatches.append(
                        f"row {i}: uid={uid} has global_index={local_gi} "
                        f"but registry has {expected_gi}"
                    )
            if mismatches:
                errors.append(
                    f"global_index mismatch for {len(mismatches)} row(s). "
                    f"First 3: {mismatches[:3]}"
                )

    return errors


# ---------------------------------------------------------------------------
# Registry reindexing
# ---------------------------------------------------------------------------


def reindex_registry(
    table: lancedb.table.Table,
    *,
    sort_by: str = "uid",
) -> int:
    """Assign contiguous ``global_index`` values to every row in a feature registry.

    Reads all rows, sorts deterministically by *sort_by*, assigns
    ``global_index = 0 .. N-1``, and writes the updated indices back via
    ``merge_insert`` on ``uid``.

    This is designed to be coupled with periodic ``table.optimize()`` and
    ``table.create_scalar_index()`` calls.

    Parameters
    ----------
    table:
        A LanceDB table backed by a :class:`FeatureBaseSchema` subclass.
    sort_by:
        Column to sort by before assigning indices.  Deterministic sort
        ensures the same input always produces the same index assignment.

    Returns
    -------
    int
        Number of features indexed.
    """
    columns = ["uid", "global_index"]
    if sort_by not in columns:
        columns.append(sort_by)
    df = table.search().select(columns).to_polars()
    if df.is_empty():
        return 0

    # Skip if already contiguous 0..N-1 with no nulls and in canonical order
    df = df.sort(sort_by)
    indices = df["global_index"]
    if (
        indices.null_count() == 0
        and int(indices[0]) == 0
        and int(indices.diff().drop_nulls().max()) <= 1
    ):
        return 0
    df = df.with_columns(pl.Series("global_index", range(len(df)), dtype=pl.Int64))

    (
        table.merge_insert(on="uid")
        .when_matched_update_all()
        .execute(df)
    )
    return len(df)
