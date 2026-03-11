"""Per-dataset var_df sidecar: read, write, validate, and remap utilities.

Convention: for a zarr group at ``{zarr_group}/``, the sidecar lives at
``{zarr_group}/var.parquet`` and the compiled remap at
``{zarr_group}/local_to_global_index.parquet``.

The var_df is a parquet file with one row per local feature.  Row *i*
corresponds to local feature index *i* in the dataset's zarr arrays.
"""

import io
from typing import TYPE_CHECKING

import numpy as np
import obstore
import polars as pl
import zarr
from pydantic import BaseModel

if TYPE_CHECKING:
    import lancedb

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
    zarr_group: str,
    remap: np.ndarray,
) -> None:
    """Write a compiled ``local_to_global_index`` array as a single-column parquet.

    Parameters
    ----------
    store:
        An obstore ObjectStore.
    zarr_group:
        Zarr group path prefix.
    remap:
        1-D int32/int64 array where ``remap[i]`` is the global_index of local
        feature *i*.
    """
    remap = np.asarray(remap)
    if remap.ndim != 1:
        raise ValueError(f"remap must be 1-D, got ndim={remap.ndim}")
    df = pl.DataFrame({"global_index": remap})
    buf = io.BytesIO()
    df.write_parquet(buf)
    obstore.put(store, remap_path(zarr_group), buf.getvalue())


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


def validate_remap(
    remap: np.ndarray,
    *,
    var_df: pl.DataFrame | None = None,
    registry_table: lancedb.table.Table | None = None,
) -> list[str]:
    """Validate a compiled ``local_to_global_index`` array.

    Parameters
    ----------
    remap:
        The 1-D integer array to validate.
    var_df:
        If provided, checks that remap length matches var_df length.
    registry_table:
        If provided, checks that all remap values are valid global_index
        values in the registry.

    Returns
    -------
    list[str]
        List of validation error messages.  Empty means valid.
    """
    errors: list[str] = []

    if remap.ndim != 1:
        errors.append(f"remap must be 1-D, got ndim={remap.ndim}")
        return errors

    if var_df is not None and len(remap) != len(var_df):
        errors.append(
            f"remap length ({len(remap)}) != var_df length ({len(var_df)})"
        )

    if registry_table is not None:
        registry_df = (
            registry_table.search()
            .select(["global_index"])
            .to_polars()
        )
        valid_indices = set(registry_df["global_index"].to_list())
        remap_set = set(remap.tolist())
        invalid = remap_set - valid_indices
        if invalid:
            errors.append(
                f"{len(invalid)} remap value(s) not in registry. "
                f"First 5: {sorted(invalid)[:5]}"
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
    df = table.search().to_polars()
    if df.is_empty():
        return 0

    df = df.sort(sort_by)
    df = df.with_columns(pl.Series("global_index", range(len(df)), dtype=pl.Int64))

    (
        table.merge_insert(on="uid")
        .when_matched_update_all()
        .execute(df)
    )
    return len(df)
