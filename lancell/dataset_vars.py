"""Per-dataset feature mapping: _dataset_vars LanceDB table helpers.

Replaces the sidecar parquet files (var.parquet, local_to_global_index.parquet)
and the _feature_dataset_pairs inverted index table.
"""

from typing import TYPE_CHECKING

import lancedb
import numpy as np
import polars as pl

from lancell._util import sql_escape

if TYPE_CHECKING:
    import zarr

    from lancell.group_specs import ZarrGroupSpec


# ---------------------------------------------------------------------------
# Build / insert
# ---------------------------------------------------------------------------


def build_dataset_vars_df(
    var_df: pl.DataFrame,
    dataset_uid: str,
    registry_table: lancedb.table.Table,
) -> pl.DataFrame:
    """Build a DataFrame ready for inserting into _dataset_vars.

    Parameters
    ----------
    var_df:
        One row per local feature, in local feature order. Must have
        ``global_feature_uid``.
    dataset_uid:
        The DatasetRecord uid for this dataset.
    registry_table:
        Feature registry. Used to resolve global_index for each feature_uid.

    Returns
    -------
    pl.DataFrame
        One row per feature with columns: feature_uid, dataset_uid,
        local_index, global_index, csc_start (null), csc_end (null).

    Raises
    ------
    ValueError
        If any feature_uid is missing from the registry or has no global_index.
    """
    if "global_feature_uid" not in var_df.columns:
        raise ValueError("var_df must have a 'global_feature_uid' column")

    feature_uids = var_df["global_feature_uid"].to_list()
    n = len(feature_uids)

    uids_sql = ", ".join(f"'{sql_escape(u)}'" for u in feature_uids)
    registry_df = (
        registry_table.search()
        .where(f"uid IN ({uids_sql})", prefilter=True)
        .select(["uid", "global_index"])
        .to_polars()
    )
    uid_to_global: dict[str, int] = dict(
        zip(registry_df["uid"].to_list(), registry_df["global_index"].to_list(), strict=True)
    )
    registry_uids = set(registry_df["uid"].to_list())

    global_indices: list[int] = []
    missing: list[str] = []
    unindexed: list[str] = []
    for uid in feature_uids:
        gi = uid_to_global.get(uid)
        if gi is not None:
            global_indices.append(gi)
        elif uid in registry_uids:
            unindexed.append(uid)
        else:
            missing.append(uid)

    if missing:
        raise ValueError(
            f"{len(missing)} uid(s) in var_df not found in registry. First 5: {sorted(missing)[:5]}"
        )
    if unindexed:
        raise ValueError(
            f"{len(unindexed)} uid(s) in registry have no global_index "
            f"(run reindex_registry first). First 5: {unindexed[:5]}"
        )

    return pl.DataFrame(
        {
            "feature_uid": feature_uids,
            "dataset_uid": [dataset_uid] * n,
            "local_index": list(range(n)),
            "global_index": global_indices,
            "csc_start": pl.Series([None] * n, dtype=pl.Int64),
            "csc_end": pl.Series([None] * n, dtype=pl.Int64),
        }
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def read_dataset_vars(
    table: lancedb.table.Table,
    dataset_uid: str,
) -> pl.DataFrame:
    """Read all DatasetVar rows for a dataset, sorted by local_index."""
    return (
        table.search()
        .where(f"dataset_uid = '{sql_escape(dataset_uid)}'", prefilter=True)
        .select(
            ["feature_uid", "dataset_uid", "local_index", "global_index", "csc_start", "csc_end"]
        )
        .to_polars()
        .sort("local_index")
    )


# ---------------------------------------------------------------------------
# Sync global_index after reindex_registry
# ---------------------------------------------------------------------------


def sync_dataset_vars_global_index(
    dataset_vars_table: lancedb.table.Table,
    registry_table: lancedb.table.Table,
) -> int:
    """Propagate updated global_index values from registry to _dataset_vars.

    After reindex_registry(), call this to keep the denormalized global_index
    in _dataset_vars consistent with the registry. Uses merge_insert to update
    matched rows.

    Returns
    -------
    int
        Number of rows updated.
    """
    registry_df = (
        registry_table.search()
        .select(["uid", "global_index"])
        .to_polars()
        .filter(pl.col("global_index").is_not_null())
    )
    if registry_df.is_empty():
        return 0

    all_rows = (
        dataset_vars_table.search()
        .select(
            ["feature_uid", "dataset_uid", "local_index", "global_index", "csc_start", "csc_end"]
        )
        .to_polars()
    )
    if all_rows.is_empty():
        return 0

    # Join to get updated global_index values from the registry
    updated = (
        all_rows.join(
            registry_df.rename({"uid": "feature_uid", "global_index": "new_global_index"}),
            on="feature_uid",
            how="inner",
        )
        .with_columns(pl.col("new_global_index").alias("global_index"))
        .drop("new_global_index")
    )

    if updated.is_empty():
        return 0

    (
        dataset_vars_table.merge_insert(on=["feature_uid", "dataset_uid"])
        .when_matched_update_all()
        .execute(updated)
    )
    return len(updated)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_dataset_vars(
    dataset_vars_table: lancedb.table.Table,
    dataset_uid: str,
    *,
    spec: "ZarrGroupSpec",
    group: "zarr.Group | None" = None,
    expected_feature_count: int | None = None,
    registry_table: lancedb.table.Table | None = None,
) -> list[str]:
    """Validate the _dataset_vars rows for one dataset.

    Parameters
    ----------
    dataset_vars_table:
        The _dataset_vars Lance table.
    dataset_uid:
        The dataset to validate.
    spec:
        The ZarrGroupSpec for the feature space.
    group:
        Optional zarr group for cross-checking row count (dense spaces).
    expected_feature_count:
        Explicit expected feature count (overrides zarr-derived count).
    registry_table:
        Optional registry table to validate feature_uid resolution.

    Returns
    -------
    list[str]
        List of validation error strings. Empty means valid.
    """

    rows = read_dataset_vars(dataset_vars_table, dataset_uid)
    errors: list[str] = []

    n_local = expected_feature_count
    if n_local is None and group is not None:
        n_local = _get_local_feature_count(group, spec)
    if n_local is not None and len(rows) != n_local:
        errors.append(f"_dataset_vars has {len(rows)} rows but expected {n_local} features")

    if rows.is_empty():
        return errors

    null_count = rows["feature_uid"].null_count()
    if null_count > 0:
        errors.append(f"feature_uid has {null_count} null(s)")

    n_unique = rows["feature_uid"].n_unique()
    if n_unique != len(rows):
        errors.append(f"feature_uid has duplicates: {len(rows)} rows but {n_unique} unique values")

    if registry_table is not None:
        registry_df = registry_table.search().select(["uid", "global_index"]).to_polars()
        registry_uids = set(registry_df["uid"].to_list())
        var_uids = rows["feature_uid"].to_list()

        unresolved = [u for u in var_uids if u not in registry_uids]
        if unresolved:
            errors.append(
                f"{len(unresolved)} uid(s) not found in registry. First 5: {unresolved[:5]}"
            )

    return errors


def _get_local_feature_count(
    group: "zarr.Group",
    spec: "ZarrGroupSpec",
) -> int | None:
    """Derive the expected number of local features from the zarr group."""
    import zarr

    from lancell.group_specs import PointerKind

    if spec.pointer_kind is PointerKind.DENSE:
        if "layers" in group and isinstance(group["layers"], zarr.Group):
            for _, arr in group["layers"].arrays():
                return arr.shape[1]
        if "data" in group and isinstance(group["data"], zarr.Array):
            return group["data"].shape[1]
    return None


# ---------------------------------------------------------------------------
# Registry reindexing (unchanged from var_df.py)
# ---------------------------------------------------------------------------


def reindex_registry(table: lancedb.table.Table) -> int:
    """Assign ``global_index`` to any features that do not yet have one.

    Reads only unindexed rows (``global_index IS NULL``) and assigns each a
    unique integer starting from ``max(existing_index) + 1`` (or 0 if the
    table is currently empty).  Rows that already have a ``global_index`` are
    never modified.

    Returns
    -------
    int
        Number of features newly indexed.  0 if all features are already indexed.
    """
    unindexed = (
        table.search().where("global_index IS NULL", prefilter=True).select(["uid"]).to_polars()
    )
    if unindexed.is_empty():
        return 0

    existing = (
        table.search()
        .where("global_index IS NOT NULL", prefilter=True)
        .select(["global_index"])
        .to_polars()
    )
    next_index = int(existing["global_index"].max()) + 1 if not existing.is_empty() else 0

    updated = unindexed.with_columns(
        pl.Series(
            "global_index", list(range(next_index, next_index + len(unindexed))), dtype=pl.Int64
        )
    )
    table.merge_insert(on="uid").when_matched_update_all().execute(updated)
    return len(updated)


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

    registry_df = registry_table.search().select(["uid", "global_index"]).to_polars()
    requested = set(feature_uids)
    registry_uids = set(registry_df["uid"].to_list())

    missing = requested - registry_uids
    if missing:
        raise ValueError(
            f"{len(missing)} UID(s) not found in registry. First 5: {sorted(missing)[:5]}"
        )

    matched = registry_df.filter(pl.col("uid").is_in(list(requested)))
    unindexed = matched.filter(pl.col("global_index").is_null())["uid"].to_list()
    if unindexed:
        raise ValueError(
            f"{len(unindexed)} UID(s) have global_index = None "
            f"(run reindex_registry first). First 5: {unindexed[:5]}"
        )

    global_index_np = matched["global_index"].to_numpy()
    return global_index_np.astype(np.int32, copy=False)[np.argsort(global_index_np)]
