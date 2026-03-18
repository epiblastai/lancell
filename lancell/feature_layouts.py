"""Feature layout helpers: _feature_layouts LanceDB table.

Each unique feature ordering is stored once as a "layout", and datasets
reference layouts by ``layout_uid``.
"""

import hashlib
from typing import TYPE_CHECKING

import lancedb
import numpy as np
import polars as pl

from lancell.util import sql_escape

if TYPE_CHECKING:
    import zarr

    from lancell.group_specs import ZarrGroupSpec


# ---------------------------------------------------------------------------
# Layout UID computation
# ---------------------------------------------------------------------------


def compute_layout_uid(feature_uids: list[str]) -> str:
    """SHA-256 hash of ordered feature list, truncated to 16 hex chars."""
    h = hashlib.sha256()
    for uid in feature_uids:
        h.update(uid.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Build / insert
# ---------------------------------------------------------------------------


def build_feature_layout_df(
    var_df: pl.DataFrame,
    registry_table: lancedb.table.Table,
) -> tuple[str, pl.DataFrame]:
    """Build a DataFrame ready for inserting into _feature_layouts.

    Parameters
    ----------
    var_df:
        One row per local feature, in local feature order. Must have
        ``global_feature_uid``.
    registry_table:
        Feature registry. Used to resolve global_index for each feature_uid.

    Returns
    -------
    tuple[str, pl.DataFrame]
        ``(layout_uid, df)`` with columns: layout_uid, feature_uid,
        local_index, global_index (may be null for unindexed features).

    Raises
    ------
    ValueError
        If any feature_uid is missing from the registry.
    """
    if "global_feature_uid" not in var_df.columns:
        raise ValueError("var_df must have a 'global_feature_uid' column")

    feature_uids = var_df["global_feature_uid"].to_list()
    n = len(feature_uids)
    layout_uid = compute_layout_uid(feature_uids)

    uids_sql = ", ".join(f"'{sql_escape(u)}'" for u in feature_uids)
    registry_df = (
        registry_table.search()
        .where(f"uid IN ({uids_sql})", prefilter=True)
        .select(["uid", "global_index"])
        .to_polars()
    )
    uid_to_global: dict[str, int | None] = dict(
        zip(registry_df["uid"].to_list(), registry_df["global_index"].to_list(), strict=True)
    )
    registry_uids = set(registry_df["uid"].to_list())

    global_indices: list[int | None] = []
    missing: list[str] = []
    for uid in feature_uids:
        if uid not in registry_uids:
            missing.append(uid)
        else:
            global_indices.append(uid_to_global.get(uid))

    if missing:
        raise ValueError(
            f"{len(missing)} uid(s) in var_df not found in registry. First 5: {sorted(missing)[:5]}"
        )

    return layout_uid, pl.DataFrame(
        {
            "layout_uid": [layout_uid] * n,
            "feature_uid": feature_uids,
            "local_index": list(range(n)),
            "global_index": pl.Series(global_indices, dtype=pl.Int64),
        }
    )


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


def layout_exists(table: lancedb.table.Table, layout_uid: str) -> bool:
    """Quick existence check for a layout_uid."""
    result = (
        table.search()
        .where(f"layout_uid = '{sql_escape(layout_uid)}'", prefilter=True)
        .select(["layout_uid"])
        .limit(1)
        .to_polars()
    )
    return not result.is_empty()


def read_feature_layout(
    table: lancedb.table.Table,
    layout_uid: str,
) -> pl.DataFrame:
    """Read all FeatureLayout rows for a layout, sorted by local_index."""
    return (
        table.search()
        .where(f"layout_uid = '{sql_escape(layout_uid)}'", prefilter=True)
        .select(["layout_uid", "feature_uid", "local_index", "global_index"])
        .to_polars()
        .sort("local_index")
    )


# ---------------------------------------------------------------------------
# Sync global_index after reindex_registry
# ---------------------------------------------------------------------------


def sync_layouts_global_index(
    layouts_table: lancedb.table.Table,
    registry_table: lancedb.table.Table,
) -> int:
    """Propagate updated global_index values from registry to _feature_layouts.

    After reindex_registry(), call this to keep the denormalized global_index
    in _feature_layouts consistent with the registry. Uses merge_insert to
    update matched rows.

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
        layouts_table.search()
        .select(["layout_uid", "feature_uid", "local_index", "global_index"])
        .to_polars()
    )
    if all_rows.is_empty():
        return 0

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
        layouts_table.merge_insert(on=["layout_uid", "feature_uid"])
        .when_matched_update_all()
        .execute(updated)
    )
    return len(updated)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_feature_layout(
    layouts_table: lancedb.table.Table,
    layout_uid: str,
    *,
    spec: "ZarrGroupSpec",
    group: "zarr.Group | None" = None,
    expected_feature_count: int | None = None,
    registry_table: lancedb.table.Table | None = None,
) -> list[str]:
    """Validate the _feature_layouts rows for one layout.

    Returns
    -------
    list[str]
        List of validation error strings. Empty means valid.
    """
    rows = read_feature_layout(layouts_table, layout_uid)
    errors: list[str] = []

    n_local = expected_feature_count
    if n_local is None and group is not None:
        n_local = _get_local_feature_count(group, spec)
    if n_local is not None and len(rows) != n_local:
        errors.append(f"_feature_layouts has {len(rows)} rows but expected {n_local} features")

    if rows.is_empty():
        return errors

    null_count = rows["feature_uid"].null_count()
    if null_count > 0:
        errors.append(f"feature_uid has {null_count} null(s)")

    null_gi = rows["global_index"].null_count()
    if null_gi > 0:
        errors.append(f"global_index has {null_gi} null(s); run optimize() to sync from registry")

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
# Registry reindexing
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
