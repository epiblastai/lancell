"""RaggedAtlas: user-facing API for writing, querying, and streaming lancell data.

The LanceDB database IS the atlas. The Python schema class (a LancellBaseSchema
subclass) serves as the descriptor — it declares which pointer fields (feature
spaces) exist and their types. ``RaggedAtlas.open(...)`` or ``.create(...)`` is
the full API — no manifest file to maintain.
"""

import dataclasses
from collections import defaultdict
from types import UnionType
from typing import TYPE_CHECKING, Union, get_args, get_origin

if TYPE_CHECKING:
    from lancell.query import AtlasQuery

import anndata as ad
import lancedb
import numpy as np
import obstore
import pandas as pd
import polars as pl
import zarr

from lancell.batch_array import BatchAsyncArray
from lancell.group_specs import PointerKind, get_spec
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)
from lancell.var_df import (
    build_remap,
    find_datasets_with_features,
    read_remap_if_fresh,
    read_var_df,
    reindex_registry,
    validate_var_df,
    write_remap,
)

# ---------------------------------------------------------------------------
# PointerFieldInfo — metadata about a schema's pointer fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PointerFieldInfo:
    """Metadata extracted from a single pointer field on a cell schema."""

    field_name: str
    feature_space: str
    pointer_kind: PointerKind
    pointer_type: type  # SparseZarrPointer or DenseZarrPointer


def _extract_pointer_fields(
    schema_cls: type[LancellBaseSchema],
) -> dict[str, PointerFieldInfo]:
    """Introspect a schema class and return info for each pointer field."""
    result: dict[str, PointerFieldInfo] = {}
    for name, annotation in schema_cls.__annotations__.items():
        if name == "uid":
            continue
        origin = get_origin(annotation)
        if origin is Union or isinstance(annotation, UnionType):
            inner_types = get_args(annotation)
        else:
            inner_types = (annotation,)

        for t in inner_types:
            if t is type(None):
                continue
            if t is SparseZarrPointer or t is DenseZarrPointer:
                # Convention: field name == feature space name.
                # Enforced at class-definition time in LancellBaseSchema.__init_subclass__.
                feature_space = name
                spec = get_spec(feature_space)
                pointer_kind = PointerKind.SPARSE if t is SparseZarrPointer else PointerKind.DENSE
                if pointer_kind is not spec.pointer_kind:
                    raise TypeError(
                        f"Field '{name}' uses {pointer_kind.value} pointer but "
                        f"feature space '{feature_space}' requires {spec.pointer_kind.value}"
                    )
                result[name] = PointerFieldInfo(
                    field_name=name,
                    feature_space=feature_space,
                    pointer_kind=pointer_kind,
                    pointer_type=t,
                )
                break
    return result


# ---------------------------------------------------------------------------
# Pre-flight schema alignment
# ---------------------------------------------------------------------------

# Fields set automatically by the atlas — never expected in user-provided obs.
_AUTO_FIELDS = {"uid", "dataset_uid"}


def _schema_obs_fields(
    cell_schema: type[LancellBaseSchema],
) -> dict[str, bool]:
    """Return {field_name: required} for user-supplied obs fields.

    Excludes auto-generated fields (uid, dataset_uid) and pointer fields.
    """
    pointer_fields = _extract_pointer_fields(cell_schema)
    result: dict[str, bool] = {}
    for name, field_info in cell_schema.model_fields.items():
        if name in _AUTO_FIELDS or name in pointer_fields:
            continue
        required = field_info.is_required()
        result[name] = required
    return result


def validate_obs_columns(
    obs: pd.DataFrame,
    cell_schema: type[LancellBaseSchema],
    obs_to_schema: dict[str, str] | None = None,
) -> list[str]:
    """Validate that obs columns match the cell schema.

    Parameters
    ----------
    obs:
        The obs DataFrame from an AnnData.
    cell_schema:
        The schema class to validate against.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.

    Returns
    -------
    list[str]
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    schema_fields = _schema_obs_fields(cell_schema)
    obs_to_schema = obs_to_schema or {}

    # Build the set of schema field names reachable from obs columns
    # (either directly or via the mapping)
    reverse_map = {v: k for k, v in obs_to_schema.items()}
    obs_cols = set(obs.columns)

    for field_name, required in schema_fields.items():
        # Field is satisfied if obs has it directly or via mapping
        obs_col = reverse_map.get(field_name, field_name)
        if required and obs_col not in obs_cols:
            errors.append(f"Missing required column '{field_name}'")

    return errors


def align_obs_to_schema(
    adata: ad.AnnData,
    cell_schema: type[LancellBaseSchema],
    *,
    obs_to_schema: dict[str, str] | None = None,
    inplace: bool = False,
) -> ad.AnnData:
    """Align an AnnData's obs to match a cell schema.

    - Renames columns according to ``obs_to_schema``.
    - Raises if required fields are missing (after renaming).
    - Adds ``None`` columns for optional fields not present.
    - Drops extra columns not in the schema.

    Parameters
    ----------
    adata:
        The AnnData to align.
    cell_schema:
        The schema class to align to.
    obs_to_schema:
        Optional mapping from obs column names to schema field names.
        Use this when obs columns have different names than the schema
        expects, e.g. ``{"donor_id": "donor", "cell_type_ontology": "cell_type"}``.
    inplace:
        If True, modify ``adata`` in place. Otherwise return a copy.

    Returns
    -------
    ad.AnnData
        The aligned AnnData.
    """
    errors = validate_obs_columns(adata.obs, cell_schema, obs_to_schema)
    if errors:
        raise ValueError(f"Cannot align obs to schema: {errors}")

    if not inplace:
        adata = adata.copy()

    # Rename obs columns according to mapping
    if obs_to_schema:
        adata.obs = adata.obs.rename(columns=obs_to_schema)

    schema_fields = _schema_obs_fields(cell_schema)
    obs_cols = set(adata.obs.columns)

    # Add None columns for optional fields not present
    for field_name, required in schema_fields.items():
        if not required and field_name not in obs_cols:
            adata.obs[field_name] = None

    # Drop extra columns not in schema
    keep = [c for c in adata.obs.columns if c in schema_fields]
    adata.obs = adata.obs[keep]

    return adata


# ---------------------------------------------------------------------------
# RaggedAtlas
# ---------------------------------------------------------------------------


class RaggedAtlas:
    """Main entry point for reading and writing lancell atlases.

    The atlas is backed by a LanceDB database (cell table + feature registries)
    and a zarr-compatible object store for array data.
    """

    def __init__(
        self,
        db: lancedb.DBConnection,
        cell_table: lancedb.table.Table,
        cell_schema: type[LancellBaseSchema],
        root: zarr.Group,
        registry_tables: dict[str, lancedb.table.Table],
        dataset_table: lancedb.table.Table,
        *,
        update_feature_registries: bool = True,
    ) -> None:
        self.db = db
        self.cell_table = cell_table
        self._cell_schema = cell_schema
        self._root = root
        self._store = root.store.store
        self._pointer_fields = _extract_pointer_fields(cell_schema)
        self._registry_tables = registry_tables
        self._dataset_table = dataset_table

        # Instance-level caches (version-aware for remaps)
        self._remap_cache: dict[tuple[str, str], tuple[int, np.ndarray]] = {}
        self._batch_reader_cache: dict[tuple[str, str], BatchAsyncArray] = {}

        # Validate that global_index is contiguous 0..N-1 within each
        # registry table. A broken index silently corrupts every remap and
        # every reconstructed AnnData.
        registry_errors = self._validate_registries()
        if registry_errors and update_feature_registries:
            for table in self._registry_tables.values():
                reindex_registry(table)
            registry_errors = self._validate_registries()
        if registry_errors:
            raise ValueError(f"Registry validation failed at init: {registry_errors}")

    # -- Construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        db_uri: str,
        cell_table_name: str,
        cell_schema: type[LancellBaseSchema],
        dataset_table_name: str,
        dataset_schema: type[DatasetRecord],
        *,
        store: obstore.store.ObjectStore,
        registry_schemas: dict[str, type[FeatureBaseSchema]],
        update_feature_registries: bool = True,
    ) -> "RaggedAtlas":
        """Create a new atlas, initialising the LanceDB tables.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI (local path or remote).
        cell_table_name:
            Name for the cell table.
        cell_schema:
            A :class:`LancellBaseSchema` subclass declaring the pointer fields.
        dataset_table_name:
            Name for the dataset metadata table.
        dataset_schema:
            A :class:`DatasetRecord` subclass for the dataset schema.
        store:
            An obstore ObjectStore for zarr I/O.
        registry_schemas:
            Mapping of feature space names to their registry schema classes.
            Table names default to ``"{feature_space}_registry"``.
        update_feature_registries:
            If ``True`` (default), automatically run
            :func:`~lancell.var_df.reindex_registry` on any registry whose
            ``global_index`` is not contiguous.  If ``False``, raise on
            broken registries instead.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.create_table(cell_table_name, schema=cell_schema)
        dataset_table = db.create_table(dataset_table_name, schema=dataset_schema)

        registry_tables: dict[str, lancedb.table.Table] = {}
        for fs, schema_cls in registry_schemas.items():
            table_name = f"{fs}_registry"
            registry_tables[fs] = db.create_table(table_name, schema=schema_cls)

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=registry_tables,
            dataset_table=dataset_table,
            update_feature_registries=update_feature_registries,
        )

    @classmethod
    def open(
        cls,
        db_uri: str,
        cell_table_name: str,
        cell_schema: type[LancellBaseSchema],
        dataset_table_name: str,
        *,
        store: obstore.store.ObjectStore,
        registry_tables: dict[str, str],
        update_feature_registries: bool = True,
    ) -> "RaggedAtlas":
        """Open an existing atlas.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        cell_table_name:
            Name of the cell table.
        cell_schema:
            The schema class (must match the table's schema).
        dataset_table_name:
            Name of the dataset metadata table.
        store:
            An obstore ObjectStore for zarr I/O.
        registry_tables:
            Mapping of feature space names to LanceDB table names.
        update_feature_registries:
            If ``True`` (default), automatically run
            :func:`~lancell.var_df.reindex_registry` on any registry whose
            ``global_index`` is not contiguous.  If ``False``, raise on
            broken registries instead.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.open_table(cell_table_name)
        dataset_table = db.open_table(dataset_table_name)

        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_tables.items():
            resolved_registries[fs] = db.open_table(table_name)

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="r")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            update_feature_registries=update_feature_registries,
        )

    # -- Store helpers ------------------------------------------------------

    def _get_remap(self, zarr_group: str, feature_space: str) -> np.ndarray:
        """Load remap, recomputing and saving if the registry version has changed.

        Checks the registry table version on every call.  If the cached remap
        was built against the current version it is returned immediately;
        otherwise the remap is rebuilt from the var_df sidecar and persisted.
        """
        registry_table = self._registry_tables[feature_space]
        current_version = registry_table.version
        cache_key = (zarr_group, feature_space)

        cached_entry = self._remap_cache.get(cache_key)
        if cached_entry is not None:
            cached_version, cached_remap = cached_entry
            if cached_version == current_version:
                return cached_remap

        # In-memory cache miss or stale — try the on-disk remap
        group = self._root[zarr_group]
        disk_remap = read_remap_if_fresh(self._store, group, current_version)
        if disk_remap is not None:
            self._remap_cache[cache_key] = (current_version, disk_remap)
            return disk_remap

        # Rebuild from var_df + registry
        var_df = read_var_df(self._store, zarr_group)
        remap = build_remap(var_df, registry_table)
        if not self._root.store.read_only:
            write_remap(self._store, group, remap, registry_version=current_version)
        self._remap_cache[cache_key] = (current_version, remap)
        return remap

    def _get_batch_reader(self, zarr_group: str, array_name: str) -> BatchAsyncArray:
        """Get a cached BatchAsyncArray reader for a zarr array."""
        cache_key = (zarr_group, array_name)
        reader = self._batch_reader_cache.get(cache_key)
        if reader is None:
            reader = BatchAsyncArray.from_array(self._root[f"{zarr_group}/{array_name}"])
            self._batch_reader_cache[cache_key] = reader
        return reader

    # -- Query entry point --------------------------------------------------

    def query(self) -> "AtlasQuery":
        """Start building a query against this atlas."""
        from lancell.query import AtlasQuery

        return AtlasQuery(self)

    # -- Feature registration -----------------------------------------------

    def register_features(
        self,
        feature_space: str,
        features: list[FeatureBaseSchema] | pl.DataFrame,
    ) -> int:
        """Register features in a feature registry.

        Must be called before ingestion for feature spaces that
        have a registry (``has_var_df=True``).

        This method only inserts new feature rows — it does **not** assign
        ``global_index``.  Call :func:`~lancell.var_df.reindex_registry`
        on the registry table after all registrations are complete to assign
        contiguous indices.  This two-step approach avoids index races when
        multiple writers register features concurrently.

        Parameters
        ----------
        feature_space:
            Which feature space to register features for.
        features:
            Either a list of ``FeatureBaseSchema`` records or a Polars
            DataFrame with at minimum a ``uid`` column.

        Returns
        -------
        int
            Number of newly registered features.
        """
        if feature_space not in self._registry_tables:
            raise ValueError(
                f"No registry table for feature space '{feature_space}'. "
                f"Ensure a registry schema was provided at create() time."
            )
        registry_table = self._registry_tables[feature_space]

        if isinstance(features, pl.DataFrame):
            if "uid" not in features.columns:
                raise ValueError("features DataFrame must have a 'uid' column")
            features_df = features
        else:
            features_df = pl.DataFrame([f.model_dump() for f in features])

        # Deduplicate within the input batch; merge_insert(on="uid") with
        # when_not_matched_insert_all handles skipping rows that already
        # exist in the registry.  global_index is NOT assigned here — call
        # reindex_registry() after all registrations to assign contiguous
        # indices, avoiding races between concurrent writers.
        new_records = features_df.unique(subset=["uid"], keep="first")

        n_before = registry_table.count_rows()
        (registry_table.merge_insert(on="uid").when_not_matched_insert_all().execute(new_records))
        return registry_table.count_rows() - n_before

    # -- Maintenance --------------------------------------------------------

    def optimize(self) -> None:
        """Compact tables and reindex feature registries.

        Calls ``table.optimize()`` on the cell, dataset, and registry tables
        to compact small Lance fragments, then assigns contiguous
        ``global_index`` values on every registry via
        :func:`~lancell.var_df.reindex_registry`.
        """
        self.cell_table.optimize()
        self._dataset_table.optimize()
        for table in self._registry_tables.values():
            table.optimize()
            reindex_registry(table)

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        *,
        check_zarr: bool = True,
        check_var_dfs: bool = True,
        check_registries: bool = True,
    ) -> list[str]:
        """Validate atlas consistency. Returns a list of error strings.

        Parameters
        ----------
        check_zarr:
            Open each unique zarr group and validate against its spec.
        check_var_dfs:
            For feature spaces with var_df, validate sidecars.
        check_registries:
            Check that registry tables exist and global_index is contiguous.
        """
        errors: list[str] = []

        # Schema validation
        for pf in self._pointer_fields.values():
            spec = get_spec(pf.feature_space)
            if pf.pointer_kind is not spec.pointer_kind:
                errors.append(
                    f"Field '{pf.field_name}': pointer_kind {pf.pointer_kind.value} "
                    f"doesn't match spec {spec.pointer_kind.value}"
                )

        if check_registries:
            errors.extend(self._validate_registries())

        # Collect unique zarr groups from dataset table
        zarr_groups_by_space = self._collect_zarr_groups()

        if check_zarr:
            errors.extend(self._validate_zarr_groups(zarr_groups_by_space))

        if check_var_dfs:
            errors.extend(self._validate_var_dfs(zarr_groups_by_space))

        return errors

    def _collect_zarr_groups(self) -> dict[str, set[str]]:
        """Collect unique zarr groups per feature space from the dataset table."""
        result: dict[str, set[str]] = defaultdict(set)
        datasets_df = (
            self._dataset_table.search().select(["feature_space", "zarr_group"]).to_polars()
        )
        if datasets_df.is_empty():
            return result
        for row in datasets_df.iter_rows(named=True):
            result[row["feature_space"]].add(row["zarr_group"])
        return result

    def list_datasets(self) -> pl.DataFrame:
        """Return a Polars DataFrame of all ingested datasets."""
        return self._dataset_table.search().to_polars()

    def datasets_with_features(
        self,
        feature_uids: str | list[str],
        feature_space: str,
    ) -> pl.DataFrame:
        """Find datasets that measured specific features.

        Uses DuckDB to query across var_df sidecars.  See
        :func:`~lancell.var_df.find_datasets_with_features` for details.

        Parameters
        ----------
        feature_uids:
            One or more ``global_feature_uid`` values to search for.
        feature_space:
            Which feature space to search within (e.g. ``"gene_expression"``).

        Returns
        -------
        polars.DataFrame
            One row per (zarr_group, feature) match with columns
            ``zarr_group``, ``global_feature_uid``, plus dataset metadata.
        """
        return find_datasets_with_features(
            self._store, self._dataset_table, feature_uids, feature_space
        )

    def _validate_registries(self) -> list[str]:
        errors: list[str] = []
        for fs, table in self._registry_tables.items():
            df = table.search().select(["uid", "global_index"]).to_polars()
            if df.is_empty():
                continue
            null_count = df["global_index"].null_count()
            if null_count > 0:
                errors.append(
                    f"Registry '{fs}': {null_count} row(s) have no global_index. "
                    f"Run reindex_registry(table) to fix."
                )
                continue
            indices = sorted(df["global_index"].to_list())
            expected = list(range(len(indices)))
            if indices != expected:
                errors.append(
                    f"Registry '{fs}': global_index is not contiguous 0..{len(indices) - 1}. "
                    f"Run reindex_registry(table) to fix."
                )
        return errors

    def _validate_zarr_groups(self, zarr_groups_by_space: dict[str, set[str]]) -> list[str]:
        errors: list[str] = []
        for fs, groups in zarr_groups_by_space.items():
            spec = get_spec(fs)
            for zg in groups:
                group = self._root[zg]
                group_errors = spec.validate_group(group)
                for e in group_errors:
                    errors.append(f"zarr group '{zg}': {e}")
        return errors

    def _validate_var_dfs(self, zarr_groups_by_space: dict[str, set[str]]) -> list[str]:
        errors: list[str] = []
        for fs, groups in zarr_groups_by_space.items():
            spec = get_spec(fs)
            if not spec.has_var_df:
                continue
            registry = self._registry_tables.get(fs)
            for zg in groups:
                var_df = read_var_df(self._store, zg)
                group = self._root[zg]
                vd_errors = validate_var_df(var_df, spec=spec, group=group, registry_table=registry)
                for e in vd_errors:
                    errors.append(f"var_df '{zg}': {e}")
        return errors
