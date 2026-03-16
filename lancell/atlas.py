"""RaggedAtlas: user-facing API for writing, querying, and streaming lancell data.

The LanceDB database IS the atlas. The Python schema class (a LancellBaseSchema
subclass) serves as the descriptor — it declares which pointer fields (feature
spaces) exist and their types. ``RaggedAtlas.open(...)`` or ``.create(...)`` is
the full API — no manifest file to maintain.
"""

import json
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lancell.group_reader import GroupReader
    from lancell.query import AtlasQuery

import lancedb
import obstore
import polars as pl
import zarr

from lancell.dataset_vars import (
    build_dataset_vars_df,
    reindex_registry,
    sync_dataset_vars_global_index,
    validate_dataset_vars,
)
from lancell.group_specs import get_spec
from lancell.obs_alignment import _extract_pointer_fields
from lancell.schema import (
    AtlasVersionRecord,
    DatasetRecord,
    DatasetVar,
    FeatureBaseSchema,
    LancellBaseSchema,
)

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
        version_table: lancedb.table.Table,
        dataset_vars_table: lancedb.table.Table,
    ) -> None:
        self.db = db
        self.cell_table = cell_table
        self._cell_schema = cell_schema
        self._root = root
        self._store = root.store.store
        self._pointer_fields = _extract_pointer_fields(cell_schema)
        self._registry_tables = registry_tables
        self._dataset_table = dataset_table
        self._version_table = version_table
        self._dataset_vars_table = dataset_vars_table

        # Instance-level cache: one GroupReader per (zarr_group, feature_space)
        self._group_readers: dict[tuple[str, str], GroupReader] = {}

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
        version_table_name: str = "atlas_versions",
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
        version_table_name:
            Name for the version tracking table.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.create_table(cell_table_name, schema=cell_schema)
        dataset_table = db.create_table(dataset_table_name, schema=dataset_schema)

        registry_tables: dict[str, lancedb.table.Table] = {}
        for fs, schema_cls in registry_schemas.items():
            table_name = f"{fs}_registry"
            registry_tables[fs] = db.create_table(table_name, schema=schema_cls)

        version_table = db.create_table(version_table_name, schema=AtlasVersionRecord)

        dataset_vars_table = db.create_table("_dataset_vars", schema=DatasetVar)
        dataset_vars_table.create_fts_index("feature_uid")
        dataset_vars_table.create_fts_index("dataset_uid")

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=registry_tables,
            dataset_table=dataset_table,
            version_table=version_table,
            dataset_vars_table=dataset_vars_table,
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
        version_table_name: str = "atlas_versions",
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
        version_table_name:
            Name of the version tracking table.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.open_table(cell_table_name)
        dataset_table = db.open_table(dataset_table_name)

        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_tables.items():
            resolved_registries[fs] = db.open_table(table_name)

        version_table = db.open_table(version_table_name)
        dataset_vars_table = db.open_table("_dataset_vars")

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="a")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            dataset_vars_table=dataset_vars_table,
        )

    # -- Store helpers ------------------------------------------------------

    def _get_group_reader(self, zarr_group: str, feature_space: str) -> "GroupReader":
        """Return (cached) GroupReader for the given zarr_group + feature_space."""
        from lancell.group_reader import GroupReader

        key = (zarr_group, feature_space)
        if key not in self._group_readers:
            datasets_df = (
                self._dataset_table.search()
                .where(
                    f"zarr_group = '{zarr_group}' AND feature_space = '{feature_space}'",
                    prefilter=True,
                )
                .select(["uid"])
                .to_polars()
            )
            dataset_uid: str | None = datasets_df["uid"][0] if not datasets_df.is_empty() else None

            self._group_readers[key] = GroupReader.from_atlas_root(
                zarr_group=zarr_group,
                feature_space=feature_space,
                store=self._store,
                dataset_vars_table=self._dataset_vars_table,
                dataset_uid=dataset_uid,
            )
        return self._group_readers[key]

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
        ``global_index``.  Call :func:`~lancell.dataset_vars.reindex_registry`
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

    # -- Dataset vars -------------------------------------------------------

    def add_dataset_vars(
        self,
        var_df: pl.DataFrame,
        dataset_uid: str,
        feature_space: str,
    ) -> None:
        """Index all feature-level metadata for a newly ingested dataset.

        Call once per dataset after zarr data is written.
        var_df must contain a 'global_feature_uid' column.
        No-op if this atlas has no _dataset_vars table.

        Parameters
        ----------
        var_df:
            One row per local feature in local feature order.
            Must have a ``global_feature_uid`` column.
        dataset_uid:
            The DatasetRecord uid for this dataset.
        feature_space:
            Which feature space this dataset belongs to (used to look up registry).
        """
        registry_table = self._registry_tables.get(feature_space)
        if registry_table is None:
            raise ValueError(
                f"No registry table for feature space '{feature_space}'. "
                f"Ensure a registry schema was provided at create() time."
            )
        df = build_dataset_vars_df(var_df, dataset_uid, registry_table)
        self._dataset_vars_table.add(df)

    # -- Maintenance --------------------------------------------------------

    def optimize(self) -> None:
        """Compact tables and reindex feature registries.

        Calls ``table.optimize()`` on the cell, dataset, and registry tables
        to compact small Lance fragments, then assigns contiguous
        ``global_index`` values on every registry via
        :func:`~lancell.dataset_vars.reindex_registry`, and propagates updated
        indices to ``_dataset_vars`` via
        :func:`~lancell.dataset_vars.sync_dataset_vars_global_index`.
        """
        self.cell_table.optimize()
        self._dataset_table.optimize()
        for table in self._registry_tables.values():
            reindex_registry(table)
            table.optimize()
            sync_dataset_vars_global_index(self._dataset_vars_table, table)

        self._dataset_vars_table.create_fts_index("feature_uid", replace=True)
        self._dataset_vars_table.create_fts_index("dataset_uid", replace=True)
        self._dataset_vars_table.optimize()

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
            For feature spaces with var_df, validate _dataset_vars rows.
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
            errors.extend(self._validate_dataset_vars(zarr_groups_by_space))

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

    def find_datasets_with_features(
        self,
        feature_uids: str | list[str],
        feature_space: str,
    ) -> pl.DataFrame:
        """Find datasets that measured specific features.

        Uses the LanceDB FTS index on ``_dataset_vars.feature_uid``.
        Note: rows added after the last FTS index build are not yet indexed;
        call ``optimize()`` after bulk ingestion to refresh.

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
        if isinstance(feature_uids, str):
            feature_uids = [feature_uids]
        return self._find_datasets_with_features_fast(feature_uids, feature_space)

    def _find_datasets_with_features_fast(
        self, feature_uids: list[str], feature_space: str
    ) -> pl.DataFrame:
        from lancedb.query import MatchQuery

        query_str = " ".join(feature_uids)
        n_datasets = self._dataset_table.count_rows()
        limit = max(len(feature_uids) * n_datasets, 1)

        pairs = (
            self._dataset_vars_table.search(MatchQuery(query_str), query_type="fts")
            .select(["feature_uid", "dataset_uid"])
            .limit(limit)
            .to_polars()
        )
        if pairs.is_empty():
            return pl.DataFrame(schema={"zarr_group": pl.Utf8, "global_feature_uid": pl.Utf8})

        pairs = pairs.unique(subset=["feature_uid", "dataset_uid"])

        datasets_df = (
            self._dataset_table.search()
            .where(f"feature_space = '{feature_space}'", prefilter=True)
            .to_polars()
        )

        return (
            pairs.rename({"feature_uid": "global_feature_uid"})
            .join(datasets_df, left_on="dataset_uid", right_on="uid", how="inner")
            .drop("dataset_uid")
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

    def _validate_dataset_vars(self, zarr_groups_by_space: dict[str, set[str]]) -> list[str]:
        errors: list[str] = []
        # Build a lookup: (zarr_group, feature_space) -> dataset_uid
        datasets_df = (
            self._dataset_table.search().select(["uid", "zarr_group", "feature_space"]).to_polars()
        )
        for fs, groups in zarr_groups_by_space.items():
            spec = get_spec(fs)
            if not spec.has_var_df:
                continue
            registry = self._registry_tables.get(fs)
            for zg in groups:
                matched = datasets_df.filter(
                    (pl.col("zarr_group") == zg) & (pl.col("feature_space") == fs)
                )
                if matched.is_empty():
                    errors.append(f"No dataset record for zarr_group='{zg}', feature_space='{fs}'")
                    continue
                dataset_uid = matched["uid"][0]
                group = self._root[zg]
                dv_errors = validate_dataset_vars(
                    self._dataset_vars_table,
                    dataset_uid,
                    spec=spec,
                    group=group,
                    registry_table=registry,
                )
                for e in dv_errors:
                    errors.append(f"_dataset_vars '{zg}': {e}")
        return errors

    # -- Versioning ---------------------------------------------------------

    def snapshot(self) -> int:
        """Record a consistent snapshot of all table versions.

        Returns the new atlas version number (0-indexed, monotonically increasing).
        Raises ``ValueError`` if the atlas was created without a version table, or if
        validation errors are found.

        """
        errors = self.validate()
        if errors:
            raise ValueError(
                "Atlas validation failed — fix errors before snapshotting:\n"
                + "\n".join(f"  • {e}" for e in errors)
            )

        existing = self._version_table.search().select(["version"]).to_polars()
        if existing.is_empty():
            next_version = 0
        else:
            next_version = existing["version"].max() + 1

        registry_names = {fs: t.name for fs, t in self._registry_tables.items()}
        registry_versions = {fs: t.version for fs, t in self._registry_tables.items()}

        record = AtlasVersionRecord(
            version=next_version,
            cell_table_name=self.cell_table.name,
            cell_table_version=self.cell_table.version,
            dataset_table_name=self._dataset_table.name,
            dataset_table_version=self._dataset_table.version,
            registry_table_names=json.dumps(registry_names),
            registry_table_versions=json.dumps(registry_versions),
            dataset_vars_table_version=self._dataset_vars_table.version,
            total_cells=self.cell_table.count_rows(),
        )
        self._version_table.add([record])
        return next_version

    @classmethod
    def list_versions(
        cls,
        db_uri: str,
        *,
        version_table_name: str = "atlas_versions",
    ) -> pl.DataFrame:
        """Return a DataFrame of all recorded snapshots, sorted by version.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        version_table_name:
            Name of the version tracking table.
        """
        db = lancedb.connect(db_uri)
        version_table = db.open_table(version_table_name)
        return version_table.search().to_polars().sort("version")

    @classmethod
    def checkout(
        cls,
        db_uri: str,
        version: int,
        cell_schema: type[LancellBaseSchema],
        store: obstore.store.ObjectStore,
        *,
        version_table_name: str = "atlas_versions",
    ) -> "RaggedAtlas":
        """Open a read-only atlas pinned to a specific snapshot version.

        Parameters
        ----------
        db_uri:
            LanceDB connection URI.
        version:
            Atlas version number (as returned by :meth:`snapshot`).
        cell_schema:
            The schema class used when the atlas was created.
        store:
            An obstore ObjectStore for zarr I/O.
        version_table_name:
            Name of the version tracking table.
        """
        db = lancedb.connect(db_uri)
        version_table = db.open_table(version_table_name)

        records = version_table.search().where(f"version = {version}", prefilter=True).to_polars()
        if records.is_empty():
            raise ValueError(
                f"Atlas version {version} not found. "
                f"Use RaggedAtlas.list_versions('{db_uri}') to see available versions."
            )
        row = records.row(0, named=True)

        cell_table = db.open_table(row["cell_table_name"])
        cell_table.checkout(row["cell_table_version"])

        dataset_table = db.open_table(row["dataset_table_name"])
        dataset_table.checkout(row["dataset_table_version"])

        registry_names: dict[str, str] = json.loads(row["registry_table_names"])
        registry_versions: dict[str, int] = json.loads(row["registry_table_versions"])
        resolved_registries: dict[str, lancedb.table.Table] = {}
        for fs, table_name in registry_names.items():
            t = db.open_table(table_name)
            t.checkout(registry_versions[fs])
            resolved_registries[fs] = t

        dataset_vars_table = db.open_table("_dataset_vars")
        dataset_vars_table.checkout(row["dataset_vars_table_version"])

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="r")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
            version_table=version_table,
            dataset_vars_table=dataset_vars_table,
        )
