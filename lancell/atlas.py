"""RaggedAtlas: user-facing API for writing, querying, and streaming lancell data.

The LanceDB database IS the atlas. The Python schema class (a LancellBaseSchema
subclass) serves as the descriptor — it declares which pointer fields (feature
spaces) exist and their types. ``RaggedAtlas.open(...)`` or ``.create(...)`` is
the full API — no manifest file to maintain.
"""

import dataclasses
import functools
from collections import defaultdict
from collections.abc import Iterator
from types import UnionType
from typing import Union, get_args, get_origin

import anndata as ad
import lancedb
import numpy as np
import obstore
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp
import zarr

from lancell.batch_array import BatchArray
from lancell.group_specs import ZARR_SPECS, FeatureSpace, PointerKind, ZarrGroupSpec
from lancell.schema import (
    DatasetRecord,
    DenseZarrPointer,
    FeatureBaseSchema,
    LancellBaseSchema,
    SparseZarrPointer,
)
from lancell.var_df import (
    build_remap,
    read_remap,
    read_var_df,
    validate_remap,
    validate_var_df,
    write_remap,
    write_var_df,
)


# ---------------------------------------------------------------------------
# PointerFieldInfo — metadata about a schema's pointer fields
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PointerFieldInfo:
    """Metadata extracted from a single pointer field on a cell schema."""

    field_name: str
    feature_space: FeatureSpace
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
                # TODO: Enforce this in validation for the LancellBaseSchema
                # Convention: field name == FeatureSpace value.
                # We use the field name (not t.feature_space) because t is
                # the *type* SparseZarrPointer, not an instance — there's no
                # .feature_space value to read at class-introspection time.
                feature_space = _field_name_to_feature_space(name)
                spec = ZARR_SPECS[feature_space]
                pointer_kind = PointerKind.SPARSE if t is SparseZarrPointer else PointerKind.DENSE
                if pointer_kind is not spec.pointer_kind:
                    raise TypeError(
                        f"Field '{name}' uses {pointer_kind.value} pointer but "
                        f"feature space '{feature_space.value}' requires {spec.pointer_kind.value}"
                    )
                result[name] = PointerFieldInfo(
                    field_name=name,
                    feature_space=feature_space,
                    pointer_kind=pointer_kind,
                    pointer_type=t,
                )
                break
    return result


def _field_name_to_feature_space(field_name: str) -> FeatureSpace:
    """Map a schema field name to a FeatureSpace enum value."""
    for fs in FeatureSpace:
        if fs.value == field_name:
            return fs
    raise ValueError(
        f"No FeatureSpace matches field name '{field_name}'. "
        f"Valid values: {[fs.value for fs in FeatureSpace]}"
    )


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
) -> list[str]:
    """Validate that obs columns match the cell schema.

    Parameters
    ----------
    obs:
        The obs DataFrame from an AnnData.
    cell_schema:
        The schema class to validate against.

    Returns
    -------
    list[str]
        List of error strings. Empty list means valid.
    """
    errors: list[str] = []
    schema_fields = _schema_obs_fields(cell_schema)
    obs_cols = set(obs.columns)

    for field_name, required in schema_fields.items():
        if required and field_name not in obs_cols:
            errors.append(f"Missing required column '{field_name}'")

    return errors


# TODO: I don't understand this function? I thought the purpose was to provide a mapping
# of current adata.obs columns to their respective fields in cell_schema. This current function
# just errors unless everything is already aligned. It only supports column dropping.
def align_obs_to_schema(
    adata: ad.AnnData,
    cell_schema: type[LancellBaseSchema],
    *,
    inplace: bool = False,
) -> ad.AnnData:
    """Align an AnnData's obs to match a cell schema.

    - Raises if required fields are missing.
    - Adds ``None`` columns for optional fields not present.
    - Drops extra columns not in the schema.

    Parameters
    ----------
    adata:
        The AnnData to align.
    cell_schema:
        The schema class to align to.
    inplace:
        If True, modify ``adata`` in place. Otherwise return a copy.

    Returns
    -------
    ad.AnnData
        The aligned AnnData.
    """
    errors = validate_obs_columns(adata.obs, cell_schema)
    if errors:
        raise ValueError(
            f"Cannot align obs to schema: {errors}"
        )

    if not inplace:
        adata = adata.copy()

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
        registry_tables: dict[FeatureSpace, lancedb.table.Table],
        dataset_table: lancedb.table.Table,
    ) -> None:
        self.db = db
        self.cell_table = cell_table
        self._cell_schema = cell_schema
        self._root = root
        self._store = root.store.store
        self._pointer_fields = _extract_pointer_fields(cell_schema)
        self._registry_tables = registry_tables
        self._dataset_table = dataset_table
        # TODO: Which of these should be LRU caches?
        self._remap_cache: dict[str, np.ndarray] = {}
        self._batch_reader_cache: dict[str, BatchArray] = {}

        # TODO: We don't have to validate everything, but we 100% need
        # to validate that global_index is contiguous and unique for any
        # and all registry_tables. It's an absolute nightmare if that's
        # broken. The feature registry tables themselves are generally quite
        # small so this ought to be cheap to do and worth it for sanity.

    # -- Construction -------------------------------------------------------

    @classmethod
    def create(
        cls,
        db_uri: str,
        cell_table_name: str,
        cell_schema: type[LancellBaseSchema],
        *,
        store: obstore.store.ObjectStore,
        registry_schemas: dict[FeatureSpace, type[FeatureBaseSchema]] | None = None,
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
        store:
            An obstore ObjectStore for zarr I/O.
        registry_schemas:
            Optional mapping of feature spaces to their registry schema classes.
            Tables are created for each. Default table name convention:
            ``"{feature_space.value}_registry"``.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.create_table(cell_table_name, schema=cell_schema)
        dataset_table = db.create_table("_datasets", schema=DatasetRecord)

        registry_tables: dict[FeatureSpace, lancedb.table.Table] = {}
        if registry_schemas:
            for fs, schema_cls in registry_schemas.items():
                table_name = f"{fs.value}_registry"
                registry_tables[fs] = db.create_table(table_name, schema=schema_cls)

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="w")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=registry_tables,
            dataset_table=dataset_table,
        )

    @classmethod
    def open(
        cls,
        db_uri: str,
        cell_table_name: str,
        cell_schema: type[LancellBaseSchema],
        *,
        store: obstore.store.ObjectStore,
        # TODO: I'd prefer to make these required and not to assign
        # defaults like `f"{pf.feature_space.value}_registry"`
        registry_tables: dict[FeatureSpace, str] | None = None,
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
        store:
            An obstore ObjectStore for zarr I/O.
        registry_tables:
            Optional mapping of feature spaces to LanceDB table names.
            Defaults to ``"{feature_space.value}_registry"`` for each
            feature space that has ``has_var_df=True`` in its spec.
        """
        db = lancedb.connect(db_uri)
        cell_table = db.open_table(cell_table_name)
        # TODO: Require dataset_table_name and dataset_schema as arguments instead
        # of hardcoding.
        dataset_table = db.open_table("_datasets")

        # _extract_pointer_fields is called in __init__, but we need it here
        # to resolve registry table names before constructing the atlas.
        pointer_fields = _extract_pointer_fields(cell_schema)
        resolved_registries: dict[FeatureSpace, lancedb.table.Table] = {}
        for pf in pointer_fields.values():
            spec = ZARR_SPECS[pf.feature_space]
            if not spec.has_var_df:
                continue
            if registry_tables and pf.feature_space in registry_tables:
                table_name = registry_tables[pf.feature_space]
            else:
                table_name = f"{pf.feature_space.value}_registry"
            resolved_registries[pf.feature_space] = db.open_table(table_name)

        root = zarr.open_group(zarr.storage.ObjectStore(store), mode="r")

        return cls(
            db=db,
            cell_table=cell_table,
            cell_schema=cell_schema,
            root=root,
            registry_tables=resolved_registries,
            dataset_table=dataset_table,
        )

    # -- Store helpers ------------------------------------------------------

    def _get_remap(self, zarr_group: str) -> np.ndarray:
        """Load a remap array, using the cache."""
        if zarr_group not in self._remap_cache:
            self._remap_cache[zarr_group] = read_remap(self._store, zarr_group)
        return self._remap_cache[zarr_group]

    # Any reason not to use the built-in functools.lru_cache for this?
    _BATCH_READER_CACHE_MAX = 64

    def _get_batch_reader(self, zarr_group: str, array_name: str) -> BatchArray:
        """Get a cached BatchArray reader for a zarr array."""
        key = f"{zarr_group}/{array_name}"
        if key not in self._batch_reader_cache:
            if len(self._batch_reader_cache) >= self._BATCH_READER_CACHE_MAX:
                # Evict oldest entry
                oldest_key = next(iter(self._batch_reader_cache))
                del self._batch_reader_cache[oldest_key]
            self._batch_reader_cache[key] = BatchArray.from_array(
                self._root[f"{zarr_group}/{array_name}"]
            )
        return self._batch_reader_cache[key]

    # -- Query entry point --------------------------------------------------

    def query(self) -> "AtlasQuery":
        """Start building a query against this atlas."""
        return AtlasQuery(self)

    # -- Feature registration -----------------------------------------------

    def register_features(
        self,
        feature_space: FeatureSpace,
        features: list[FeatureBaseSchema] | pl.DataFrame,
    ) -> int:
        """Register features in a feature registry.

        Must be called before ``add_from_anndata`` for feature spaces that
        have a registry (``has_var_df=True``).

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
                f"No registry table for feature space '{feature_space.value}'. "
                f"Ensure a registry schema was provided at create() time."
            )
        registry_table = self._registry_tables[feature_space]

        if isinstance(features, pl.DataFrame):
            if "uid" not in features.columns:
                raise ValueError("features DataFrame must have a 'uid' column")
            features_df = features
        else:
            features_df = pl.DataFrame([f.model_dump() for f in features])

        uids = features_df["uid"].to_list()

        # Load existing registry
        existing_df = registry_table.search().select(["uid", "global_index"]).to_polars()
        if existing_df.is_empty():
            existing_uids: set[str] = set()
            next_index = 0
        else:
            existing_uids = set(existing_df["uid"].to_list())
            indexed = existing_df.filter(pl.col("global_index").is_not_null())
            next_index = int(indexed["global_index"].max()) + 1 if not indexed.is_empty() else 0

        new_uids = [u for u in uids if u not in existing_uids]
        if not new_uids:
            return 0

        # Deduplicate preserving first occurrence
        seen: set[str] = set()
        unique_new: list[str] = []
        for u in new_uids:
            if u not in seen:
                unique_new.append(u)
                seen.add(u)

        # Filter input to new unique uids, assign global_index
        new_records = features_df.filter(pl.col("uid").is_in(unique_new)).unique(
            subset=["uid"], keep="first"
        )
        # Sort to match unique_new order for index assignment
        uid_order = {uid: i for i, uid in enumerate(unique_new)}
        new_records = new_records.with_columns(
            pl.col("uid").replace_strict(uid_order, return_dtype=pl.Int64).alias("_order")
        ).sort("_order").drop("_order")
        new_records = new_records.with_columns(
            pl.Series("global_index", list(range(next_index, next_index + len(unique_new))))
        )

        (
            registry_table.merge_insert(on="uid")
            .when_not_matched_insert_all()
            .execute(new_records)
        )
        return len(unique_new)

    # -- Writing: add_from_anndata ------------------------------------------

    def add_from_anndata(
        self,
        adata: ad.AnnData,
        *,
        feature_space: FeatureSpace,
        zarr_group: str,
        # TODO: layer_name should apply to dense data as well
        layer_name: str | None = None,
        chunk_size: int = 4096,
        shard_size: int = 65536,
    ) -> int:
        """Ingest an AnnData into the atlas.

        Writes zarr arrays, var_df sidecar, remap, and inserts cell records
        into the cell table. Features must already be registered via
        :meth:`register_features`, and ``adata.var`` must contain a
        ``global_feature_uid`` column.

        Parameters
        ----------
        adata:
            The AnnData to ingest.
        feature_space:
            Which feature space this data belongs to.
        zarr_group:
            Zarr group path (relative to atlas store) for this ingestion.
            Must be unique per ingestion call.
        layer_name:
            Required for sparse feature spaces — name of the layer to write
            (e.g. "counts"). Unused for dense feature spaces.
        chunk_size:
            Zarr chunk size for 1D arrays.
        shard_size:
            Zarr shard size for 1D arrays.

        Returns
        -------
        int
            Number of cells ingested.
        """
        spec = ZARR_SPECS[feature_space]

        if spec.pointer_kind is PointerKind.SPARSE and layer_name is None:
            raise ValueError(
                "layer_name is required for sparse feature spaces "
                f"(feature_space='{feature_space.value}')"
            )

        # Pre-flight: validate obs columns match schema before any writes
        obs_errors = validate_obs_columns(adata.obs, self._cell_schema)
        if obs_errors:
            raise ValueError(
                f"obs columns do not match cell schema: {obs_errors}"
            )

        # Find the pointer field for this feature space
        pointer_field = None
        for pf in self._pointer_fields.values():
            if pf.feature_space == feature_space:
                pointer_field = pf
                break
        if pointer_field is None:
            raise ValueError(
                f"Schema {self._cell_schema.__name__} has no pointer field "
                f"for feature space '{feature_space.value}'"
            )

        n_cells = adata.n_obs

        # Create dataset record first (FK for cells)
        dataset_record = DatasetRecord(
            zarr_group=zarr_group,
            feature_space=feature_space.value,
            n_cells=n_cells,
        )
        dataset_arrow = pa.Table.from_pylist(
            [dataset_record.model_dump()],
            schema=DatasetRecord.to_arrow_schema(),
        )
        self._dataset_table.add(dataset_arrow)

        # Write zarr arrays
        if spec.pointer_kind is PointerKind.SPARSE:
            starts, ends = self._write_sparse_zarr(
                adata, zarr_group, layer_name, chunk_size, shard_size
            )
        else:
            self._write_dense_zarr(adata, zarr_group, chunk_size, shard_size)

        # Write var_df sidecar
        if spec.has_var_df:
            self._write_var_sidecar(adata, feature_space, zarr_group)

        # Build cell records from obs columns
        obs_field_names = list(_schema_obs_fields(self._cell_schema).keys())
        records = []
        for i in range(n_cells):
            if spec.pointer_kind is PointerKind.SPARSE:
                pointer = SparseZarrPointer(
                    feature_space=feature_space,
                    zarr_group=zarr_group,
                    start=int(starts[i]),
                    end=int(ends[i]),
                )
            else:
                pointer = DenseZarrPointer(
                    feature_space=feature_space,
                    zarr_group=zarr_group,
                    position=i,
                )

            extra = {
                col: adata.obs.iloc[i][col]
                for col in obs_field_names
                if col in adata.obs.columns
            }
            record_kwargs = {
                pointer_field.field_name: pointer,
                "dataset_uid": dataset_record.uid,
                **extra,
            }
            records.append(self._cell_schema(**record_kwargs))

        arrow_schema = self._cell_schema.to_arrow_schema()
        arrow_table = pa.Table.from_pylist(
            [r.model_dump() for r in records], schema=arrow_schema
        )
        self.cell_table.add(arrow_table)
        return n_cells

    def _write_sparse_zarr(
        self,
        adata: ad.AnnData,
        zarr_group: str,
        layer_name: str,
        chunk_size: int,
        shard_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Write sparse (CSR) data to zarr arrays. Returns (starts, ends)."""
        csr = sp.csr_matrix(adata.X)
        flat_indices = csr.indices.astype(np.uint32)
        flat_values = csr.data

        starts = csr.indptr[:-1].astype(np.int64)
        ends = csr.indptr[1:].astype(np.int64)

        group = self._root.create_group(zarr_group)

        group.create_array(
            "indices",
            data=flat_indices,
            chunks=(chunk_size,),
            shards=(shard_size,),
        )

        layers = group.create_group("layers")
        layers.create_array(
            layer_name,
            data=flat_values,
            chunks=(chunk_size,),
            shards=(shard_size,),
        )

        return starts, ends

    def _write_dense_zarr(
        self,
        adata: ad.AnnData,
        zarr_group: str,
        chunk_size: int,
        shard_size: int,
    ) -> None:
        """Write dense data to a 2D zarr array."""
        data = np.asarray(adata.X, dtype=np.float32)

        group = self._root.create_group(zarr_group)

        n_cells, n_features = data.shape
        group.create_array(
            "data",
            data=data,
            chunks=(chunk_size, n_features),
            shards=(shard_size, n_features),
        )

    def _write_var_sidecar(
        self,
        adata: ad.AnnData,
        feature_space: FeatureSpace,
        zarr_group: str,
    ) -> None:
        """Write var.parquet and remap.parquet for a dataset.

        Requires ``global_feature_uid`` in ``adata.var`` and features to
        already be registered via :meth:`register_features`.
        """
        var_df = pl.from_pandas(adata.var.reset_index())
        if "global_feature_uid" not in var_df.columns:
            raise ValueError(
                "adata.var must have a 'global_feature_uid' column. "
                "Set it before calling add_from_anndata()."
            )

        write_var_df(self._store, zarr_group, var_df)

        if feature_space in self._registry_tables:
            registry_table = self._registry_tables[feature_space]
            remap = build_remap(var_df, registry_table)
            write_remap(self._store, zarr_group, remap)

    # -- Validation ---------------------------------------------------------

    def validate(
        self,
        *,
        check_zarr: bool = True,
        check_remaps: bool = True,
        check_registries: bool = True,
    ) -> list[str]:
        """Validate atlas consistency. Returns a list of error strings.

        Parameters
        ----------
        check_zarr:
            Open each unique zarr group and validate against its spec.
        check_remaps:
            For feature spaces with var_df, validate sidecars and remaps.
        check_registries:
            Check that registry tables exist and global_index is contiguous.
        """
        errors: list[str] = []

        # Schema validation
        for pf in self._pointer_fields.values():
            spec = ZARR_SPECS[pf.feature_space]
            if pf.pointer_kind is not spec.pointer_kind:
                errors.append(
                    f"Field '{pf.field_name}': pointer_kind {pf.pointer_kind.value} "
                    f"doesn't match spec {spec.pointer_kind.value}"
                )

        if check_registries:
            errors.extend(self._validate_registries())

        # Collect unique zarr groups from cell table
        zarr_groups_by_space = self._collect_zarr_groups()

        if check_zarr:
            errors.extend(self._validate_zarr_groups(zarr_groups_by_space))

        if check_remaps:
            errors.extend(self._validate_remaps(zarr_groups_by_space))

        return errors

    def _collect_zarr_groups(self) -> dict[FeatureSpace, set[str]]:
        """Collect unique zarr groups per feature space from the dataset table."""
        result: dict[FeatureSpace, set[str]] = defaultdict(set)
        datasets_df = self._dataset_table.search().to_polars()
        if datasets_df.is_empty():
            return result
        for row in datasets_df.iter_rows(named=True):
            fs = FeatureSpace(row["feature_space"])
            result[fs].add(row["zarr_group"])
        return result

    def list_datasets(self) -> pl.DataFrame:
        """Return a Polars DataFrame of all ingested datasets."""
        return self._dataset_table.search().to_polars()

    def _validate_registries(self) -> list[str]:
        errors: list[str] = []
        for fs, table in self._registry_tables.items():
            df = table.search().select(["uid", "global_index"]).to_polars()
            if df.is_empty():
                continue
            indexed = df.filter(pl.col("global_index").is_not_null())
            if indexed.is_empty():
                errors.append(f"Registry '{fs.value}': no rows have global_index assigned")
                continue
            indices = sorted(indexed["global_index"].to_list())
            expected = list(range(len(indices)))
            if indices != expected:
                errors.append(
                    f"Registry '{fs.value}': global_index is not contiguous 0..{len(indices)-1}"
                )
        return errors

    def _validate_zarr_groups(
        self, zarr_groups_by_space: dict[FeatureSpace, set[str]]
    ) -> list[str]:
        errors: list[str] = []
        for fs, groups in zarr_groups_by_space.items():
            spec = ZARR_SPECS[fs]
            for zg in groups:
                group = self._root[zg]
                group_errors = spec.validate_group(group)
                for e in group_errors:
                    errors.append(f"zarr group '{zg}': {e}")
        return errors

    # TODO: The more I think about it the less I like storing `global_index` anywhere but
    # in the registry table. We can compute it pretty easily when we need to. The risk of
    # drift and contamination from parallel writes to the registry table is just too high.
    def _validate_remaps(
        self, zarr_groups_by_space: dict[FeatureSpace, set[str]]
    ) -> list[str]:
        errors: list[str] = []
        for fs, groups in zarr_groups_by_space.items():
            spec = ZARR_SPECS[fs]
            if not spec.has_var_df:
                continue
            registry = self._registry_tables.get(fs)
            for zg in groups:
                # Validate var_df
                var_df = read_var_df(self._store, zg)
                group = self._root[zg]
                vd_errors = validate_var_df(var_df, spec=spec, group=group, registry_table=registry)
                for e in vd_errors:
                    errors.append(f"var_df '{zg}': {e}")

                # Validate remap
                remap = read_remap(self._store, zg)
                rm_errors = validate_remap(remap, var_df=var_df, registry_table=registry)
                for e in rm_errors:
                    errors.append(f"remap '{zg}': {e}")
        return errors


# ---------------------------------------------------------------------------
# AtlasQuery — fluent query builder
# ---------------------------------------------------------------------------


class AtlasQuery:
    """Fluent query builder for reading cells from a RaggedAtlas."""

    def __init__(self, atlas: RaggedAtlas) -> None:
        self._atlas = atlas
        self._where_clause: str | None = None
        self._limit_n: int | None = None
        self._feature_spaces: list[FeatureSpace] | None = None
        self._layer_overrides: dict[FeatureSpace, list[str]] = {}

    # TODO: We definitely want `search()` as a method because
    # we want to make use of lance's powerful FTS and vector search
    # capabilities.

    def where(self, condition: str) -> "AtlasQuery":
        """Add a SQL WHERE filter (LanceDB syntax)."""
        self._where_clause = condition
        return self

    def limit(self, n: int) -> "AtlasQuery":
        """Limit the number of cells returned."""
        self._limit_n = n
        return self

    def feature_spaces(self, *spaces: FeatureSpace) -> "AtlasQuery":
        """Restrict reconstruction to specific feature spaces."""
        self._feature_spaces = list(spaces)
        return self

    # TODO: Again, can generalize this for sparse and dense arrays
    def layers(self, feature_space: FeatureSpace, names: list[str]) -> "AtlasQuery":
        """Specify which layers to read for a sparse feature space."""
        self._layer_overrides[feature_space] = names
        return self

    # -- Execution ----------------------------------------------------------

    def _build_scanner(self) -> lancedb.table.Table:
        """Build a LanceDB query from the current state."""
        q = self._atlas.cell_table.search()
        if self._where_clause is not None:
            q = q.where(self._where_clause)
        if self._limit_n is not None:
            q = q.limit(self._limit_n)
        return q

    def _active_pointer_fields(self) -> dict[str, PointerFieldInfo]:
        """Return pointer fields filtered by requested feature spaces."""
        pfs = self._atlas._pointer_fields
        if self._feature_spaces is None:
            return pfs
        return {k: v for k, v in pfs.items() if v.feature_space in self._feature_spaces}

    def to_polars(self) -> pl.DataFrame:
        """Execute the query and return a Polars DataFrame of cell metadata."""
        return self._build_scanner().to_polars()

    def to_anndata(self) -> ad.AnnData:
        """Execute the query and reconstruct an AnnData.

        If multiple feature spaces are active, only the first sparse feature
        space is used for X. Use :meth:`to_mudata` for multi-modal.
        """
        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return ad.AnnData()

        active_pfs = self._active_pointer_fields()
        # Pick the first feature space for X
        if not active_pfs:
            return _build_obs_only_anndata(cells_pl)

        # Use first pointer field
        pf = next(iter(active_pfs.values()))
        return self._reconstruct_single_space(cells_pl, pf)

    def to_mudata(self) -> "mu.MuData":
        """Execute the query and return a MuData with one modality per feature space."""
        import mudata as mu

        cells_pl = self._build_scanner().to_polars()
        if cells_pl.is_empty():
            return mu.MuData({})

        active_pfs = self._active_pointer_fields()
        modalities: dict[str, ad.AnnData] = {}
        for pf in active_pfs.values():
            adata = self._reconstruct_single_space(cells_pl, pf)
            if adata.n_obs > 0:
                modalities[pf.feature_space.value] = adata

        return mu.MuData(modalities)

    def to_batches(self, batch_size: int = 1024) -> Iterator[ad.AnnData]:
        """Stream results as AnnData batches.

        Each batch contains up to ``batch_size`` cells. BatchArray readers
        and remap arrays are cached on the atlas for reuse across batches.
        """
        q = self._build_scanner()
        arrow_table = q.to_arrow()
        n_total = arrow_table.num_rows
        if n_total == 0:
            return

        active_pfs = self._active_pointer_fields()
        if not active_pfs:
            # Obs-only batches
            for start in range(0, n_total, batch_size):
                batch_arrow = arrow_table.slice(start, batch_size)
                batch_pl = pl.from_arrow(batch_arrow)
                yield _build_obs_only_anndata(batch_pl)
            return

        pf = next(iter(active_pfs.values()))
        for start in range(0, n_total, batch_size):
            batch_arrow = arrow_table.slice(start, batch_size)
            batch_pl = pl.from_arrow(batch_arrow)
            yield self._reconstruct_single_space(batch_pl, pf)

    # -- Reconstruction internals -------------------------------------------

    def _reconstruct_single_space(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
    ) -> ad.AnnData:
        """Reconstruct an AnnData for a single feature space."""
        spec = ZARR_SPECS[pf.feature_space]
        if pf.pointer_kind is PointerKind.SPARSE:
            return self._reconstruct_sparse(cells_pl, pf, spec)
        else:
            return self._reconstruct_dense(cells_pl, pf, spec)

    def _reconstruct_sparse(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
    ) -> ad.AnnData:
        """Reconstruct sparse data (e.g. gene expression) across zarr groups."""
        # Determine index array name from spec's required_arrays
        if len(spec.required_arrays) != 1:
            raise NotImplementedError(
                f"Sparse reconstruction for feature space '{pf.feature_space.value}' "
                f"is not yet supported (requires {len(spec.required_arrays)} "
                f"primary arrays: {[a.array_name for a in spec.required_arrays]})"
            )
        index_array_name = spec.required_arrays[0].array_name

        cells_pl, groups = _prepare_sparse_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, union_globals, group_remap_to_union, n_features = _load_remaps_and_union(
            self._atlas, groups, spec
        )

        # Determine which layers to read (explicit opt-in only)
        layers_to_read = self._layer_overrides.get(pf.feature_space)
        if layers_to_read is None:
            # Default: read required layers from spec's subgroup
            layers_to_read = []
            for sg in spec.required_subgroups:
                if sg.required_arrays:
                    layers_to_read.extend(a.array_name for a in sg.required_arrays)
            if not layers_to_read:
                raise ValueError(
                    f"No layers specified and spec for '{pf.feature_space.value}' "
                    f"has no required subgroup arrays"
                )

        # Process each zarr group
        all_csrs: dict[str, list[sp.csr_matrix]] = {ln: [] for ln in layers_to_read}
        obs_parts: list[pl.DataFrame] = []

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            starts = group_cells["_start"].to_numpy().astype(np.int64)
            ends = group_cells["_end"].to_numpy().astype(np.int64)
            n_cells_group = len(starts)

            # Batch-read index array via Rust reader
            indices_reader = self._atlas._get_batch_reader(zg, index_array_name)
            flat_indices, lengths = indices_reader.read_ranges(starts, ends)

            # Remap local indices -> union positions
            if zg in group_remap_to_union:
                union_remap = group_remap_to_union[zg]
                union_indices = union_remap[flat_indices.astype(np.intp)]
            else:
                union_indices = flat_indices.astype(np.int32)

            # Build indptr from lengths
            indptr = np.zeros(n_cells_group + 1, dtype=np.int64)
            np.cumsum(lengths, out=indptr[1:])

            # Batch-read each layer
            for ln in layers_to_read:
                layer_reader = self._atlas._get_batch_reader(
                    zg, f"{spec.required_subgroups[0].subgroup_name}/{ln}"
                )
                flat_values, _ = layer_reader.read_ranges(starts, ends)

                csr = sp.csr_matrix(
                    (flat_values, union_indices, indptr),
                    shape=(n_cells_group, n_features),
                )
                all_csrs[ln].append(csr)

            obs_parts.append(group_cells)

        # Stack CSRs
        stacked: dict[str, sp.csr_matrix] = {}
        for ln, csr_list in all_csrs.items():
            if csr_list:
                stacked[ln] = sp.vstack(csr_list, format="csr")

        # Build obs
        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)

        # Build var from registry
        var = self._build_var(pf.feature_space, union_globals)

        # First layer becomes X, rest go to layers
        first_layer = layers_to_read[0]
        X = stacked.get(first_layer)
        extra_layers = {ln: stacked[ln] for ln in layers_to_read[1:] if ln in stacked}

        adata = ad.AnnData(X=X, obs=obs, var=var, layers=extra_layers if extra_layers else None)
        return adata

    def _reconstruct_dense(
        self,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
    ) -> ad.AnnData:
        """Reconstruct dense data (e.g. protein abundance) across zarr groups."""
        cells_pl, groups = _prepare_dense_cells(cells_pl, pf)
        if not groups:
            return ad.AnnData()

        _, union_globals, group_remap_to_union, n_union_features = _load_remaps_and_union(
            self._atlas, groups, spec
        )

        n_total_cells = cells_pl.height
        out = np.zeros((n_total_cells, n_union_features), dtype=np.float32)
        obs_parts: list[pl.DataFrame] = []
        offset = 0

        for zg in groups:
            group_cells = cells_pl.filter(pl.col("_zg") == zg)
            positions = group_cells["_pos"].to_numpy().astype(np.intp)
            n_cells_group = len(positions)

            group = self._atlas._root[zg]
            data_arr = group["data"]
            local_data = data_arr[positions, :]

            if zg in group_remap_to_union:
                union_cols = group_remap_to_union[zg]
                out[offset : offset + n_cells_group][:, union_cols] = local_data
            else:
                n_local_features = local_data.shape[1]
                out[offset : offset + n_cells_group, :n_local_features] = local_data

            obs_parts.append(group_cells)
            offset += n_cells_group

        obs_pl = pl.concat(obs_parts, how="diagonal_relaxed")
        obs = _build_obs_df(obs_pl)
        var = self._build_var(pf.feature_space, union_globals)

        return ad.AnnData(X=out, obs=obs, var=var)

    def _build_var(
        self, feature_space: FeatureSpace, union_globals: np.ndarray
    ) -> pd.DataFrame:
        """Build a var DataFrame from the feature registry."""
        if feature_space not in self._atlas._registry_tables or len(union_globals) == 0:
            return pd.DataFrame(index=pd.RangeIndex(len(union_globals)))

        registry_table = self._atlas._registry_tables[feature_space]
        registry_df = registry_table.search().to_polars()

        # Filter to union globals
        registry_df = registry_df.filter(
            pl.col("global_index").is_in(union_globals.tolist())
        ).sort("global_index")

        var = registry_df.to_pandas()
        if "uid" in var.columns:
            var = var.set_index("uid")
        return var


# ---------------------------------------------------------------------------
# Reconstruction helpers
# ---------------------------------------------------------------------------


def _prepare_sparse_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest sparse pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_start``, ``_end``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["start"].alias("_start"),
        struct_df["end"].alias("_end"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _prepare_dense_cells(
    cells_pl: pl.DataFrame,
    pf: PointerFieldInfo,
) -> tuple[pl.DataFrame, list[str]]:
    """Unnest dense pointer struct, filter empty, return (filtered_df, unique_groups).

    Adds internal columns ``_zg``, ``_pos``.
    """
    col = pf.field_name
    struct_df = cells_pl[col].struct.unnest()
    cells_pl = cells_pl.with_columns(
        struct_df["zarr_group"].alias("_zg"),
        struct_df["position"].alias("_pos"),
    )
    cells_pl = cells_pl.filter(pl.col("_zg") != "")
    groups = cells_pl["_zg"].unique().to_list() if not cells_pl.is_empty() else []
    return cells_pl, groups


def _load_remaps_and_union(
    atlas: "RaggedAtlas",
    groups: list[str],
    spec: ZarrGroupSpec,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], int]:
    """Load remaps for groups, build union feature space.

    Returns (group_remaps, union_globals, group_remap_to_union, n_features).
    """
    group_remaps: dict[str, np.ndarray] = {}
    if spec.has_var_df:
        for zg in groups:
            group_remaps[zg] = atlas._get_remap(zg)

    if group_remaps:
        union_globals, group_remap_to_union = _build_union_feature_space(group_remaps)
        n_features = len(union_globals)
    else:
        union_globals = np.array([], dtype=np.int32)
        group_remap_to_union = {}
        n_features = 0

    return group_remaps, union_globals, group_remap_to_union, n_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_union_feature_space(
    remaps: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Compute union of global indices and per-group local-to-union mappings.

    Parameters
    ----------
    remaps:
        ``{zarr_group: remap_array}`` where ``remap[local_i] = global_index``.

    Returns
    -------
    (union_globals, group_remap_to_union)
        ``union_globals``: sorted array of unique global indices in the union.
        ``group_remap_to_union[zg]``: array where ``arr[local_i]`` is the
        column position in the union-space matrix.
    """
    union_globals = functools.reduce(np.union1d, remaps.values()).astype(np.int32)

    global_to_union = np.empty(union_globals.max() + 1, dtype=np.int32)
    global_to_union[union_globals] = np.arange(len(union_globals), dtype=np.int32)

    group_remap_to_union = {
        group: global_to_union[remap] for group, remap in remaps.items()
    }
    return union_globals, group_remap_to_union


def _build_obs_df(cells_pl: pl.DataFrame) -> pd.DataFrame:
    """Build an obs DataFrame from query results, excluding pointer/internal columns."""
    # Drop struct columns (pointer fields) and internal helper columns
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct and not c.startswith("_")
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return obs


def _build_obs_only_anndata(cells_pl: pl.DataFrame) -> ad.AnnData:
    """Build an AnnData with only obs, no X."""
    keep_cols = [
        c for c in cells_pl.columns
        if cells_pl[c].dtype != pl.Struct
    ]
    obs = cells_pl.select(keep_cols).to_pandas()
    if "uid" in obs.columns:
        obs = obs.set_index("uid")
    return ad.AnnData(obs=obs)
