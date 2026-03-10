import dataclasses

import anndata as ad
import duckdb
import lancedb
import mudata as mu
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp
from lancedb.query import FullTextOperator, MatchQuery

from epiblast.multimodal_rag.ingestion_utils import lookup_gene_indices_from_table
from epiblast.multimodal_rag.schema import (
    CHROMATIN_ACCESSIBILITY_TABLE,
    FEATURE_SPACE_TO_TABLE,
    GENE_EXPRESSION_TABLE,
    IMAGE_FEATURE_VECTORS_TABLE,
    IMAGE_TILES_TABLE,
    PROTEIN_ABUNDANCE_TABLE,
)

# Scalar-indexed columns on modality tables that support exact-match filtering.
_SCALAR_FIELDS: dict[str, type] = {
    "dataset_uid": str,
    "cell_line": str,
    "assay": str,
    "cell_type": str,
    "development_stage": str,
    "disease": str,
    "organism": str,
    "tissue": str,
    "is_control": bool,
}

_TABLE_TO_FEATURE_SPACE = {v: k for k, v in FEATURE_SPACE_TO_TABLE.items()}

# Columns that are data-specific (not metadata) for each feature table.
# These are combined with metadata columns when scanning.
_DATA_COLUMNS: dict[str, list[str]] = {
    GENE_EXPRESSION_TABLE: ["gene_indices", "counts"],
    PROTEIN_ABUNDANCE_TABLE: ["protein_indices", "counts"],
    CHROMATIN_ACCESSIBILITY_TABLE: [
        "chromosome_indices",
        "fragment_start_positions",
        "fragment_lengths",
    ],
    IMAGE_FEATURE_VECTORS_TABLE: [
        "feature_values",
    ],
    IMAGE_TILES_TABLE: [
        "image",
        "image_shape",
        "image_dtype",
        "channel_names",
    ],
}

# Metadata columns present on all denormalized data tables (from _CellMetadataMixin).
_METADATA_COLUMNS = [
    "cell_uid",
    "dataset_uid",
    "assay",
    "cell_line",
    "development_stage",
    "disease",
    "organism",
    "cell_type",
    "tissue",
    "additional_metadata",
    "coord_t",
    "coord_z",
    "coord_y",
    "coord_x",
    "time_unit",
    "spatial_unit",
    "is_control",
    "chemical_perturbation_uid",
    "chemical_perturbation_concentration",
    "chemical_perturbation_additional_metadata",
    "genetic_perturbation_gene_index",
    "genetic_perturbation_method",
    "genetic_perturbation_concentration",
    "genetic_perturbation_additional_metadata",
    "perturbation_search_string",
]

# Metadata columns used for obs (everything except search string)
_OBS_METADATA_COLUMNS = [c for c in _METADATA_COLUMNS if c != "perturbation_search_string"]


@dataclasses.dataclass
class AtlasQuery:
    """Structured query for filtering cells across multi-table schema."""

    # Scalar-indexed filters (exact match)
    dataset_uid: str | None = None
    cell_line: str | None = None
    assay: str | None = None
    cell_type: str | None = None
    development_stage: str | None = None
    disease: str | None = None
    organism: str | None = None
    tissue: str | None = None
    is_control: bool | None = None

    # Perturbation label filters (FTS index on perturbation_search_string)
    gene_names: list[str] | None = None
    ensembl_ids: list[str] | None = None
    perturbation_method: str | None = None
    chemical_perturbation_uid: str | None = None

    # Post-filter on concentration (applied via duckdb after LanceDB query)
    genetic_perturbation_concentration_min: float | None = None
    genetic_perturbation_concentration_max: float | None = None
    chemical_perturbation_concentration_min: float | None = None
    chemical_perturbation_concentration_max: float | None = None

    # Which modality tables to query (defaults to all)
    feature_spaces: list[str] | None = None


def build_search_query(
    db: lancedb.DBConnection,
    query: AtlasQuery,
    operator: FullTextOperator = FullTextOperator.OR,
) -> MatchQuery | None:
    """Build a full-text MatchQuery from perturbation filters."""
    tokens: list[str] = []
    if query.gene_names:
        genes_table = db.open_table("genes")
        gene_index_map = lookup_gene_indices_from_table(
            genes_table, organism=query.organism or "human", gene_names=query.gene_names
        )
        for name in query.gene_names:
            if name in gene_index_map:
                tokens.append(f"GENE_ID:{gene_index_map[name]}")

    if query.ensembl_ids:
        genes_table = db.open_table("genes")
        gene_index_map = lookup_gene_indices_from_table(
            genes_table, organism=query.organism or "human", ensembl_ids=query.ensembl_ids
        )
        for eid in query.ensembl_ids:
            if eid in gene_index_map:
                tokens.append(f"GENE_ID:{gene_index_map[eid]}")

    if query.perturbation_method:
        tokens.append(f"METHOD:{query.perturbation_method}")

    if query.chemical_perturbation_uid:
        tokens.append(f"SM:{query.chemical_perturbation_uid}")

    if not tokens:
        return None

    return MatchQuery(
        " ".join(tokens),
        column="perturbation_search_string",
        operator=operator,
    )


def build_where_clause(query: AtlasQuery) -> str | None:
    """Build a SQL WHERE clause from scalar filters.

    Returns None if no filters are set.
    """
    conditions = []

    for field_name, field_type in _SCALAR_FIELDS.items():
        value = getattr(query, field_name)
        if value is None:
            continue
        if field_type is bool:
            conditions.append(f"{field_name} = {str(value).lower()}")
        else:
            escaped = value.replace("'", "''")
            conditions.append(f"{field_name} = '{escaped}'")

    if not conditions:
        return None
    return " AND ".join(conditions)


def execute_query(
    db: lancedb.DBConnection,
    table_name: str,
    query: AtlasQuery,
    max_records: int | None = None,
    select_cols: list[str] | None = None,
) -> pa.Table:
    """Execute a filtered query directly against a denormalized data table.

    Returns a PyArrow Table with both metadata and data columns.
    """
    search_query = build_search_query(db, query)
    where_clause = build_where_clause(query)

    table = db.open_table(table_name)

    if search_query is not None:
        lance_query = table.search(search_query)
    else:
        lance_query = table.search()

    if where_clause is not None:
        lance_query = lance_query.where(where_clause)

    if select_cols:
        lance_query = lance_query.select(select_cols)

    if max_records is not None:
        lance_query = lance_query.limit(max_records)

    arrow_table = lance_query.to_arrow()
    arrow_table = filter_by_concentration(arrow_table, query)
    return arrow_table


def filter_by_concentration(cells: pa.Table, query: AtlasQuery) -> pa.Table:
    """Post-filter cells by perturbation concentration ranges using duckdb unnest.

    Uses ANY semantics: a cell is kept if at least one perturbation in its list
    falls within the specified concentration range.
    """
    has_genetic = (
        query.genetic_perturbation_concentration_min is not None
        or query.genetic_perturbation_concentration_max is not None
    )
    has_chemical = (
        query.chemical_perturbation_concentration_min is not None
        or query.chemical_perturbation_concentration_max is not None
    )
    if not has_genetic and not has_chemical:
        return cells

    con = duckdb.connect()
    con.register("cells", cells)
    keep_conditions = []

    if has_genetic:
        genetic_conditions = []
        if query.genetic_perturbation_concentration_min is not None:
            genetic_conditions.append(f"conc >= {query.genetic_perturbation_concentration_min}")
        if query.genetic_perturbation_concentration_max is not None:
            genetic_conditions.append(f"conc <= {query.genetic_perturbation_concentration_max}")
        genetic_where = " AND ".join(genetic_conditions)
        keep_conditions.append(
            f"""cell_uid IN (
                SELECT cell_uid FROM (
                    SELECT cell_uid, unnest(genetic_perturbation_concentration) AS conc
                    FROM cells
                    WHERE genetic_perturbation_concentration IS NOT NULL
                ) WHERE {genetic_where}
            )"""
        )

    if has_chemical:
        chemical_conditions = []
        if query.chemical_perturbation_concentration_min is not None:
            chemical_conditions.append(f"conc >= {query.chemical_perturbation_concentration_min}")
        if query.chemical_perturbation_concentration_max is not None:
            chemical_conditions.append(f"conc <= {query.chemical_perturbation_concentration_max}")
        chemical_where = " AND ".join(chemical_conditions)
        keep_conditions.append(
            f"""cell_uid IN (
                SELECT cell_uid FROM (
                    SELECT cell_uid, unnest(chemical_perturbation_concentration) AS conc
                    FROM cells
                    WHERE chemical_perturbation_concentration IS NOT NULL
                ) WHERE {chemical_where}
            )"""
        )

    full_where = " AND ".join(keep_conditions)
    result = con.execute(f"SELECT * FROM cells WHERE {full_where}").fetch_arrow_table()
    con.close()
    return result


def _unpack_arrow_binary(arrow_col, dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """Return (offsets_in_elements, flat_data) from a binary Arrow column.

    Handles both regular and large binary Arrow types by detecting whether
    offsets are 32-bit or 64-bit. The returned offsets are converted from byte
    offsets to element offsets using ``dtype.itemsize``.
    """
    offsets_dtype = np.int64 if "Large" in type(arrow_col).__name__ else np.int32
    byte_offsets = np.frombuffer(arrow_col.buffers()[1], dtype=offsets_dtype)
    flat_data = np.frombuffer(arrow_col.buffers()[2], dtype=dtype)
    element_offsets = byte_offsets // dtype.itemsize
    return element_offsets, flat_data


# ---------------------------------------------------------------------------
# Per-feature-space reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_gene_expression(
    group_df: pl.DataFrame,
    measured_indices: np.ndarray,
    gene_ensembl_arr: np.ndarray,
    max_gene_index: int,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict]:
    """Reconstruct a sparse CSR matrix for gene expression data."""
    n_features = len(measured_indices)
    n_cells = group_df.height

    global_to_local = np.empty(max_gene_index + 1, dtype=np.int32)
    global_to_local[measured_indices] = np.arange(n_features, dtype=np.int32)

    indices_arrow = group_df["gene_indices"].to_arrow()
    values_arrow = group_df["counts"].to_arrow()

    idx_offsets, all_global_indices = _unpack_arrow_binary(indices_arrow, np.dtype(np.int32))
    val_offsets, all_values = _unpack_arrow_binary(values_arrow, np.dtype(np.float32))

    indptr = idx_offsets[: n_cells + 1].astype(np.int64)
    all_global_indices = all_global_indices[: indptr[-1]]
    all_values = all_values[: indptr[-1]]
    all_local_indices = global_to_local[all_global_indices]

    X = sp.csr_matrix(
        (all_values, all_local_indices, indptr),
        shape=(n_cells, n_features),
    )

    measured_gene_ensembl = gene_ensembl_arr[measured_indices]
    var = pd.DataFrame(
        {"gene_id": measured_gene_ensembl},
        index=measured_gene_ensembl,
    )
    return X, var, {}


def _reconstruct_image_feature_vectors(
    group_df: pl.DataFrame,
    measured_indices: np.ndarray,
    image_feature_names: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """Reconstruct a dense float32 matrix for image features."""
    n_features = len(measured_indices)
    n_cells = group_df.height
    feature_names = image_feature_names[measured_indices]

    values_arrow = group_df["feature_values"].to_arrow()
    offsets, all_values = _unpack_arrow_binary(values_arrow, np.dtype(np.float32))

    # Dense features: all cells have the same number of features, so reshape
    start = offsets[0]
    end = offsets[n_cells]
    X = all_values[start:end].reshape(n_cells, n_features)

    var = pd.DataFrame(index=feature_names)
    return X, var, {}


def _reconstruct_chromatin_accessibility(
    group_df: pl.DataFrame,
    measured_indices: np.ndarray,
    chrom_cumulative_offsets: np.ndarray,
    total_genome_size: int,
) -> tuple[None, pd.DataFrame, dict]:
    """Reconstruct ATAC-seq fragments as a SnapATAC2-style CSR matrix.

    Encodes each fragment as a single entry in a CSR matrix stored in
    ``uns["fragment_paired"]``:
    - Row = cell
    - Column = cumulative genomic position (``chrom_offset + start``)
    - Value = fragment length (``end - start``)

    May contain duplicate column indices (multiple fragments at the same
    start position), matching the SnapATAC2 convention.
    """
    n_cells = group_df.height

    idx_offsets, all_chrom_idx = _unpack_arrow_binary(
        group_df["chromosome_indices"].to_arrow(), np.dtype(np.int32)
    )
    start_offsets, all_starts = _unpack_arrow_binary(
        group_df["fragment_start_positions"].to_arrow(), np.dtype(np.int32)
    )
    length_offsets, all_lengths = _unpack_arrow_binary(
        group_df["fragment_lengths"].to_arrow(), np.dtype(np.int32)
    )

    # Need to use int64 because the cumulative genomic positions often exceed 2^31
    # int32 for per-chromosome indices should be OK though
    indptr = idx_offsets[: n_cells + 1].astype(np.int64)
    n_total = indptr[-1]
    all_chrom_idx = all_chrom_idx[:n_total]
    all_starts = all_starts[:n_total]
    all_lengths = all_lengths[:n_total]

    # Encode (chrom, start) as a single column index
    all_col_indices = chrom_cumulative_offsets[all_chrom_idx].astype(np.int64) + all_starts
    all_lengths = all_lengths.astype(np.int64)

    fragment_matrix = sp.csr_matrix(
        (all_lengths, all_col_indices, indptr),
        shape=(n_cells, total_genome_size),
    )

    return None, pd.DataFrame(), {"fragment_paired": fragment_matrix}


def _reconstruct_protein_abundance(
    group_df: pl.DataFrame,
    measured_indices: np.ndarray,
    protein_uniprot_arr: np.ndarray,
    protein_name_arr: np.ndarray,
    protein_gene_name_arr: np.ndarray,
    max_protein_index: int,
) -> tuple[sp.csr_matrix, pd.DataFrame, dict]:
    """Reconstruct a sparse CSR matrix for protein abundance data.

    Proteins have their own index space (protein_index in ProteinSchema),
    separate from gene_index.
    """
    n_features = len(measured_indices)
    n_cells = group_df.height

    global_to_local = np.empty(max_protein_index + 1, dtype=np.int32)
    global_to_local[measured_indices] = np.arange(n_features, dtype=np.int32)

    indices_arrow = group_df["protein_indices"].to_arrow()
    values_arrow = group_df["counts"].to_arrow()

    idx_offsets, all_global_indices = _unpack_arrow_binary(indices_arrow, np.dtype(np.int32))
    val_offsets, all_values = _unpack_arrow_binary(values_arrow, np.dtype(np.float32))

    indptr = idx_offsets[: n_cells + 1].astype(np.int64)
    all_global_indices = all_global_indices[: indptr[-1]]
    all_values = all_values[: indptr[-1]]
    all_local_indices = global_to_local[all_global_indices]

    X = sp.csr_matrix(
        (all_values, all_local_indices, indptr),
        shape=(n_cells, n_features),
    )

    measured_uniprot = protein_uniprot_arr[measured_indices]
    measured_names = protein_name_arr[measured_indices]
    measured_gene_names = protein_gene_name_arr[measured_indices]
    var = pd.DataFrame(
        {
            "uniprot_id": measured_uniprot,
            "protein_name": measured_names,
            "gene_name": measured_gene_names,
        },
        index=measured_uniprot,
    )
    return X, var, {}


_FEATURE_SPACE_RECONSTRUCTORS = {
    "gene_expression": _reconstruct_gene_expression,
    "protein_abundance": _reconstruct_protein_abundance,
    "image_feature_vectors": _reconstruct_image_feature_vectors,
    "chromatin_accessibility": _reconstruct_chromatin_accessibility,
}


# ---------------------------------------------------------------------------
# Reference data lookups
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _FeatureLookups:
    """Lazily-loaded reference arrays used during AnnData reconstruction."""

    db: lancedb.DBConnection
    gene_names_arr: np.ndarray
    gene_ensembl_arr: np.ndarray
    max_gene_index: int

    _image_feature_names: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _protein_uniprot_arr: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _protein_name_arr: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _protein_gene_name_arr: np.ndarray | None = dataclasses.field(default=None, repr=False)
    _max_protein_index: int | None = dataclasses.field(default=None, repr=False)
    _chrom_offsets_cache: dict[tuple[int, ...], tuple[np.ndarray, int]] = dataclasses.field(
        default_factory=dict, repr=False
    )

    @classmethod
    def from_db(cls, db: lancedb.DBConnection) -> "_FeatureLookups":
        genes_df = db.open_table("genes").search().to_polars()
        max_gene_index = genes_df["gene_index"].max()
        gene_names_arr = np.empty(max_gene_index + 1, dtype=object)
        gene_ensembl_arr = np.empty(max_gene_index + 1, dtype=object)
        indices = genes_df["gene_index"].to_numpy()
        gene_names_arr[indices] = genes_df["gene_name"].to_list()
        gene_ensembl_arr[indices] = genes_df["ensembl_id"].to_list()
        # Fill gene_ensembl_arr with gene names where it is None
        missing_mask = gene_ensembl_arr == None  # noqa: E711
        gene_ensembl_arr[missing_mask] = gene_names_arr[missing_mask]

        return cls(
            db=db,
            gene_names_arr=gene_names_arr,
            gene_ensembl_arr=gene_ensembl_arr,
            max_gene_index=max_gene_index,
        )

    @property
    def image_feature_names(self) -> np.ndarray:
        if self._image_feature_names is None:
            img_df = (
                self.db.open_table("image_features")
                .search()
                .select(["feature_index", "feature_name"])
                .to_polars()
            )
            max_idx = img_df["feature_index"].max()
            arr = np.empty(max_idx + 1, dtype=object)
            arr[img_df["feature_index"].to_numpy()] = img_df["feature_name"].to_list()
            self._image_feature_names = arr
        return self._image_feature_names

    def _load_protein_data(self) -> None:
        """Lazily load protein reference arrays from the proteins table."""
        if self._protein_uniprot_arr is not None:
            return
        prot_df = (
            self.db.open_table("proteins")
            .search()
            .select(["protein_index", "uniprot_id", "protein_name", "gene_name"])
            .to_polars()
        )
        max_idx = prot_df["protein_index"].max()
        uniprot_arr = np.empty(max_idx + 1, dtype=object)
        name_arr = np.empty(max_idx + 1, dtype=object)
        gene_name_arr = np.empty(max_idx + 1, dtype=object)
        indices = prot_df["protein_index"].to_numpy()
        uniprot_arr[indices] = prot_df["uniprot_id"].to_list()
        name_arr[indices] = prot_df["protein_name"].to_list()
        gene_name_arr[indices] = prot_df["gene_name"].to_list()
        self._protein_uniprot_arr = uniprot_arr
        self._protein_name_arr = name_arr
        self._protein_gene_name_arr = gene_name_arr
        self._max_protein_index = max_idx

    def _load_chrom_data(self, measured_indices: np.ndarray) -> tuple[np.ndarray, int]:
        """Load chromosome sizes and compute cumulative offsets for the given chromosome indices.

        Only loads sizes for the chromosomes actually referenced by the dataset
        (via ``measured_indices``), so the result is correct regardless of how
        many organisms or assemblies exist in the table.
        """
        cache_key = tuple(int(i) for i in measured_indices)
        if cache_key in self._chrom_offsets_cache:
            return self._chrom_offsets_cache[cache_key]
        idx_list = ", ".join(str(int(i)) for i in measured_indices)
        chrom_df = (
            self.db.open_table("chromosomes")
            .search()
            .where(f"chromosome_index IN ({idx_list})")
            .select(["chromosome_index", "chromosome_size"])
            .to_polars()
        )
        assert chrom_df.height > 0, (
            "No chromosomes found for the dataset's measured_feature_indices"
        )
        max_idx = chrom_df["chromosome_index"].max()
        sizes = np.zeros(max_idx + 1, dtype=np.int64)
        sizes[chrom_df["chromosome_index"].to_numpy()] = chrom_df["chromosome_size"].to_numpy()
        # cumulative_offsets[i] = sum of sizes of chromosomes 0..i-1
        cumulative_offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)])
        total_genome_size = int(cumulative_offsets[-1])
        self._chrom_offsets_cache[cache_key] = (cumulative_offsets, total_genome_size)
        return cumulative_offsets, total_genome_size

    def reconstructor_kwargs(self, feature_space: str, measured_indices: np.ndarray) -> dict:
        """Return the extra keyword arguments needed by the given reconstructor."""
        if feature_space == "gene_expression":
            return {
                "gene_ensembl_arr": self.gene_ensembl_arr,
                "max_gene_index": self.max_gene_index,
            }
        if feature_space == "protein_abundance":
            self._load_protein_data()
            return {
                "protein_uniprot_arr": self._protein_uniprot_arr,
                "protein_name_arr": self._protein_name_arr,
                "protein_gene_name_arr": self._protein_gene_name_arr,
                "max_protein_index": self._max_protein_index,
            }
        if feature_space == "image_feature_vectors":
            return {"image_feature_names": self.image_feature_names}
        if feature_space == "chromatin_accessibility":
            cumulative_offsets, total_genome_size = self._load_chrom_data(measured_indices)
            return {
                "chrom_cumulative_offsets": cumulative_offsets,
                "total_genome_size": total_genome_size,
            }
        raise ValueError(f"Unknown feature_space '{feature_space}'")


# ---------------------------------------------------------------------------
# Obs construction
# ---------------------------------------------------------------------------


def _build_obs(
    group_df: pl.DataFrame,
    gene_names_arr: np.ndarray,
) -> pd.DataFrame:
    """Build the obs DataFrame from metadata columns.

    Resolves ``genetic_perturbation_gene_index`` lists to comma-separated
    gene name strings using a vectorized explode/lookup/agg approach.
    """
    available_cols = [c for c in _OBS_METADATA_COLUMNS if c in group_df.columns]
    obs_pl = group_df.select(available_cols)

    if "genetic_perturbation_gene_index" in obs_pl.columns:
        col = obs_pl["genetic_perturbation_gene_index"]
        # Vectorized: explode, lookup via numpy, re-aggregate
        temp = pl.DataFrame(
            {
                "row_nr": pl.arange(0, obs_pl.height, eager=True),
                "idx_list": col,
            }
        )
        exploded = temp.explode("idx_list").filter(pl.col("idx_list").is_not_null())
        if exploded.height > 0:
            looked_up = gene_names_arr[exploded["idx_list"].to_numpy().astype(np.int64)]
            exploded = exploded.with_columns(pl.Series("gene_name", looked_up))
            reagg = exploded.group_by("row_nr", maintain_order=True).agg(
                pl.col("gene_name").str.concat(",").alias("gene_names_str")
            )
            mapping = dict(
                zip(reagg["row_nr"].to_list(), reagg["gene_names_str"].to_list(), strict=False)
            )
            resolved = pl.Series(
                "genetic_perturbation_gene_index",
                [mapping.get(i) for i in range(obs_pl.height)],
                dtype=pl.String,
            )
        else:
            resolved = pl.Series(
                "genetic_perturbation_gene_index",
                [None] * obs_pl.height,
                dtype=pl.String,
            )
        obs_pl = obs_pl.with_columns(resolved)

    return obs_pl.to_pandas().set_index("cell_uid")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def create_anndatas_from_query(
    db: lancedb.DBConnection,
    query: AtlasQuery,
    max_records: int | None = None,
) -> dict[str, dict[str, ad.AnnData | mu.MuData]]:
    """Query denormalized data tables directly and reconstruct AnnData/MuData.

    Returns a nested dictionary keyed by ``assay`` → ``dataset_uid``.
    Datasets with a single feature space produce an :class:`~anndata.AnnData`;
    datasets with multiple feature spaces (e.g. paired RNA + protein) produce
    a :class:`~mudata.MuData` whose modalities are keyed by feature space.
    """
    lookups = _FeatureLookups.from_db(db)

    feature_spaces_to_query = list(FEATURE_SPACE_TO_TABLE.keys())
    if query.feature_spaces:
        feature_spaces_to_query = [
            fs for fs in feature_spaces_to_query if fs in query.feature_spaces
        ]

    # Collect all dataset_uids across tables so we can batch-load dataset metadata.
    all_dataset_uids: set[str] = set()
    table_results: dict[str, pl.DataFrame] = {}

    for feature_space in feature_spaces_to_query:
        table_name = FEATURE_SPACE_TO_TABLE[feature_space]
        arrow_table = execute_query(db, table_name, query, max_records)
        if arrow_table.num_rows == 0:
            continue
        table_results[feature_space] = pl.from_arrow(arrow_table)
        all_dataset_uids.update(arrow_table.column("dataset_uid").to_pylist())

    if not table_results:
        return {}

    # Load dataset metadata for measured_feature_indices lookup
    uid_list = ", ".join(f"'{uid}'" for uid in all_dataset_uids)
    datasets_df = (
        db.open_table("datasets")
        .search()
        .where(f"dataset_uid IN ({uid_list})")
        .select(["dataset_uid", "measured_feature_indices", "feature_space"])
        .to_polars()
    )

    dataset_measured: dict[tuple[str, str], bytes] = {}
    for row in datasets_df.iter_rows(named=True):
        key = (row["dataset_uid"], row["feature_space"])
        dataset_measured[key] = row["measured_feature_indices"]

    per_dataset: dict[str, dict[str, dict[str, ad.AnnData]]] = {}

    for feature_space, result_pl in table_results.items():
        for (dataset_uid,), group_df in result_pl.group_by(["dataset_uid"]):
            key = (dataset_uid, feature_space)
            if key not in dataset_measured:
                continue
            measured_indices = np.frombuffer(dataset_measured[key], dtype=np.int32)
            assay = group_df["assay"][0]

            reconstruct = _FEATURE_SPACE_RECONSTRUCTORS.get(feature_space)
            if reconstruct is None:
                raise ValueError(
                    f"Unknown feature_space '{feature_space}' for dataset {dataset_uid}"
                )

            extra_kwargs = lookups.reconstructor_kwargs(feature_space, measured_indices)
            X, var, obsm = reconstruct(group_df, measured_indices, **extra_kwargs)
            obs = _build_obs(group_df, lookups.gene_names_arr)

            adata = ad.AnnData(obs=obs, var=var, obsm=obsm)
            if X is not None:
                adata.X = X
            per_dataset.setdefault(assay, {}).setdefault(dataset_uid, {})[feature_space] = adata

    # Collapse single-feature-space datasets to AnnData,
    # wrap multi-feature-space datasets in MuData.
    result: dict[str, dict[str, ad.AnnData | mu.MuData]] = {}
    for assay, datasets in per_dataset.items():
        for dataset_uid, feature_spaces_map in datasets.items():
            if len(feature_spaces_map) == 1:
                data = next(iter(feature_spaces_map.values()))
            else:
                data = mu.MuData(feature_spaces_map)
            result.setdefault(assay, {})[dataset_uid] = data

    return result


def load_image(db: lancedb.DBConnection, cell_uid: str) -> np.ndarray | None:
    """Load the image for a single cell by cell_uid.

    Reconstructs the numpy array using the stored shape and dtype metadata.
    Returns None if the cell is not found or has no blob.
    """
    image_table = db.open_table(IMAGE_TILES_TABLE)
    escaped = cell_uid.replace("'", "''")
    df = (
        image_table.search()
        .where(f"cell_uid = '{escaped}'")
        .select(["image", "image_shape", "image_dtype"])
        .limit(1)
        .to_polars()
    )
    if df.height == 0:
        return None
    blob = df["image"][0]
    if blob is None:
        return None
    shape = df["image_shape"][0]
    dtype = df["image_dtype"][0]
    return np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
