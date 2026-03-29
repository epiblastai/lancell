"""PerturbationQuery: domain-specific query builder for the multimodal perturbation atlas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from lancell.query import AtlasQuery
from lancell.util import sql_escape

if TYPE_CHECKING:
    import lancedb

    from lancell.atlas import RaggedAtlas

# Prefix tokens used in perturbation_search_string (must match schema.py)
_PREFIX_GENETIC = "GP"
_PREFIX_SMALL_MOLECULE = "SM"
_PREFIX_BIOLOGIC = "BIO"

# FK table names (must match schema.py FK_TABLE_SCHEMAS keys)
_TABLE_GENETIC = "genetic_perturbations"
_TABLE_SMALL_MOLECULE = "small_molecules"
_TABLE_BIOLOGIC = "biologic_perturbations"
_TABLE_PUBLICATIONS = "publications"
_TABLE_DATASETS = "datasets"
_TABLE_DATASET_PERTURBATION_INDEX = "dataset_perturbation_index"


def _open_table(atlas: RaggedAtlas, name: str) -> "lancedb.table.Table":
    return atlas.db.open_table(name)


def _query_uids(atlas: RaggedAtlas, table_name: str, where: str) -> list[str]:
    """Query a FK table and return matching UIDs."""
    rows = (
        _open_table(atlas, table_name)
        .search()
        .where(where, prefilter=True)
        .select(["uid"])
        .to_polars()
    )
    return rows["uid"].to_list()


def _perturbation_where(uids: list[str], prefix: str) -> str:
    """Build a WHERE clause matching perturbation_search_string for the given UIDs."""
    if not uids:
        # Impossible condition — no rows match
        return "1 = 0"
    parts = [
        f"perturbation_search_string LIKE '%{prefix}:{sql_escape(uid)}%'"
        for uid in uids
    ]
    return "(" + " OR ".join(parts) + ")"


# ---------------------------------------------------------------------------
# Perturbation metadata enrichment (standalone utility)
# ---------------------------------------------------------------------------


def resolve_perturbation_metadata(
    cells_df: pl.DataFrame,
    db: "lancedb.DBConnection",
    *,
    genetic_columns: list[str] | None = None,
    small_molecule_columns: list[str] | None = None,
    biologic_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Resolve perturbation UIDs in a cells DataFrame to full metadata.

    Adds columns from each perturbation FK table, prefixed by type, as list
    columns parallel to ``perturbation_uids``.

    Parameters
    ----------
    cells_df:
        A Polars DataFrame with ``perturbation_uids`` and ``perturbation_types``.
    db:
        LanceDB connection (``atlas.db``).
    genetic_columns:
        Columns to fetch from ``genetic_perturbations``. Defaults to
        ``["uid", "intended_gene_name", "perturbation_type"]``.
    small_molecule_columns:
        Columns to fetch from ``small_molecules``. Defaults to
        ``["uid", "name", "smiles"]``.
    biologic_columns:
        Columns to fetch from ``biologic_perturbations``. Defaults to
        ``["uid", "biologic_name", "biologic_type"]``.

    Returns
    -------
    pl.DataFrame
        The input DataFrame with additional list columns for each perturbation
        type's metadata.
    """
    if "perturbation_uids" not in cells_df.columns:
        return cells_df
    if "perturbation_types" not in cells_df.columns:
        return cells_df

    if genetic_columns is None:
        genetic_columns = ["uid", "intended_gene_name", "perturbation_type"]
    if small_molecule_columns is None:
        small_molecule_columns = ["uid", "name", "smiles"]
    if biologic_columns is None:
        biologic_columns = ["uid", "biologic_name", "biologic_type"]

    # Collect all unique UIDs grouped by type
    type_uids: dict[str, set[str]] = {}
    for row in cells_df.select("perturbation_uids", "perturbation_types").iter_rows():
        uids_val, types_val = row
        if uids_val is None or types_val is None:
            continue
        for uid, ptype in zip(uids_val, types_val):
            type_uids.setdefault(ptype, set()).add(uid)

    # Batch-fetch each FK table
    lookup: dict[str, dict[str, dict]] = {}  # {type: {uid: {col: val}}}

    def _fetch(table_name: str, ptype: str, columns: list[str]) -> None:
        uids = type_uids.get(ptype, set())
        if not uids:
            return
        table = db.open_table(table_name)
        uid_list = list(uids)
        # Batch in chunks of 500
        frames: list[pl.DataFrame] = []
        for i in range(0, len(uid_list), 500):
            batch = uid_list[i : i + 500]
            in_clause = ", ".join(f"'{sql_escape(u)}'" for u in batch)
            df = (
                table.search()
                .where(f"uid IN ({in_clause})", prefilter=True)
                .select(columns)
                .to_polars()
            )
            frames.append(df)
        if frames:
            all_df = pl.concat(frames)
            uid_map: dict[str, dict] = {}
            for record in all_df.iter_rows(named=True):
                uid_map[record["uid"]] = record
            lookup[ptype] = uid_map

    _fetch(_TABLE_GENETIC, "genetic_perturbation", genetic_columns)
    _fetch(_TABLE_SMALL_MOLECULE, "small_molecule", small_molecule_columns)
    _fetch(_TABLE_BIOLOGIC, "biologic_perturbation", biologic_columns)

    # Build enrichment columns: for each perturbation metadata field, create a
    # list column parallel to perturbation_uids
    all_meta_cols: dict[str, list] = {}
    # Discover which columns we need
    meta_fields: set[str] = set()
    for uid_map in lookup.values():
        for record in uid_map.values():
            meta_fields.update(k for k in record if k != "uid")
            break  # all records in a table have the same keys

    for field in sorted(meta_fields):
        col_name = f"perturbation_{field}"
        all_meta_cols[col_name] = []

    n_rows = cells_df.height
    uids_series = cells_df["perturbation_uids"]
    types_series = cells_df["perturbation_types"]

    for i in range(n_rows):
        uids_val = uids_series[i]
        types_val = types_series[i]
        row_meta: dict[str, list] = {k: [] for k in all_meta_cols}

        if uids_val is not None and types_val is not None:
            for uid, ptype in zip(uids_val, types_val):
                uid_map = lookup.get(ptype, {})
                record = uid_map.get(uid, {})
                for field in sorted(meta_fields):
                    col_name = f"perturbation_{field}"
                    row_meta[col_name].append(record.get(field))

        for col_name, values in row_meta.items():
            all_meta_cols[col_name].append(values if values else None)

    for col_name, col_data in all_meta_cols.items():
        cells_df = cells_df.with_columns(pl.Series(col_name, col_data))

    return cells_df


# ---------------------------------------------------------------------------
# PerturbationQuery
# ---------------------------------------------------------------------------


class PerturbationQuery(AtlasQuery):
    """Fluent query builder with perturbation-aware lookup methods.

    Extends :class:`AtlasQuery` with methods that resolve human-readable
    identifiers (gene names, compound names, accessions) into the appropriate
    foreign-key UIDs and filter cells accordingly.
    """

    def __init__(self, atlas: "RaggedAtlas") -> None:
        super().__init__(atlas)
        self._enrich_perturbations: bool = False
        self._genetic_columns: list[str] | None = None
        self._small_molecule_columns: list[str] | None = None
        self._biologic_columns: list[str] | None = None

    # -- Internal helpers ---------------------------------------------------

    def _and_where(self, condition: str) -> None:
        """AND a condition onto the existing WHERE clause."""
        if self._where_clause is None:
            self._where_clause = condition
        else:
            self._where_clause = f"({self._where_clause}) AND ({condition})"

    # -- Perturbation lookup methods ----------------------------------------

    def by_gene(
        self,
        name: str | None = None,
        *,
        ensembl_id: str | None = None,
        perturbation_type: str | None = None,
    ) -> "PerturbationQuery":
        """Filter to cells with a genetic perturbation targeting a gene.

        Parameters
        ----------
        name:
            Gene name to match against ``intended_gene_name``.
        ensembl_id:
            Ensembl gene ID to match against ``intended_ensembl_gene_id``.
        perturbation_type:
            Optional filter on ``perturbation_type`` (e.g. ``"knockout"``,
            ``"knockdown"``, ``"activation"``).
        """
        if name is None and ensembl_id is None:
            raise ValueError("At least one of name or ensembl_id must be provided")

        clauses: list[str] = []
        if name is not None:
            clauses.append(f"intended_gene_name = '{sql_escape(name)}'")
        if ensembl_id is not None:
            clauses.append(f"intended_ensembl_gene_id = '{sql_escape(ensembl_id)}'")
        if perturbation_type is not None:
            clauses.append(f"perturbation_type = '{sql_escape(perturbation_type)}'")

        where = " AND ".join(clauses)
        uids = _query_uids(self._atlas, _TABLE_GENETIC, where)
        self._and_where(_perturbation_where(uids, _PREFIX_GENETIC))
        return self

    def by_compound(
        self,
        *,
        name: str | None = None,
        smiles: str | None = None,
        pubchem_cid: int | None = None,
    ) -> "PerturbationQuery":
        """Filter to cells treated with a small molecule.

        At least one identifier must be provided.

        Parameters
        ----------
        name:
            Common compound name to match against ``name``.
        smiles:
            SMILES string to match against ``smiles``.
        pubchem_cid:
            PubChem CID to match against ``pubchem_cid``.
        """
        if name is None and smiles is None and pubchem_cid is None:
            raise ValueError("At least one of name, smiles, or pubchem_cid must be provided")

        clauses: list[str] = []
        if name is not None:
            clauses.append(f"name = '{sql_escape(name)}'")
        if smiles is not None:
            clauses.append(f"smiles = '{sql_escape(smiles)}'")
        if pubchem_cid is not None:
            clauses.append(f"pubchem_cid = {pubchem_cid}")

        where = " AND ".join(clauses)
        uids = _query_uids(self._atlas, _TABLE_SMALL_MOLECULE, where)
        self._and_where(_perturbation_where(uids, _PREFIX_SMALL_MOLECULE))
        return self

    def by_biologic(
        self,
        name: str,
        *,
        biologic_type: str | None = None,
    ) -> "PerturbationQuery":
        """Filter to cells treated with a biologic agent.

        Parameters
        ----------
        name:
            Biologic agent name to match against ``biologic_name``.
        biologic_type:
            Optional filter on ``biologic_type`` (e.g. ``"cytokine"``).
        """
        clauses = [f"biologic_name = '{sql_escape(name)}'"]
        if biologic_type is not None:
            clauses.append(f"biologic_type = '{sql_escape(biologic_type)}'")

        where = " AND ".join(clauses)
        uids = _query_uids(self._atlas, _TABLE_BIOLOGIC, where)
        self._and_where(_perturbation_where(uids, _PREFIX_BIOLOGIC))
        return self

    def by_publication(
        self,
        *,
        doi: str | None = None,
        pmid: int | None = None,
        title: str | None = None,
    ) -> "PerturbationQuery":
        """Filter to cells from datasets linked to a publication.

        At least one identifier must be provided.

        Parameters
        ----------
        doi:
            DOI string.
        pmid:
            PubMed ID.
        title:
            Exact publication title.
        """
        if doi is None and pmid is None and title is None:
            raise ValueError("At least one of doi, pmid, or title must be provided")

        clauses: list[str] = []
        if doi is not None:
            clauses.append(f"doi = '{sql_escape(doi)}'")
        if pmid is not None:
            clauses.append(f"pmid = {pmid}")
        if title is not None:
            clauses.append(f"title = '{sql_escape(title)}'")

        where = " AND ".join(clauses)
        pub_uids = _query_uids(self._atlas, _TABLE_PUBLICATIONS, where)
        if not pub_uids:
            self._and_where("1 = 0")
            return self

        # Find dataset UIDs linked to these publications
        dataset_uids = self._resolve_dataset_uids_for_publications(pub_uids)
        self._add_dataset_filter(dataset_uids)
        return self

    def by_accession(
        self,
        accession_id: str,
        database: str = "GEO",
    ) -> "PerturbationQuery":
        """Filter to cells from datasets with a specific accession.

        Parameters
        ----------
        accession_id:
            Accession identifier (e.g. ``"GSE153056"``).
        database:
            Accession database (default ``"GEO"``).
        """
        where = (
            f"accession_id = '{sql_escape(accession_id)}' "
            f"AND accession_database = '{sql_escape(database)}'"
        )
        rows = (
            _open_table(self._atlas, _TABLE_DATASETS)
            .search()
            .where(where, prefilter=True)
            .select(["uid"])
            .to_polars()
        )
        dataset_uids = rows["uid"].unique().to_list()
        self._add_dataset_filter(dataset_uids)
        return self

    def controls_only(
        self,
        control_type: str | None = None,
    ) -> "PerturbationQuery":
        """Filter to negative control cells.

        Parameters
        ----------
        control_type:
            Optional specific control type (e.g. ``"nontargeting"``, ``"DMSO"``).
        """
        self._and_where("is_negative_control = true")
        if control_type is not None:
            self._and_where(f"negative_control_type = '{sql_escape(control_type)}'")
        return self

    def with_perturbation_metadata(
        self,
        *,
        genetic_columns: list[str] | None = None,
        small_molecule_columns: list[str] | None = None,
        biologic_columns: list[str] | None = None,
    ) -> "PerturbationQuery":
        """Enrich results with resolved perturbation records.

        When set, execution methods like ``to_polars()`` will join metadata
        from the perturbation FK tables into the result.

        Parameters
        ----------
        genetic_columns:
            Columns to fetch from ``genetic_perturbations``.
        small_molecule_columns:
            Columns to fetch from ``small_molecules``.
        biologic_columns:
            Columns to fetch from ``biologic_perturbations``.
        """
        self._enrich_perturbations = True
        self._genetic_columns = genetic_columns
        self._small_molecule_columns = small_molecule_columns
        self._biologic_columns = biologic_columns
        return self

    # -- Internal helpers for dataset-level filtering -----------------------

    def _resolve_dataset_uids_for_publications(self, pub_uids: list[str]) -> list[str]:
        in_clause = ", ".join(f"'{sql_escape(u)}'" for u in pub_uids)
        rows = (
            _open_table(self._atlas, _TABLE_DATASETS)
            .search()
            .where(f"publication_uid IN ({in_clause})", prefilter=True)
            .select(["uid"])
            .to_polars()
        )
        return rows["uid"].unique().to_list()

    def _add_dataset_filter(self, dataset_uids: list[str]) -> None:
        if not dataset_uids:
            self._and_where("1 = 0")
            return
        in_clause = ", ".join(f"'{sql_escape(u)}'" for u in dataset_uids)
        self._and_where(f"dataset_uid IN ({in_clause})")

    # -- Execution overrides ------------------------------------------------

    def _materialize_cells(self) -> pl.DataFrame:
        df = super()._materialize_cells()
        if self._enrich_perturbations:
            df = resolve_perturbation_metadata(
                df,
                self._atlas.db,
                genetic_columns=self._genetic_columns,
                small_molecule_columns=self._small_molecule_columns,
                biologic_columns=self._biologic_columns,
            )
        return df
