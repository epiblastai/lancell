"""PerturbationAtlas: domain-specific atlas wrapper for the multimodal perturbation atlas."""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING

import polars as pl

from lancell.atlas import RaggedAtlas
from lancell.standardization.assemblies import get_assembly_report
from lancell_examples.multimodal_perturbation_atlas.schema import (
    FK_TABLE_SCHEMAS,
    DatasetPerturbationIndex,
)

if TYPE_CHECKING:
    import lancedb
    from lancedb.pydantic import LanceModel

from lancell_examples.multimodal_perturbation_atlas.query import PerturbationQuery

# Dedup key(s) for each FK table and the dataset_perturbation_index.
_FK_DEDUP_KEYS: dict[str, list[str]] = {
    "publications": ["uid"],
    "publication_sections": ["publication_uid", "section_text"],
    "genetic_perturbations": ["uid"],
    "small_molecules": ["uid"],
    "biologic_perturbations": ["uid"],
    "dataset_perturbation_index": ["dataset_uid", "perturbation_uid"],
}


class PerturbationAtlas(RaggedAtlas):
    """A :class:`RaggedAtlas` subclass with perturbation-aware query support.

    Inherits all factory methods (``checkout``, ``checkout_latest``,
    ``restore``, etc.) from :class:`RaggedAtlas`. Since those use ``cls()``,
    calling ``PerturbationAtlas.checkout_latest(...)`` returns a
    ``PerturbationAtlas`` automatically.

    The only behavioural difference is that :meth:`query` returns a
    :class:`PerturbationQuery` instead of a plain :class:`AtlasQuery`.
    """

    def query(self) -> PerturbationQuery:
        """Start building a perturbation-aware query against this atlas."""
        if self._checked_out_version is None:
            raise RuntimeError(
                "query() is only available on a versioned atlas. "
                "After ingestion, call atlas.snapshot() then "
                "PerturbationAtlas.checkout(db_uri, version, schema, store) to pin to a "
                "validated snapshot. For convenience, use PerturbationAtlas.checkout_latest(...)."
            )
        return PerturbationQuery(self)

    # -- Convenience table accessors ----------------------------------------

    @cached_property
    def genetic_perturbations_table(self) -> "lancedb.table.Table":
        return self.db.open_table("genetic_perturbations")

    @cached_property
    def small_molecules_table(self) -> "lancedb.table.Table":
        return self.db.open_table("small_molecules")

    @cached_property
    def biologic_perturbations_table(self) -> "lancedb.table.Table":
        return self.db.open_table("biologic_perturbations")

    @cached_property
    def publications_table(self) -> "lancedb.table.Table":
        return self.db.open_table("publications")

    @cached_property
    def dataset_perturbation_index_table(self) -> "lancedb.table.Table":
        return self.db.open_table("dataset_perturbation_index")

    # -- Genomic coordinate search ------------------------------------------

    def search_perturbations_by_region(
        self,
        chromosome: str,
        start: int,
        end: int,
        *,
        organism: str = "human",
        assembly: str = "GRCh38",
    ) -> pl.DataFrame:
        """Find genetic perturbations whose target overlaps a genomic region.

        Parameters
        ----------
        chromosome:
            UCSC chromosome name (e.g. ``"chr1"``).
        start:
            Start of the query region (0-based).
        end:
            End of the query region.
        organism:
            Organism common name (default ``"human"``).
        assembly:
            Genome assembly (default ``"GRCh38"``).

        Returns
        -------
        pl.DataFrame
            All rows from ``genetic_perturbations`` whose
            ``[target_start, target_end]`` interval intersects ``[start, end]``.
        """
        report = get_assembly_report(organism, assembly)
        seq = report.lookup(chromosome)
        if seq is None or seq.genbank_accession is None:
            raise ValueError(
                f"Could not resolve chromosome {chromosome!r} to a GenBank accession "
                f"in {organism} {assembly}."
            )

        genbank = seq.genbank_accession
        # Overlap condition: target_start <= end AND target_end >= start
        where = (
            f"target_chromosome = '{genbank}' "
            f"AND target_start <= {end} "
            f"AND target_end >= {start}"
        )
        return (
            self.genetic_perturbations_table
            .search()
            .where(where, prefilter=True)
            .to_polars()
        )

    # -- FK table maintenance -----------------------------------------------

    def optimize(self) -> None:
        """Optimize core tables (cells, registries, etc.) and FK tables."""
        super().optimize()
        self.optimize_fk_tables()

    def optimize_fk_tables(self) -> None:
        """Deduplicate and compact all foreign-key tables.

        Loads each FK table fully into memory, deduplicates on its natural
        key(s), validates the result against the schema, then overwrites.
        Also handles ``dataset_perturbation_index``.
        """
        all_schemas: dict[str, type[LanceModel]] = {
            **FK_TABLE_SCHEMAS,
            "dataset_perturbation_index": DatasetPerturbationIndex,
        }
        existing = set(self.db.list_tables().tables)

        for table_name, schema_cls in all_schemas.items():
            if table_name not in existing:
                continue
            subset = _FK_DEDUP_KEYS[table_name]
            table = self.db.open_table(table_name)
            _deduplicate_fk_table(table, schema_cls, subset)
            table.optimize()


def _deduplicate_fk_table(
    table: "lancedb.table.Table",
    schema_cls: type["LanceModel"],
    subset: list[str],
) -> None:
    """Load *table* into memory, deduplicate on *subset*, and rewrite.

    The rewritten data is cast to the Arrow schema derived from *schema_cls*
    so that column types stay correct after the round-trip.
    """
    arrow_schema = schema_cls.to_arrow_schema()
    all_rows = table.search().to_arrow()
    n_before = len(all_rows)
    if n_before == 0:
        print(f"  {table.name}: empty, skipping")
        return

    df = pl.from_arrow(all_rows).unique(subset=subset, keep="first")
    n_after = len(df)
    n_removed = n_before - n_after

    # Validate by casting back through the schema
    deduped_arrow = df.to_arrow().cast(arrow_schema)

    # Rewrite: delete all rows then add the clean set
    table.delete("true")
    table.add(deduped_arrow)

    if n_removed > 0:
        print(f"  {table.name}: removed {n_removed} duplicates ({n_before} -> {n_after})")
    else:
        print(f"  {table.name}: {n_after} rows, no duplicates")
