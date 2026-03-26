"""Result types for the standardization suite.

Every resolver returns structured dataclasses instead of raw dicts or bare strings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import pandas as pd

from lancell.schema import make_stable_uid


@dataclass
class Resolution:
    """Base result for any single resolution attempt."""

    _identity_fields: ClassVar[tuple[str, ...]] = ()

    input_value: str
    resolved_value: str | None  # Canonical form, or None if failed
    confidence: float  # 1.0 = exact, 0.0 = failed
    source: str  # Which API/ontology provided the resolution
    alternatives: list[str] = field(default_factory=list)

    @property
    def stable_uid(self) -> str:
        """Deterministic UID derived from identity fields.

        Falls back to hashing ("unresolved", input_value) when any identity
        field is None, so re-runs produce the same UID for unresolved entries.
        """
        values = []
        for f in self._identity_fields:
            val = getattr(self, f)
            if val is None:
                return make_stable_uid("unresolved", self.input_value)
            values.append(str(val))
        return make_stable_uid(*values)


@dataclass
class GeneResolution(Resolution):
    _identity_fields: ClassVar[tuple[str, ...]] = ("ensembl_gene_id", "organism")

    ensembl_gene_id: str | None = None
    symbol: str | None = None  # HGNC/MGI canonical symbol
    organism: str | None = None
    ncbi_gene_id: int | None = None


@dataclass
class MoleculeResolution(Resolution):
    _identity_fields: ClassVar[tuple[str, ...]] = ("pubchem_cid",)

    pubchem_cid: int | None = None
    canonical_smiles: str | None = None
    inchi_key: str | None = None
    iupac_name: str | None = None
    chembl_id: str | None = None


@dataclass
class ProteinResolution(Resolution):
    _identity_fields: ClassVar[tuple[str, ...]] = ("uniprot_id", "organism")

    uniprot_id: str | None = None
    gene_name: str | None = None
    protein_name: str | None = None
    organism: str | None = None
    sequence: str | None = None
    sequence_length: int | None = None


@dataclass
class GuideRnaResolution(Resolution):
    _identity_fields: ClassVar[tuple[str, ...]] = (
        "chromosome",
        "target_start",
        "target_end",
        "target_strand",
        "assembly",
    )

    chromosome: str | None = None  # e.g. "chr17"
    target_start: int | None = None
    target_end: int | None = None
    target_strand: str | None = None  # "+" or "-"
    intended_gene_name: str | None = None
    intended_ensembl_gene_id: str | None = None
    target_context: str | None = None  # TargetContext value
    assembly: str | None = None  # e.g. "hg38"
    blat_pct_match: float | None = None


@dataclass
class CellLineResolution(Resolution):
    cellosaurus_id: str | None = None  # e.g., "CVCL_0030"
    cell_line_name: str | None = None  # e.g., "HeLa"
    species: str | None = None  # e.g., "Homo sapiens"
    disease: str | None = None  # e.g., "Cervical adenocarcinoma"
    sex: str | None = None
    category: str | None = None  # e.g., "Cancer cell line"


@dataclass
class OntologyResolution(Resolution):
    ontology_term_id: str | None = None  # e.g., "CL:0000540", "UBERON:0002048"
    ontology_name: str | None = None  # e.g., "Cell Ontology", "UBERON"


@dataclass
class ResolutionReport:
    """Summary of a batch resolution run."""

    total: int
    resolved: int
    unresolved: int
    ambiguous: int
    results: list[Resolution]  # One per input value

    @property
    def unresolved_values(self) -> list[str]:
        return [r.input_value for r in self.results if r.resolved_value is None]

    @property
    def ambiguous_values(self) -> list[str]:
        return [r.input_value for r in self.results if len(r.alternatives) > 1]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            row = {
                "input_value": r.input_value,
                "resolved_value": r.resolved_value,
                "confidence": r.confidence,
                "source": r.source,
                "alternatives": "; ".join(r.alternatives) if r.alternatives else None,
            }
            # Add subclass-specific fields
            if isinstance(r, GeneResolution):
                row["ensembl_gene_id"] = r.ensembl_gene_id
                row["symbol"] = r.symbol
                row["organism"] = r.organism
                row["ncbi_gene_id"] = r.ncbi_gene_id
            elif isinstance(r, MoleculeResolution):
                row["pubchem_cid"] = r.pubchem_cid
                row["canonical_smiles"] = r.canonical_smiles
                row["inchi_key"] = r.inchi_key
                row["iupac_name"] = r.iupac_name
                row["chembl_id"] = r.chembl_id
            elif isinstance(r, ProteinResolution):
                row["uniprot_id"] = r.uniprot_id
                row["gene_name"] = r.gene_name
                row["protein_name"] = r.protein_name
                row["organism"] = r.organism
                row["sequence"] = r.sequence
                row["sequence_length"] = r.sequence_length
            elif isinstance(r, GuideRnaResolution):
                row["chromosome"] = r.chromosome
                row["target_start"] = r.target_start
                row["target_end"] = r.target_end
                row["target_strand"] = r.target_strand
                row["intended_gene_name"] = r.intended_gene_name
                row["intended_ensembl_gene_id"] = r.intended_ensembl_gene_id
                row["target_context"] = r.target_context
                row["assembly"] = r.assembly
                row["blat_pct_match"] = r.blat_pct_match
            elif isinstance(r, CellLineResolution):
                row["cellosaurus_id"] = r.cellosaurus_id
                row["cell_line_name"] = r.cell_line_name
                row["species"] = r.species
                row["disease"] = r.disease
                row["sex"] = r.sex
                row["category"] = r.category
            elif isinstance(r, OntologyResolution):
                row["ontology_term_id"] = r.ontology_term_id
                row["ontology_name"] = r.ontology_name
            rows.append(row)
        return pd.DataFrame(rows)
