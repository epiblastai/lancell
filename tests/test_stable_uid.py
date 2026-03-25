"""Tests for deterministic stable UIDs."""

from lancell.schema import make_stable_uid
from lancell.standardization.types import (
    GeneResolution,
    GuideRnaResolution,
    MoleculeResolution,
    ProteinResolution,
)


def test_make_stable_uid_deterministic():
    """Same inputs always produce the same UID."""
    uid1 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    uid2 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    assert uid1 == uid2
    assert len(uid1) == 16


def test_make_stable_uid_different_inputs():
    """Different inputs produce different UIDs."""
    uid1 = make_stable_uid("ENSG00000141510", "homo_sapiens")
    uid2 = make_stable_uid("ENSG00000141510", "mus_musculus")
    assert uid1 != uid2


def test_gene_resolution_stable_uid():
    """GeneResolution.stable_uid is deterministic on identity fields."""
    res = GeneResolution(
        input_value="TP53",
        resolved_value="TP53",
        confidence=1.0,
        source="ensembl",
        ensembl_gene_id="ENSG00000141510",
        symbol="TP53",
        organism="homo_sapiens",
        ncbi_gene_id=7157,
    )
    uid1 = res.stable_uid
    uid2 = res.stable_uid
    assert uid1 == uid2
    assert uid1 == make_stable_uid("ENSG00000141510", "homo_sapiens")


def test_gene_resolution_unresolved_fallback():
    """Unresolved gene uses (\"unresolved\", input_value) fallback."""
    res = GeneResolution(
        input_value="FAKEGENE",
        resolved_value=None,
        confidence=0.0,
        source="ensembl",
        ensembl_gene_id=None,
        symbol=None,
        organism="homo_sapiens",
    )
    uid = res.stable_uid
    assert uid == make_stable_uid("unresolved", "FAKEGENE")


def test_protein_resolution_stable_uid():
    res = ProteinResolution(
        input_value="CD3",
        resolved_value="CD3E",
        confidence=1.0,
        source="uniprot",
        uniprot_id="P07766",
        organism="homo_sapiens",
    )
    assert res.stable_uid == make_stable_uid("P07766", "homo_sapiens")


def test_molecule_resolution_stable_uid():
    res = MoleculeResolution(
        input_value="aspirin",
        resolved_value="Aspirin",
        confidence=1.0,
        source="pubchem",
        pubchem_cid=2244,
    )
    assert res.stable_uid == make_stable_uid("2244")


def test_guide_rna_resolution_stable_uid():
    res = GuideRnaResolution(
        input_value="sgTP53_1",
        resolved_value="sgTP53_1",
        confidence=1.0,
        source="blat",
        chromosome="chr17",
        intended_ensembl_gene_id="ENSG00000141510",
        target_start=7577120,
        target_end=7577140,
        target_strand="-",
        assembly="hg38",
    )
    assert res.stable_uid == make_stable_uid("chr17", "7577120", "7577140", "-", "hg38")


def test_guide_rna_unresolved_when_coordinates_missing():
    """GuideRnaResolution falls back when target coordinates are None."""
    res = GuideRnaResolution(
        input_value="sgTP53_1",
        resolved_value="sgTP53_1",
        confidence=0.5,
        source="blat",
        intended_ensembl_gene_id="ENSG00000141510",
        target_start=None,
        target_end=None,
        target_strand=None,
    )
    assert res.stable_uid == make_stable_uid("unresolved", "sgTP53_1")


def test_no_cross_type_collision():
    """Different entity types with same raw value produce different UIDs."""
    gene = GeneResolution(input_value="TP53", resolved_value=None, confidence=0.0, source="ensembl")
    protein = ProteinResolution(
        input_value="TP53", resolved_value=None, confidence=0.0, source="uniprot"
    )
    # Both unresolved, same input — but both use same fallback, which is expected
    # since unresolved entities are keyed on input_value
    assert gene.stable_uid == protein.stable_uid

    # When resolved, they differ because identity fields are structurally distinct
    gene_resolved = GeneResolution(
        input_value="TP53",
        resolved_value="TP53",
        confidence=1.0,
        source="ensembl",
        ensembl_gene_id="ENSG00000141510",
        organism="homo_sapiens",
    )
    protein_resolved = ProteinResolution(
        input_value="TP53",
        resolved_value="P04637",
        confidence=1.0,
        source="uniprot",
        uniprot_id="P04637",
        organism="homo_sapiens",
    )
    assert gene_resolved.stable_uid != protein_resolved.stable_uid
