"""Perturbation utilities: control detection, combinatorial parsing, classification.

No external API calls — pure parsing and classification logic.
"""

from __future__ import annotations

import re

from lancell_examples.multimodal_perturbation_atlas.schema import GeneticPerturbationType

# ---------------------------------------------------------------------------
# Control label detection
# ---------------------------------------------------------------------------

# Negative control patterns for genetic perturbations
_GENETIC_NEGATIVE_CONTROLS: set[str] = {
    "non-targeting",
    "nontargeting",
    "non_targeting",
    "nt",
    "scramble",
    "scrambled",
    "safe-targeting",
    "safe_targeting",
    "safetargeting",
    "luciferase",
    "lacz",
    "gfp",
    "rfp",
    "luc",
    "intergenic",
    "empty",
    "empty vector",
    "empty_vector",
    "negative_control",
    "negative control",
    "neg_ctrl",
    "control",
    "ctrl",
}

# Negative control patterns for chemical perturbations
_CHEMICAL_NEGATIVE_CONTROLS: set[str] = {
    "dmso",
    "vehicle",
    "pbs",
    "untreated",
    "unperturbed",
    "no treatment",
    "no_treatment",
    "mock",
    "media",
    "medium",
    "water",
    "ethanol",
    "etoh",
    "saline",
    "control",
    "ctrl",
    "negative_control",
    "negative control",
    "neg_ctrl",
}

# Combined set for quick membership testing
_ALL_CONTROLS: set[str] = _GENETIC_NEGATIVE_CONTROLS | _CHEMICAL_NEGATIVE_CONTROLS


def detect_control_labels(values: list[str]) -> list[bool]:
    """Return a boolean list indicating which values are control labels.

    Case-insensitive. Strips whitespace before matching.
    """
    return [v.strip().lower() in _ALL_CONTROLS for v in values]


def is_control_label(value: str) -> bool:
    """Check if a single value is a control label."""
    return value.strip().lower() in _ALL_CONTROLS


def detect_negative_control_type(value: str) -> str | None:
    """Classify the type of negative control, or return None if not a control.

    Returns canonical control type strings like ``"nontargeting"``, ``"DMSO"``,
    ``"vehicle"``, ``"untreated"``, ``"scramble"``, ``"intergenic"``, etc.
    """
    v = value.strip().lower()
    if v in {"non-targeting", "nontargeting", "non_targeting", "nt"}:
        return "nontargeting"
    if v in {"scramble", "scrambled"}:
        return "scramble"
    if v in {"intergenic"}:
        return "intergenic"
    if v in {"safe-targeting", "safe_targeting", "safetargeting"}:
        return "safe-targeting"
    if v in {"luciferase", "lacz", "gfp", "rfp", "luc"}:
        return v.lower()
    if v in {"empty", "empty vector", "empty_vector"}:
        return "empty_vector"
    if v in {"dmso"}:
        return "DMSO"
    if v in {"vehicle"}:
        return "vehicle"
    if v in {"pbs"}:
        return "PBS"
    if v in {"untreated", "unperturbed", "no treatment", "no_treatment"}:
        return "untreated"
    if v in {"mock"}:
        return "mock"
    if v in {"media", "medium"}:
        return "media"
    if v in {"water"}:
        return "water"
    if v in {"ethanol", "etoh"}:
        return "ethanol"
    if v in {"saline"}:
        return "saline"
    if v in {"control", "ctrl", "negative_control", "negative control", "neg_ctrl"}:
        return "control"
    return None


# ---------------------------------------------------------------------------
# Combinatorial perturbation parsing
# ---------------------------------------------------------------------------

# Separators used in combinatorial perturbation strings
_COMBO_SEPARATORS = re.compile(r"\s*[+&;|]\s*|\s*,\s+")


def parse_combinatorial_perturbations(value: str) -> list[str]:
    """Split a combinatorial perturbation string into individual targets.

    Handles common separators: ``+``, ``&``, ``;``, ``|``, ``, `` (comma-space).
    Strips whitespace from each component.

    Examples
    --------
    >>> parse_combinatorial_perturbations("TP53+BRCA1")
    ['TP53', 'BRCA1']
    >>> parse_combinatorial_perturbations("TP53 & BRCA1 & KRAS")
    ['TP53', 'BRCA1', 'KRAS']
    >>> parse_combinatorial_perturbations("dexamethasone, imatinib")
    ['dexamethasone', 'imatinib']
    >>> parse_combinatorial_perturbations("single_gene")
    ['single_gene']
    """
    parts = _COMBO_SEPARATORS.split(value.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Perturbation method classification
# ---------------------------------------------------------------------------

_METHOD_PATTERNS: list[tuple[re.Pattern, GeneticPerturbationType]] = [
    (
        re.compile(r"crispr[\s_-]?ko|crispr.*knockout|cas9.*ko", re.I),
        GeneticPerturbationType.CRISPR_KO,
    ),
    (
        re.compile(r"crispr[\s_-]?i|crispr.*interference|dcas9.*krab", re.I),
        GeneticPerturbationType.CRISPR_I,
    ),
    (
        re.compile(r"crispr[\s_-]?a|crispr.*activation|dcas9.*vp64|dcas9.*p65", re.I),
        GeneticPerturbationType.CRISPR_A,
    ),
    (re.compile(r"\bsi[\s_-]?rna\b|small.interfering.rna", re.I), GeneticPerturbationType.SI_RNA),
    (re.compile(r"\bsh[\s_-]?rna\b|short.hairpin.rna", re.I), GeneticPerturbationType.SH_RNA),
    (re.compile(r"\baso\b|antisense.oligo", re.I), GeneticPerturbationType.ASO),
    (
        re.compile(r"overexpression|over[\s_-]?express|oe\b|orf\b|gain.of.function", re.I),
        GeneticPerturbationType.OVEREXPRESSION,
    ),
]


def classify_perturbation_method(value: str) -> GeneticPerturbationType | None:
    """Classify a perturbation method string into a ``GeneticPerturbationType``.

    Returns ``None`` if no known method is detected.
    """
    for pattern, method in _METHOD_PATTERNS:
        if pattern.search(value):
            return method
    # Bare "CRISPR" without further qualifier defaults to KO
    if re.search(r"\bcrispr\b", value, re.I):
        return GeneticPerturbationType.CRISPR_KO
    return None
