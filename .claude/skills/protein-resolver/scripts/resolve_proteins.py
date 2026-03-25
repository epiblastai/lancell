"""Resolve protein identifiers in a CSV to canonical UniProt IDs and metadata.

Generic script: takes any CSV with protein identifiers (aliases, gene names,
UniProt accessions), resolves them against the lancell reference DB, and writes
a resolved CSV with standardized feature_name, uniprot_id, protein_name,
gene_name, organism, sequence, sequence_length, resolved, and uid columns.

Schema-specific columns (feature_id, feature_type, etc.) are NOT added here —
the calling agent handles those based on the target schema.

Usage:
    python resolve_proteins.py <input_csv> <output_csv> [options]

Options:
    --protein-col COL   Column with protein identifiers. Default: auto-detect.
    --organism ORG      Organism for resolution (default: human).
    --index-col COL     CSV column to use as index. Use "none" to disable index handling.
    --dry-run           Print detected columns and planned operations without writing output.
"""

import argparse
import sys

import pandas as pd

from lancell.schema import make_stable_uid
from lancell.standardization import (
    is_control_label,
    resolve_organisms,
    resolve_proteins,
)
from lancell.standardization.types import ProteinResolution

# Isotype control patterns — is_control_label() does NOT detect these.
_ISOTYPE_PATTERNS = {"igg1", "igg2a", "igg2b", "igg2c", "igm", "iga", "igd", "ige"}


def _is_isotype_control(name: str) -> bool:
    """Check if a protein name is an isotype control antibody."""
    lower = name.strip().lower()
    return (
        any(lower == p for p in _ISOTYPE_PATTERNS)
        or "isotype" in lower
        or lower.startswith("mouse-igg")
        or lower.startswith("rat-igg")
    )


def _detect_protein_column(df: pd.DataFrame) -> tuple[str | None, bool]:
    """Return (column_name, is_index) for the column containing protein identifiers.

    Checks common column names first, then falls back to the index.
    """
    candidates = ["var_index", "feature_name", "protein", "protein_name", "antibody", "target"]
    for col in candidates:
        if col in df.columns:
            return col, False

    # Check if the index looks like protein names (not numeric)
    sample = [str(v) for v in df.index[:20]]
    if all(not s.isdigit() for s in sample):
        return None, True

    return None, False


def resolve_protein_csv(
    input_path: str,
    output_path: str,
    protein_col: str | None = None,
    organism: str = "human",
    index_col: int | None = 0,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Resolve protein identifiers in a CSV and write results.

    Adds columns: feature_name, uniprot_id, protein_name, gene_name,
    organism, sequence, sequence_length, resolved, uid.
    Returns the resolved DataFrame.
    """
    df = pd.read_csv(input_path, index_col=index_col)
    print(f"Loaded {len(df)} features, columns: {list(df.columns)}")

    # Detect protein column
    if protein_col is None:
        protein_col, protein_is_index = _detect_protein_column(df)
        if protein_is_index:
            print("Protein identifiers detected in index")
        elif protein_col:
            print(f"Protein identifiers detected in column: {protein_col}")
    else:
        protein_is_index = protein_col == df.index.name

    if protein_col is None and not protein_is_index:
        print("ERROR: No protein identifier column found. Provide --protein-col.", file=sys.stderr)
        sys.exit(1)

    # Get identifier list
    if protein_is_index:
        protein_aliases = [str(v) for v in df.index]
    else:
        protein_aliases = df[protein_col].astype(str).tolist()

    print(f"Sample identifiers: {protein_aliases[:10]}")

    # Separate isotype controls from actual proteins
    isotype_mask = [_is_isotype_control(p) for p in protein_aliases]
    control_mask = [is_control_label(p) for p in protein_aliases]
    exclude_mask = [iso or ctrl for iso, ctrl in zip(isotype_mask, control_mask)]

    actual_proteins = [p for p, exc in zip(protein_aliases, exclude_mask) if not exc]
    isotype_count = sum(isotype_mask)
    control_count = sum(control_mask)
    print(f"Actual proteins: {len(actual_proteins)}, Isotype controls: {isotype_count}, Other controls: {control_count}")

    if dry_run:
        print("Dry run summary:")
        print(f"  index_col={index_col}")
        print(f"  protein source={'index' if protein_is_index else protein_col}")
        print(f"  organism={organism}")
        print(f"  action=resolve {len(actual_proteins)} proteins, skip {sum(exclude_mask)} controls")
        return df

    # Resolve actual proteins
    print(f"\nResolving {len(actual_proteins)} proteins for {organism}...")
    report = resolve_proteins(actual_proteins, organism=organism)
    print(f"  {report.resolved} resolved, {report.unresolved} unresolved")
    if report.unresolved_values:
        print(f"  Unresolved sample: {report.unresolved_values[:10]}")

    # Map organism to scientific name
    org_report = resolve_organisms([organism])
    organism_scientific = organism
    if org_report.results[0].resolved_value:
        organism_scientific = org_report.results[0].resolved_value
    print(f"Organism: {organism} -> {organism_scientific}")

    # Build result lookup: index into report.results for non-excluded proteins
    result_iter = iter(report.results)
    all_results: list[ProteinResolution | None] = []
    for exc in exclude_mask:
        if exc:
            all_results.append(None)
        else:
            all_results.append(next(result_iter))

    # Build output columns
    out = df.copy()

    # Avoid column name collisions with the index
    output_col_names = {"feature_name", "uniprot_id", "protein_name", "gene_name",
                        "organism", "sequence", "sequence_length", "resolved", "uid"}
    if out.index.name in output_col_names:
        out.index = out.index.rename(f"raw_{out.index.name}")

    out["feature_name"] = protein_aliases

    out["uniprot_id"] = [
        res.uniprot_id if res is not None else None
        for res in all_results
    ]
    out["protein_name"] = [
        res.protein_name if res is not None and res.protein_name else protein_aliases[i]
        for i, res in enumerate(all_results)
    ]
    out["gene_name"] = [
        res.gene_name if res is not None else None
        for res in all_results
    ]
    out["organism"] = organism_scientific
    out["sequence"] = [
        res.sequence if res is not None else None
        for res in all_results
    ]
    out["sequence_length"] = [
        res.sequence_length if res is not None else None
        for res in all_results
    ]
    out["resolved"] = [
        res is not None and res.resolved_value is not None
        for res in all_results
    ]
    out["uid"] = [
        res.stable_uid if res is not None
        else make_stable_uid("control", protein_aliases[i].lower())
        for i, res in enumerate(all_results)
    ]

    out.to_csv(output_path)

    resolved_count = out["resolved"].sum()
    unresolved_count = (~out["resolved"]).sum()
    print(f"\nWrote {output_path}: {len(out)} features, {resolved_count} resolved, {unresolved_count} unresolved")

    # Show unresolved
    unresolved = out[~out["resolved"]]
    if len(unresolved) > 0:
        print(f"\nUnresolved examples:")
        for idx, row in unresolved.head(20).iterrows():
            is_iso = _is_isotype_control(str(row["feature_name"]))
            is_ctrl = is_control_label(str(row["feature_name"]))
            print(f"  {idx} -> feature_name={row['feature_name']}, "
                  f"isotype_control={is_iso}, other_control={is_ctrl}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve protein identifiers in a CSV")
    parser.add_argument("input_csv", help="Input CSV with protein identifiers")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("--protein-col", default=None, help="Column with protein identifiers (default: auto-detect)")
    parser.add_argument("--organism", default="human", help="Organism for resolution (default: human)")
    parser.add_argument(
        "--index-col",
        default="0",
        help='CSV column to use as index. Use "none" to disable index handling (default: 0).',
    )
    parser.add_argument("--dry-run", action="store_true", help="Print detected columns and planned operations only")
    args = parser.parse_args()

    index_col = None if str(args.index_col).lower() == "none" else int(args.index_col)

    resolve_protein_csv(
        args.input_csv,
        args.output_csv,
        protein_col=args.protein_col,
        organism=args.organism,
        index_col=index_col,
        dry_run=args.dry_run,
    )
