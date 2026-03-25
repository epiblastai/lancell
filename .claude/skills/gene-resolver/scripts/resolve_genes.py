"""Resolve gene identifiers in a CSV to canonical Ensembl IDs and symbols.

Generic script: takes any CSV with gene identifiers (Ensembl IDs or symbols),
resolves them against the lancell reference DB, and writes a resolved CSV with
standardized gene_name, ensembl_gene_id, organism, resolved, and uid columns.

Schema-specific columns (feature_id, feature_type, etc.) are NOT added here —
the calling agent handles those based on the target schema.

Usage:
    python resolve_genes.py <input_csv> <output_csv> [options]

Options:
    --ensembl-col COL   Column with Ensembl IDs. Default: auto-detect from index.
    --symbol-col COL    Column with gene symbols (used for fallback). Default: auto-detect.
    --organism ORG      Override organism instead of auto-detecting from Ensembl prefixes.
    --index-col COL     CSV column to use as index. Use "none" to disable index handling.
    --dry-run           Print detected columns and planned operations without writing output.
"""

import argparse
import sys

import pandas as pd

from lancell.standardization import (
    detect_organism_from_ensembl_ids,
    is_placeholder_symbol,
    resolve_genes,
    resolve_organisms,
)
from lancell.standardization.types import GeneResolution


def _detect_ensembl_column(df: pd.DataFrame) -> tuple[str | None, bool]:
    """Return (column_name, is_index) for the column containing Ensembl IDs."""
    # Check index first
    sample = [str(v) for v in df.index[:20]]
    if any(s.startswith("ENS") for s in sample):
        return None, True

    # Check columns
    for col in df.columns:
        sample = [str(v) for v in df[col].dropna().head(20)]
        if any(s.startswith("ENS") for s in sample):
            return col, False

    return None, False


def _detect_symbol_column(df: pd.DataFrame, ensembl_col: str | None, ensembl_is_index: bool) -> str | None:
    """Find a column that looks like gene symbols (not the Ensembl column).

    Also checks ``df.index.name``. If the index holds symbols, returns
    the sentinel ``"__index__"`` so the caller can use ``df.index``.
    """
    candidates = ["gene_name", "gene_symbol", "gene_symbols", "feature_name", "symbol", "name"]
    for name in candidates:
        if name in df.columns:
            if not ensembl_is_index and name == ensembl_col:
                continue
            return name
    # Check if the index itself holds symbols (common when Ensembl IDs are also
    # in a regular column, or when the CSV was written with gene names as index)
    if df.index.name in candidates and not ensembl_is_index:
        return "__index__"
    return None


def _resolve_ensembl_ids(
    ensembl_ids: list[str],
    symbols: list[str] | None,
) -> tuple[list[GeneResolution], dict[str, str]]:
    """Resolve Ensembl IDs with per-organism handling and symbol fallback.

    Returns (results_list, organism_map) where organism_map maps common names
    to scientific names.
    """
    # Strip version suffixes
    clean_ids = [eid.split(".")[0] for eid in ensembl_ids]

    # Detect organisms from prefixes
    id_to_organism = detect_organism_from_ensembl_ids(clean_ids)
    unique_organisms = set(v for v in id_to_organism.values() if v != "unknown")
    gene_organisms = [id_to_organism.get(eid, "unknown") for eid in clean_ids]

    print(f"Organisms detected: {unique_organisms}")
    for org in unique_organisms:
        count = sum(1 for v in id_to_organism.values() if v == org)
        print(f"  {org}: {count} genes")

    unknown_count = sum(1 for v in id_to_organism.values() if v == "unknown")
    if unknown_count:
        print(f"  unknown: {unknown_count} genes")

    # Resolve per organism
    all_results: list[GeneResolution | None] = [None] * len(clean_ids)

    for organism in unique_organisms:
        org_mask = [go == organism for go in gene_organisms]
        org_ids = [eid for eid, m in zip(clean_ids, org_mask) if m]
        org_indices = [i for i, m in enumerate(org_mask) if m]

        print(f"\nResolving {len(org_ids)} Ensembl IDs for {organism}...")
        report = resolve_genes(org_ids, organism=organism, input_type="ensembl_id")
        print(f"  {report.resolved} resolved, {report.unresolved} unresolved")
        if report.unresolved_values:
            print(f"  Unresolved sample: {report.unresolved_values[:10]}")

        for idx, res in zip(org_indices, report.results):
            all_results[idx] = res

    # Symbol fallback for unresolved
    if symbols is not None:
        unresolved_indices = [i for i, r in enumerate(all_results) if r is not None and r.resolved_value is None]
        if unresolved_indices:
            print(f"\n{len(unresolved_indices)} unresolved, attempting symbol fallback...")
            for organism in unique_organisms:
                org_sub = [(j, unresolved_indices[j]) for j in range(len(unresolved_indices))
                           if gene_organisms[unresolved_indices[j]] == organism]
                if not org_sub:
                    continue
                org_symbols = [symbols[orig_idx] for _, orig_idx in org_sub]
                valid = [(j, sym) for (j, _), sym in zip(org_sub, org_symbols) if not is_placeholder_symbol(sym)]
                if valid:
                    valid_indices, valid_symbols = zip(*valid)
                    fb_report = resolve_genes(list(valid_symbols), organism=organism, input_type="symbol")
                    print(f"  Symbol fallback for {organism}: {fb_report.resolved}/{fb_report.total} resolved")
                    for j, fb_res in zip(valid_indices, fb_report.results):
                        if fb_res.resolved_value is not None:
                            all_results[org_sub[j][1]] = fb_res

    # Handle unknowns — try first detected organism as default
    unknown_indices = [i for i, r in enumerate(all_results) if r is None]
    if unknown_indices and unique_organisms:
        default_org = next(iter(unique_organisms))
        print(f"\n{len(unknown_indices)} genes with unknown organism, resolving as {default_org}...")
        unk_ids = [clean_ids[i] for i in unknown_indices]
        report = resolve_genes(unk_ids, organism=default_org, input_type="ensembl_id")
        for idx, res in zip(unknown_indices, report.results):
            all_results[idx] = res

    # Build organism common->scientific map
    org_report = resolve_organisms(list(unique_organisms))
    organism_map = {
        inp: res.resolved_value
        for inp, res in zip(unique_organisms, org_report.results)
        if res.resolved_value is not None
    }
    print(f"Organism mapping: {organism_map}")

    return all_results, organism_map


def _resolve_symbols(
    symbols: list[str],
    organism: str,
) -> tuple[list[GeneResolution], dict[str, str]]:
    """Resolve gene symbols when no Ensembl IDs are available."""
    print(f"\nResolving {len(symbols)} symbols for {organism}...")
    report = resolve_genes(symbols, organism=organism, input_type="symbol")
    print(f"  {report.resolved} resolved, {report.unresolved} unresolved")
    if report.unresolved_values:
        print(f"  Unresolved sample: {report.unresolved_values[:10]}")

    org_report = resolve_organisms([organism])
    organism_map = {}
    if org_report.results[0].resolved_value:
        organism_map[organism] = org_report.results[0].resolved_value

    return list(report.results), organism_map


def resolve_gene_csv(
    input_path: str,
    output_path: str,
    ensembl_col: str | None = None,
    symbol_col: str | None = None,
    organism: str | None = None,
    index_col: int | None = 0,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Resolve gene identifiers in a CSV and write results.

    Adds columns: gene_name, ensembl_gene_id, organism, resolved, uid.
    Returns the resolved DataFrame.
    """
    df = pd.read_csv(input_path, index_col=index_col)
    print(f"Loaded {len(df)} features, columns: {list(df.columns)}")

    # Detect columns
    if ensembl_col is None:
        ensembl_col, ensembl_is_index = _detect_ensembl_column(df)
        if ensembl_is_index:
            print(f"Ensembl IDs detected in index")
        elif ensembl_col:
            print(f"Ensembl IDs detected in column: {ensembl_col}")
    else:
        ensembl_is_index = ensembl_col == df.index.name

    if symbol_col is None:
        symbol_col = _detect_symbol_column(df, ensembl_col, ensembl_is_index if ensembl_col or ensembl_is_index else False)
        if symbol_col == "__index__":
            print(f"Symbol column detected in index: {df.index.name}")
        elif symbol_col:
            print(f"Symbol column detected: {symbol_col}")

    # Get identifier lists
    has_ensembl = ensembl_col is not None or (ensembl_col is None and ensembl_is_index)
    if symbol_col == "__index__":
        symbols = [str(v) for v in df.index]
    elif symbol_col:
        symbols = df[symbol_col].astype(str).tolist()
    else:
        symbols = None

    if has_ensembl:
        ensembl_ids = (
            [str(v) for v in df.index]
            if ensembl_is_index
            else df[ensembl_col].astype(str).tolist()
        )
        if dry_run:
            print("Dry run summary:")
            print(f"  index_col={index_col}")
            print(f"  ensembl source={'index' if ensembl_is_index else ensembl_col}")
            print(f"  symbol source={symbol_col}")
            print("  action=resolve Ensembl IDs with symbol fallback")
            return df
        all_results, organism_map = _resolve_ensembl_ids(ensembl_ids, symbols)
    elif symbols is not None:
        if organism is None:
            print("ERROR: No Ensembl IDs found. Must provide --organism when resolving by symbol only.", file=sys.stderr)
            sys.exit(1)
        if dry_run:
            print("Dry run summary:")
            print(f"  index_col={index_col}")
            print(f"  symbol source={symbol_col}")
            print(f"  organism={organism}")
            print("  action=resolve symbols only")
            return df
        all_results, organism_map = _resolve_symbols(symbols, organism)
    else:
        print("ERROR: No gene identifiers found. Provide --ensembl-col or --symbol-col.", file=sys.stderr)
        sys.exit(1)

    # Build output columns
    out = df.copy()

    # Avoid column name collisions: if the index has the same name as an
    # output column we're about to write, rename the index.
    output_col_names = {"gene_name", "ensembl_gene_id", "organism", "resolved", "uid"}
    if out.index.name in output_col_names:
        out.index = out.index.rename(f"raw_{out.index.name}")

    out["gene_name"] = [
        res.symbol if res.symbol else (symbols[i] if symbols else res.input_value)
        for i, res in enumerate(all_results)
    ]
    out["ensembl_gene_id"] = [
        res.ensembl_gene_id if res.ensembl_gene_id else res.input_value
        for res in all_results
    ]
    out["organism"] = [
        organism_map.get(res.organism, res.organism) if res.organism else next(iter(organism_map.values()), "unknown")
        for res in all_results
    ]
    out["resolved"] = [res.resolved_value is not None for res in all_results]
    out["uid"] = [res.stable_uid for res in all_results]

    out.to_csv(output_path)

    resolved_count = out["resolved"].sum()
    unresolved_count = (~out["resolved"]).sum()
    print(f"\nWrote {output_path}: {len(out)} features, {resolved_count} resolved, {unresolved_count} unresolved")

    # Show unresolved
    unresolved = out[~out["resolved"]]
    if len(unresolved) > 0:
        print(f"\nUnresolved examples:")
        for idx, row in unresolved.head(20).iterrows():
            print(f"  {idx} -> gene_name={row['gene_name']}, ensembl_gene_id={row['ensembl_gene_id']}, "
                  f"placeholder={is_placeholder_symbol(str(row['gene_name']))}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resolve gene identifiers in a CSV")
    parser.add_argument("input_csv", help="Input CSV with gene identifiers")
    parser.add_argument("output_csv", help="Output CSV path")
    parser.add_argument("--ensembl-col", default=None, help="Column with Ensembl IDs (default: auto-detect)")
    parser.add_argument("--symbol-col", default=None, help="Column with gene symbols (default: auto-detect)")
    parser.add_argument("--organism", default=None, help="Override organism (e.g. 'human', 'mouse')")
    parser.add_argument(
        "--index-col",
        default="0",
        help='CSV column to use as index. Use "none" to disable index handling (default: 0).',
    )
    parser.add_argument("--dry-run", action="store_true", help="Print detected columns and planned operations only")
    args = parser.parse_args()

    index_col = None if str(args.index_col).lower() == "none" else int(args.index_col)

    resolve_gene_csv(
        args.input_csv,
        args.output_csv,
        ensembl_col=args.ensembl_col,
        symbol_col=args.symbol_col,
        organism=args.organism,
        index_col=index_col,
        dry_run=args.dry_run,
    )
