"""Molecule resolution for SmallMolecule_raw.csv.

Resolves compound names to canonical structures via PubChem/ChEMBL, detects
controls, assigns UIDs, and writes SmallMolecule_resolved.csv.

This script handles the general name-resolution workflow. Dataset-specific
enrichment (correction mappings, SMILES fallback, vendor metadata) should be
done in a follow-up step.

Usage
-----
    python resolve_molecules.py <input_csv> <compound_column> \
        [--smiles-column COL] \
        [--vendor-column COL] \
        [--catalog-column COL] \
        [--output-dir <dir>]

Arguments
---------
input_csv         Path to SmallMolecule_raw.csv (index_col=0).
compound_column   Column containing compound names / control labels.

Options
-------
--smiles-column     Column with SMILES strings to carry through.
--vendor-column     Column with vendor names to carry through.
--catalog-column    Column with catalog numbers to carry through.
--output-dir        Directory for output CSV. Defaults to same directory as
                    input_csv.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from lancell.schema import make_stable_uid
from lancell.standardization import (
    detect_control_labels,
    is_control_label,
    resolve_molecules,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Molecule resolution for SmallMolecule_raw.csv",
    )
    parser.add_argument("input_csv", type=Path, help="Path to raw CSV (index_col=0)")
    parser.add_argument("compound_column", help="Column with compound names / control labels")
    parser.add_argument("--smiles-column", default=None, help="Column with SMILES strings")
    parser.add_argument("--vendor-column", default=None, help="Column with vendor names")
    parser.add_argument("--catalog-column", default=None, help="Column with catalog numbers")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory (default: same as input)"
    )
    args = parser.parse_args(argv)

    raw_df = pd.read_csv(args.input_csv, index_col=0)
    compound_col = args.compound_column
    output_dir = args.output_dir or args.input_csv.parent

    if compound_col not in raw_df.columns:
        print(
            f"ERROR: column '{compound_col}' not found. Available: {list(raw_df.columns)}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Compounds: {len(raw_df)}, Columns: {list(raw_df.columns)}")

    # ------------------------------------------------------------------
    # Control detection
    # ------------------------------------------------------------------
    unique_compounds = raw_df[compound_col].dropna().unique().tolist()
    control_mask = detect_control_labels(unique_compounds)
    control_labels = [
        c for c, is_ctrl in zip(unique_compounds, control_mask, strict=False) if is_ctrl
    ]
    actual_compounds = [
        c for c, is_ctrl in zip(unique_compounds, control_mask, strict=False) if not is_ctrl
    ]
    print(f"Control labels: {control_labels}")
    print(f"Actual compounds: {len(actual_compounds)}")

    # ------------------------------------------------------------------
    # Resolve molecules
    # ------------------------------------------------------------------
    print(f"Resolving {len(actual_compounds)} unique compounds...")
    report = resolve_molecules(actual_compounds, input_type="name")
    print(f"Resolved: {report.resolved}/{report.total}, Unresolved: {report.unresolved}")

    resolution_map = {res.input_value: res for res in report.results}

    if report.unresolved_values:
        print(f"Unresolved ({len(report.unresolved_values)}): {report.unresolved_values[:20]}")

    # ------------------------------------------------------------------
    # Build resolved dataframe
    # ------------------------------------------------------------------
    resolved_df = raw_df.copy()

    names = []
    smiles_list = []
    pubchem_cids = []
    iupac_names = []
    inchi_keys = []
    chembl_ids = []
    resolved_flags = []

    for _, row in raw_df.iterrows():
        compound = row[compound_col]
        if pd.isna(compound) or is_control_label(str(compound)):
            names.append(None)
            smiles_list.append(None)
            pubchem_cids.append(None)
            iupac_names.append(None)
            inchi_keys.append(None)
            chembl_ids.append(None)
            resolved_flags.append(True)
        elif (
            str(compound) in resolution_map
            and resolution_map[str(compound)].resolved_value is not None
        ):
            res = resolution_map[str(compound)]
            names.append(res.resolved_value or res.input_value)
            smiles_list.append(res.canonical_smiles)
            pubchem_cids.append(res.pubchem_cid)
            iupac_names.append(res.iupac_name)
            inchi_keys.append(res.inchi_key)
            chembl_ids.append(res.chembl_id)
            resolved_flags.append(True)
        else:
            # Unresolved — keep original name, structural fields None
            names.append(str(compound))
            smiles_list.append(None)
            pubchem_cids.append(None)
            iupac_names.append(None)
            inchi_keys.append(None)
            chembl_ids.append(None)
            resolved_flags.append(False)

    resolved_df["name"] = names
    resolved_df["smiles"] = smiles_list
    resolved_df["pubchem_cid"] = pubchem_cids
    resolved_df["iupac_name"] = iupac_names
    resolved_df["inchi_key"] = inchi_keys
    resolved_df["chembl_id"] = chembl_ids
    resolved_df["resolved"] = resolved_flags

    # Carry through vendor/catalog if columns specified
    if args.vendor_column:
        if args.vendor_column in raw_df.columns:
            resolved_df["vendor"] = raw_df[args.vendor_column]
        else:
            print(f"WARNING: vendor column '{args.vendor_column}' not found", file=sys.stderr)

    if args.catalog_column:
        if args.catalog_column in raw_df.columns:
            resolved_df["catalog_number"] = raw_df[args.catalog_column]
        else:
            print(f"WARNING: catalog column '{args.catalog_column}' not found", file=sys.stderr)

    # Assign stable UIDs
    uids = []
    for _, row in raw_df.iterrows():
        compound = row[compound_col]
        if pd.isna(compound) or is_control_label(str(compound)):
            uids.append(make_stable_uid("control", str(compound).lower()))
        elif (
            str(compound) in resolution_map
            and resolution_map[str(compound)].resolved_value is not None
        ):
            uids.append(resolution_map[str(compound)].stable_uid)
        else:
            uids.append(make_stable_uid("unresolved", str(compound)))
    resolved_df["uid"] = uids

    output_path = output_dir / "SmallMolecule_resolved.csv"
    resolved_df.to_csv(output_path)

    n_controls = sum(
        1
        for _, row in resolved_df.iterrows()
        if pd.notna(row[compound_col]) and is_control_label(str(row[compound_col]))
    )
    n_resolved = resolved_df["resolved"].sum()
    n_compound_resolved = n_resolved - n_controls
    n_compound_total = len(resolved_df) - n_controls
    print(f"\nWrote {output_path}: {len(resolved_df)} compounds")
    print(f"  Controls: {n_controls}")
    print(f"  Compounds resolved: {n_compound_resolved}/{n_compound_total}")
    print(f"  Total resolved (incl controls): {n_resolved}/{len(resolved_df)}")


if __name__ == "__main__":
    main()
