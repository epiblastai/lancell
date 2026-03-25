"""Write per-experiment standardized var CSVs mapping var indices to global feature UIDs.

Reads the resolved CSV (which contains var_index and uid columns) and each
experiment's raw var CSV (which has the authoritative feature order).  For each
experiment, produces {feature_space}_standardized_var.csv with the original var
index and a ``global_feature_uid`` column.

Usage:
    python write_standardized_var.py <accession_dir> \
        --resolved-csv GenomicFeatureSchema_resolved.csv \
        --feature-space gene_expression
"""

import argparse
from pathlib import Path

import pandas as pd


def write_standardized_var(
    accession_dir: Path,
    resolved_csv: str,
    feature_space: str,
) -> list[Path]:
    """Build var_index → uid mapping and write per-experiment standardized var CSVs.

    Returns list of written file paths.
    """
    resolved_path = accession_dir / resolved_csv
    resolved = pd.read_csv(resolved_path)
    print(f"Loaded {len(resolved)} resolved features from {resolved_path.name}")

    if "var_index" not in resolved.columns:
        raise ValueError(
            f"Expected 'var_index' column in {resolved_path}. "
            f"Available columns: {list(resolved.columns)}"
        )
    if "uid" not in resolved.columns:
        raise ValueError(
            f"Expected 'uid' column in {resolved_path}. "
            f"Available columns: {list(resolved.columns)}"
        )

    var_to_uid = dict(zip(resolved["var_index"].astype(str), resolved["uid"]))

    # Discover experiment directories containing raw var CSVs
    raw_var_pattern = f"{feature_space}_raw_var.csv"
    exp_dirs = sorted(
        d for d in accession_dir.iterdir()
        if d.is_dir() and (d / raw_var_pattern).exists()
    )

    if not exp_dirs:
        print(f"No experiment directories found with {raw_var_pattern}")
        return []

    written = []
    for exp_dir in exp_dirs:
        raw_var = pd.read_csv(exp_dir / raw_var_pattern, index_col=0, usecols=[0])
        var_indices = [str(v) for v in raw_var.index]

        uids = []
        missing = 0
        for idx in var_indices:
            uid = var_to_uid.get(idx)
            if uid is None:
                missing += 1
            uids.append(uid)

        if missing:
            print(
                f"WARNING: {missing}/{len(var_indices)} var indices in "
                f"{exp_dir.name} not found in resolved CSV"
            )

        out = pd.DataFrame(
            {"global_feature_uid": uids},
            index=raw_var.index,
        )
        out_path = exp_dir / f"{feature_space}_standardized_var.csv"
        out.to_csv(out_path)
        print(f"Wrote {out_path.name}: {len(out)} features ({exp_dir.name})")
        written.append(out_path)

    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write per-experiment standardized var CSVs with global feature UIDs"
    )
    parser.add_argument("accession_dir", help="Path to the accession directory")
    parser.add_argument(
        "--resolved-csv",
        default="GenomicFeatureSchema_resolved.csv",
        help="Filename of the resolved CSV (default: GenomicFeatureSchema_resolved.csv)",
    )
    parser.add_argument(
        "--feature-space",
        default="gene_expression",
        help="Feature space name (default: gene_expression)",
    )
    args = parser.parse_args()

    write_standardized_var(
        Path(args.accession_dir),
        args.resolved_csv,
        args.feature_space,
    )
