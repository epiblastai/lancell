"""Assemble resolver fragment CSVs into final standardized obs/var CSVs.

Each resolver produces an isolated fragment file (e.g., fragment_ontology_obs.csv,
fragment_gene_var.csv). This script merges all fragments column-wise into the
final standardized_obs.csv and standardized_var.csv files that the downstream
geo-data-curator expects.

Supports the `|` convention for multi-source columns: when multiple resolvers
write to the same schema field, they use `{target_field}|{SourceClassName}`
column names. This script merges them into the final `{target_field}` column
using type-aware rules derived from the schema.

Usage:
    python scripts/assemble_fragments.py <experiment_dir> [--feature-spaces fs1 fs2 ...] [--schema path/to/schema.py]

Arguments:
    experiment_dir    Path to the experiment directory (e.g., /tmp/geo_agent/GSE264667/HepG2)
    --feature-spaces  Feature space names to assemble for (e.g., gene_expression protein_abundance).
                      If omitted, auto-detected from existing _raw_var.csv files.
    --schema          Path to the Python schema file. Required when fragments use the | convention.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

import pandas as pd


def discover_feature_spaces(experiment_dir: Path) -> list[str]:
    """Auto-detect feature spaces from existing raw var CSV files."""
    feature_spaces = []
    for path in sorted(experiment_dir.glob("*_raw_var.csv")):
        # Extract feature space from filename: {fs}_raw_var.csv
        stem = path.stem  # e.g., gene_expression_raw_var
        suffix = "_raw_var"
        if stem.endswith(suffix):
            fs = stem[: -len(suffix)]
            feature_spaces.append(fs)
    return feature_spaces


def load_fragments(experiment_dir: Path, glob_pattern: str) -> list[pd.DataFrame]:
    """Load all fragment CSVs matching a glob pattern.

    The preparer fragment (filename contains ``fragment_preparer``) provides
    baseline values.  If any resolver fragment supplies the same column, the
    resolver's column wins and the duplicate column is dropped from the
    preparer before concatenation.
    """
    preparer_df: pd.DataFrame | None = None
    resolver_dfs: list[pd.DataFrame] = []

    for path in sorted(experiment_dir.glob(glob_pattern)):
        df = pd.read_csv(path, index_col=0)
        if df.empty:
            continue
        print(f"  loaded {path.name}: {len(df.columns)} columns")
        if "fragment_preparer" in path.name:
            preparer_df = df
        else:
            resolver_dfs.append(df)

    if preparer_df is not None and resolver_dfs:
        resolver_cols = set()
        for rdf in resolver_dfs:
            resolver_cols.update(rdf.columns)
        overlap = set(preparer_df.columns) & resolver_cols
        if overlap:
            print(f"  dropping preparer columns overridden by resolvers: {sorted(overlap)}")
            preparer_df = preparer_df.drop(columns=list(overlap))

    # Resolver fragments first, then preparer (order doesn't matter now
    # that duplicates are removed, but keeps printout intuitive)
    fragments = resolver_dfs
    if preparer_df is not None and not preparer_df.empty:
        fragments.append(preparer_df)
    return fragments


def load_schema_class(schema_path: str) -> type | None:
    """Dynamically import the schema file and find the obs schema class (LancellBaseSchema subclass)."""
    path = Path(schema_path)
    if not path.exists():
        print(f"WARNING: Schema file not found: {schema_path}")
        return None

    spec = importlib.util.spec_from_file_location("_schema_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_schema_module"] = module
    spec.loader.exec_module(module)

    # Find the LancellBaseSchema subclass (the obs schema)
    from lancell.schema import LancellBaseSchema

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, LancellBaseSchema)
            and attr is not LancellBaseSchema
        ):
            return attr
    print(f"WARNING: No LancellBaseSchema subclass found in {schema_path}")
    return None


def get_field_type_category(schema_class: type, field_name: str) -> str:
    """Determine the merge category for a schema field: 'list', 'bool', or 'str'.

    Returns 'str' as the default fallback.
    """
    if schema_class is None:
        return "str"

    field_info = schema_class.model_fields.get(field_name)
    if field_info is None:
        return "str"

    annotation = field_info.annotation

    # Unwrap Optional (Union[X, None])
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            annotation = inner[0]
            origin = get_origin(annotation)

    # Check for list types
    if origin is list:
        return "list"

    # Check for bool (must come before int check since bool is subclass of int)
    if annotation is bool:
        return "bool"

    return "str"


def merge_pipe_columns(assembled: pd.DataFrame, schema_class: type | None) -> pd.DataFrame:
    """Detect `|` columns, group by target field, and merge using type-aware rules.

    Merge rules (determined by schema field type):
    - list fields: concatenate JSON lists across sources
    - bool fields: AND across non-null values (cell is control only if ALL are control)
    - str fields: pipe-join non-null values
    """
    # Find all pipe columns
    pipe_cols = [c for c in assembled.columns if "|" in c]
    if not pipe_cols:
        return assembled

    # Group by target field (prefix before |)
    groups: dict[str, list[str]] = {}
    for col in pipe_cols:
        target_field = col.split("|")[0]
        groups.setdefault(target_field, []).append(col)

    for target_field, source_cols in groups.items():
        category = get_field_type_category(schema_class, target_field)
        print(f"  merging {len(source_cols)} sources for '{target_field}' (type: {category})")

        if category == "list":
            assembled[target_field] = _merge_list_columns(assembled, source_cols)
        elif category == "bool":
            assembled[target_field] = _merge_bool_columns(assembled, source_cols)
        else:
            assembled[target_field] = _merge_str_columns(assembled, source_cols)

        # Drop the source | columns
        assembled = assembled.drop(columns=source_cols)

    return assembled


def _merge_list_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Concatenate JSON list values across source columns. NaN/None values are skipped."""

    def concat_lists(row):
        result = []
        for col in cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, str):
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    result.extend(parsed)
                else:
                    result.append(parsed)
            else:
                result.append(val)
        return json.dumps(result) if result else None

    return df[cols].apply(concat_lists, axis=1)


def _merge_bool_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """AND across non-null boolean values. Cell is control only if ALL sources say so."""

    def and_bools(row):
        values = []
        for col in cols:
            val = row[col]
            if pd.notna(val):
                values.append(bool(val))
        if not values:
            return None
        return all(values)

    return df[cols].apply(and_bools, axis=1)


def _merge_str_columns(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Pipe-join non-null string values across source columns."""

    def join_strs(row):
        values = []
        for col in cols:
            val = row[col]
            if pd.notna(val):
                values.append(str(val))
        if not values:
            return None
        return "|".join(values)

    return df[cols].apply(join_strs, axis=1)


def coerce_string_dtypes(assembled: pd.DataFrame) -> pd.DataFrame:
    """Coerce object columns: replace NaN with None for proper null semantics."""
    for col in assembled.columns:
        if assembled[col].dtype == object:
            assembled[col] = assembled[col].where(assembled[col].notna(), None)
    return assembled


def merge_resolved_columns(assembled: pd.DataFrame) -> pd.DataFrame:
    """Find all *_resolved columns, compute combined resolved, drop per-resolver columns."""
    resolved_cols = [c for c in assembled.columns if c.endswith("_resolved")]
    if not resolved_cols:
        return assembled

    # A row is resolved if ALL resolver-specific resolved columns are True
    assembled["resolved"] = assembled[resolved_cols].all(axis=1)
    assembled = assembled.drop(columns=resolved_cols)
    return assembled


def assemble_obs(experiment_dir: Path, feature_space: str, schema_class: type | None) -> Path:
    """Merge all obs fragment CSVs into {fs}_standardized_obs.csv."""
    print(f"Assembling obs for {feature_space}...")

    # Load raw obs for the authoritative index
    raw_obs_path = experiment_dir / f"{feature_space}_raw_obs.csv"
    if not raw_obs_path.exists():
        raise FileNotFoundError(f"Raw obs CSV not found: {raw_obs_path}")
    raw_obs_index = pd.read_csv(raw_obs_path, index_col=0, usecols=[0]).index

    # Load all obs fragments for this feature space
    pattern = f"{feature_space}_fragment_*_obs.csv"
    fragments = load_fragments(experiment_dir, pattern)

    if not fragments:
        print("  no obs fragments found, creating empty standardized obs")
        assembled = pd.DataFrame(index=raw_obs_index)
    else:
        assembled = pd.concat(fragments, axis=1)
        # Verify index alignment
        if not assembled.index.equals(raw_obs_index):
            print("  WARNING: fragment indices do not match raw obs index, reindexing")
            assembled = assembled.reindex(raw_obs_index)

    # Merge | columns using type-aware rules
    assembled = merge_pipe_columns(assembled, schema_class)

    # Merge resolved columns
    assembled = merge_resolved_columns(assembled)

    # Coerce string dtypes (NaN → None)
    assembled = coerce_string_dtypes(assembled)

    # Write final standardized obs
    output_path = experiment_dir / f"{feature_space}_standardized_obs.csv"
    assembled.to_csv(output_path)
    print(f"  wrote {output_path.name}: {len(assembled.columns)} columns, {len(assembled)} rows")
    return output_path


def assemble_var(experiment_dir: Path, feature_space: str, schema_class: type | None) -> Path:
    """Merge all var fragment CSVs for a feature space into {fs}_standardized_var.csv.

    If a standardized var file already exists (e.g., written by the gene-resolver
    with ``global_feature_uid``), it is left untouched unless there are var
    fragment files that need merging.
    """
    print(f"Assembling var for {feature_space}...")

    output_path = experiment_dir / f"{feature_space}_standardized_var.csv"

    # Check for var fragment files that need merging
    pattern = f"{feature_space}_fragment_*_var.csv"
    fragment_paths = sorted(experiment_dir.glob(pattern))

    if not fragment_paths and output_path.exists():
        print(f"  {output_path.name} already exists (written by resolver), skipping")
        return output_path

    # Load raw var for the authoritative index
    raw_var_path = experiment_dir / f"{feature_space}_raw_var.csv"
    if not raw_var_path.exists():
        raise FileNotFoundError(f"Raw var CSV not found: {raw_var_path}")
    raw_var_index = pd.read_csv(raw_var_path, index_col=0, usecols=[0]).index

    # Load all var fragments for this feature space
    fragments = load_fragments(experiment_dir, pattern)

    if not fragments:
        print("  no var fragments found, creating empty standardized var")
        assembled = pd.DataFrame(index=raw_var_index)
    else:
        assembled = pd.concat(fragments, axis=1)
        if not assembled.index.equals(raw_var_index):
            print("  WARNING: fragment indices do not match raw var index, reindexing")
            assembled = assembled.reindex(raw_var_index)

    # Merge | columns (less common for var, but supported)
    assembled = merge_pipe_columns(assembled, schema_class)

    # Merge resolved columns
    assembled = merge_resolved_columns(assembled)

    # Coerce string dtypes
    assembled = coerce_string_dtypes(assembled)

    # Write final standardized var
    assembled.to_csv(output_path)
    print(f"  wrote {output_path.name}: {len(assembled.columns)} columns, {len(assembled)} rows")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Assemble resolver fragments into standardized CSVs"
    )
    parser.add_argument("experiment_dir", type=str, help="Path to the experiment directory")
    parser.add_argument(
        "--feature-spaces",
        nargs="*",
        default=None,
        help="Feature space names (auto-detected if omitted)",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="Path to the Python schema file (required for | column merging)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)

    # Load schema if provided
    schema_class = None
    if args.schema:
        schema_class = load_schema_class(args.schema)
        if schema_class:
            print(f"Loaded schema: {schema_class.__name__}")

    # Determine feature spaces
    feature_spaces = args.feature_spaces
    if feature_spaces is None:
        feature_spaces = discover_feature_spaces(experiment_dir)
        if feature_spaces:
            print(f"Auto-detected feature spaces: {feature_spaces}")
        else:
            print("No feature spaces detected, nothing to assemble")
            return

    # Assemble obs and var for each feature space
    for fs in feature_spaces:
        assemble_obs(experiment_dir, fs, schema_class)
        assemble_var(experiment_dir, fs, schema_class)

    print("Assembly complete.")


if __name__ == "__main__":
    main()
