"""Finalize a resolved CSV against a target Pydantic schema.

Takes a resolved CSV (e.g. GenomicFeatureSchema_resolved.csv from
resolve_genes.py), strips non-schema columns (including ``resolved``),
coerces types, fills missing nullable columns, and writes parquet with
correct types so the output can be loaded directly into LanceDB.

Does NOT do per-row pydantic validation — type coercion + parquet schema
enforcement is sufficient. Type errors surface at LanceDB insertion time.

Usage:
    python finalize_features.py <resolved_csv> <output_parquet> \
        <schema_module> <schema_class> [--column KEY=VALUE ...]

Example:
    python finalize_features.py \
        /tmp/GSE123/GenomicFeatureSchema_resolved.csv \
        /tmp/GSE123/GenomicFeatureSchema.parquet \
        lancell_examples.multimodal_perturbation_atlas.schema \
        GenomicFeatureSchema \
        --column feature_type=gene \
        --column feature_id=ensembl_gene_id
"""

import argparse
import importlib
import json
import sys
from types import UnionType
from typing import Union, get_args, get_origin

import pandas as pd
from pydantic_core import PydanticUndefined

# -- Type introspection helpers (same logic as validate_obs.py) ----------------


def _get_field_type_category(annotation: type) -> str:
    """Return 'list', 'bool', 'int', 'float', or 'str' for a field annotation."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            annotation = inner[0]
            origin = get_origin(annotation)

    if origin is list:
        return "list"
    if annotation is bool:
        return "bool"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    return "str"


def _is_nullable(annotation: type) -> bool:
    """Return True if the annotation accepts None (e.g. str | None)."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        return type(None) in get_args(annotation)
    return False


def _coerce_column(series: pd.Series, category: str) -> pd.Series:
    """Coerce a pandas Series so its dtype matches what pyarrow expects."""
    if category == "list":

        def _parse(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, str):
                return json.loads(v)
            return v

        return series.apply(_parse)

    if category == "bool":

        def _to_bool(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, bool):
                return v
            return str(v).lower() in ("true", "1", "yes")

        return series.apply(_to_bool)

    if category == "int":
        return series.astype("Int64")

    if category == "float":
        return series.astype("Float64")

    # str: cast to string, preserving nulls as None.
    return series.apply(
        lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    )


# -- Schema introspection -----------------------------------------------------

# Fields auto-assigned by the atlas, never expected in resolver output.
_ATLAS_AUTO_FIELDS = {"global_index"}


def _get_schema_fields(schema_class: type) -> dict[str, dict]:
    """Extract user-facing fields from a LanceModel schema class.

    Returns {field_name: {"category": str, "nullable": bool, "has_default": bool}}
    Excludes atlas-managed fields (global_index).
    """
    fields = {}
    for name, field_info in schema_class.model_fields.items():
        if name in _ATLAS_AUTO_FIELDS:
            continue

        has_default = (
            field_info.default is not PydanticUndefined or field_info.default_factory is not None
        )
        fields[name] = {
            "category": _get_field_type_category(field_info.annotation),
            "nullable": _is_nullable(field_info.annotation),
            "has_default": has_default,
        }
    return fields


# -- Main finalization ---------------------------------------------------------


def finalize(
    resolved_path: str,
    output_path: str,
    schema_class: type,
    column_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Finalize a resolved CSV: strip, coerce, and write parquet.

    Parameters
    ----------
    resolved_path : str
        Path to the resolved CSV (with ``resolved`` column and extra raw columns).
    output_path : str
        Path to write the finalized parquet.
    schema_class : type
        Pydantic LanceModel class to validate against.
    column_defaults : dict, optional
        Extra columns to add. Values that match an existing column name
        are treated as column copies (e.g. ``{"feature_id": "ensembl_gene_id"}``).
        If 'None'/'null' (case-insensitive), sets actual Python None.
        Otherwise the literal string is used as a constant.

    Returns
    -------
    pd.DataFrame
        The finalized DataFrame.
    """
    df = pd.read_csv(resolved_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {resolved_path}")

    # 1. Apply --column defaults
    for col, value in (column_defaults or {}).items():
        if value in df.columns:
            df[col] = df[value]
        elif value.lower() in ("none", "null"):
            df[col] = None
        else:
            df[col] = value

    # 2. Identify schema fields
    schema_fields = _get_schema_fields(schema_class)
    print(f"Schema fields: {len(schema_fields)}")

    # 3. Fill missing nullable/defaulted columns with None, error on missing required
    missing = set(schema_fields) - set(df.columns)
    errors = []
    for name in sorted(missing):
        info = schema_fields[name]
        if info["nullable"] or info["has_default"]:
            df[name] = None
        else:
            errors.append(name)

    if errors:
        print(f"ERROR: Missing required non-nullable columns: {errors}", file=sys.stderr)
        print("Use --column KEY=VALUE to provide them.", file=sys.stderr)
        sys.exit(1)

    # 4. Strip non-schema columns (including 'resolved', raw columns, etc.)
    extra = set(df.columns) - set(schema_fields)
    if extra:
        print(f"Dropping {len(extra)} non-schema columns: {sorted(extra)}")
    out = df[[name for name in schema_fields if name in df.columns]].copy()

    # 5. Coerce types
    for name in out.columns:
        info = schema_fields[name]
        out[name] = _coerce_column(out[name], info["category"])

    # 6. Write parquet
    out.to_parquet(output_path, index=False)
    print(f"Wrote {len(out)} rows, {len(out.columns)} columns to {output_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finalize resolved data against a schema")
    parser.add_argument("resolved_csv", help="Input resolved CSV")
    parser.add_argument("output_parquet", help="Output finalized parquet")
    parser.add_argument(
        "schema_module", help="Dotted module path (e.g. lancell_examples.foo.schema)"
    )
    parser.add_argument("schema_class", help="Schema class name (e.g. GenomicFeatureSchema)")
    parser.add_argument(
        "--column",
        action="append",
        default=[],
        help="KEY=VALUE to add. If VALUE is a column name, copies it; "
        "if 'None'/'null', sets None; otherwise uses as constant.",
    )
    args = parser.parse_args()

    col_defaults = {}
    for item in args.column:
        key, _, value = item.partition("=")
        col_defaults[key] = value

    mod = importlib.import_module(args.schema_module)
    cls = getattr(mod, args.schema_class)

    finalize(args.resolved_csv, args.output_parquet, cls, col_defaults)
