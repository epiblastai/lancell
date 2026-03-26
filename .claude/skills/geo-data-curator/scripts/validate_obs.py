"""Validate a standardized obs CSV against the obs (LancellBaseSchema) schema.

Strips non-schema columns, adds missing nullable columns as None, applies
``--column`` defaults, and coerces types (JSON lists, bools, numerics).
Does NOT do per-row pydantic validation — type errors surface at LanceDB
insertion time.

Usage:
    python validate_obs.py <standardized_obs_csv> <output_parquet> \
        <schema_module> <schema_class> [--column KEY=VALUE ...]

Example:
    python validate_obs.py \
        /tmp/geo_agent/GSE123/HepG2/gene_expression_standardized_obs.csv \
        /tmp/geo_agent/GSE123/HepG2/gene_expression_validated_obs.parquet \
        lancell_examples.multimodal_perturbation_atlas.schema \
        CellIndex \
        --column cell_type=None --column days_in_vitro=3.0
"""

import argparse
import importlib
import json
import sys
from types import UnionType
from typing import Union, get_args, get_origin

import pandas as pd
from pydantic_core import PydanticUndefined

from lancell.schema import AUTO_FIELDS, DenseZarrPointer, SparseZarrPointer


def _is_zarr_pointer_field(annotation: type) -> bool:
    """Check if a type annotation is a ZarrPointer (possibly Optional)."""
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        inner = [a for a in get_args(annotation) if a is not type(None)]
        if len(inner) == 1:
            annotation = inner[0]
    return annotation is SparseZarrPointer or annotation is DenseZarrPointer


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
    """Coerce a pandas Series so its dtype matches what pyarrow expects.

    After coercion, pa.array(series.values, type=arrow_type) must work
    without any further casting.
    """
    if category == "list":
        # Parse JSON strings to actual Python lists. Parquet preserves them.
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
        # Nullable int: use pd.array with Int64 dtype
        return series.astype("Int64")

    if category == "float":
        return series.astype("Float64")

    # str: cast to string, preserving nulls as None.
    # Handles int→str (e.g. batch_id) and all-null float64 columns.
    return series.apply(
        lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    )


def _get_obs_fields(schema_class: type) -> dict[str, dict]:
    """Extract user-facing obs fields from a schema class.

    Returns {field_name: {"category": str, "nullable": bool, "has_default": bool}}
    Excludes AUTO_FIELDS (uid, dataset_uid) and ZarrPointer fields.
    Auto-computed fields are included here but filled by compute_auto_fields after coercion.
    """
    skip = AUTO_FIELDS

    fields = {}
    for name, field_info in schema_class.model_fields.items():
        if name in skip:
            continue
        if _is_zarr_pointer_field(field_info.annotation):
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


def validate_obs(
    standardized_path: str,
    output_path: str,
    schema_class: type,
    column_defaults: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(standardized_path, index_col=0)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {standardized_path}")

    # 1. Apply --column defaults
    for col, value in (column_defaults or {}).items():
        if value in df.columns:
            df[col] = df[value]
        elif value.lower() in ("none", "null"):
            df[col] = None
        else:
            df[col] = value

    # 2. Compute auto-generated fields so they go through validation too
    df = schema_class.compute_auto_fields(df)

    # 3. Identify schema fields
    obs_fields = _get_obs_fields(schema_class)
    print(f"Schema obs fields: {len(obs_fields)}")

    # 4. Fill missing nullable columns with None, error on missing required
    missing = set(obs_fields) - set(df.columns)
    errors = []
    for name in sorted(missing):
        info = obs_fields[name]
        if info["nullable"] or info["has_default"]:
            df[name] = None
        else:
            errors.append(name)

    if errors:
        print(f"ERROR: Missing required non-nullable columns: {errors}", file=sys.stderr)
        print("Use --column KEY=VALUE to provide them.", file=sys.stderr)
        sys.exit(1)

    # 5. Strip non-schema columns
    extra = set(df.columns) - set(obs_fields)
    if extra:
        print(f"Dropping {len(extra)} non-schema columns: {sorted(extra)}")
    out = df[[name for name in obs_fields if name in df.columns]].copy()

    # 6. Coerce types
    for name in out.columns:
        info = obs_fields[name]
        out[name] = _coerce_column(out[name], info["category"])

    out.to_parquet(output_path)
    print(f"Wrote {len(out)} rows, {len(out.columns)} columns to {output_path}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate standardized obs against schema")
    parser.add_argument("standardized_obs_csv", help="Input standardized obs CSV")
    parser.add_argument("output_parquet", help="Output validated obs parquet")
    parser.add_argument(
        "schema_module", help="Dotted module path (e.g. lancell_examples.foo.schema)"
    )
    parser.add_argument("schema_class", help="Schema class name (e.g. CellIndex)")
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

    validate_obs(args.standardized_obs_csv, args.output_parquet, cls, col_defaults)
