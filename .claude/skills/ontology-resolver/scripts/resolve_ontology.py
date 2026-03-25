"""Ontology resolution for raw obs CSVs.

Resolves free-text biological metadata columns to canonical ontology terms
and CURIEs. Handles control detection, organism-aware development stages,
and writes a fragment CSV with resolved columns plus an ontology_resolved
boolean.

Usage
-----
    python resolve_ontology.py <input_csv> <output_csv> \
        --field <obs_col>:<schema_field>:<entity> [...] \
        [--organism human] \
        [--corrections corrections.json] \
        [--report-dir resolver_reports]

Arguments
---------
input_csv       Path to raw obs CSV (index_col=0).
output_csv      Path to output fragment CSV.

Options
-------
--field             Repeatable. Format: obs_column:schema_field:ENTITY_TYPE.
                    Entity is the OntologyEntity enum name (e.g. CELL_TYPE,
                    TISSUE, DISEASE, ORGANISM, ASSAY, DEVELOPMENT_STAGE,
                    ETHNICITY, SEX, CELL_LINE).
--organism          Organism context for development_stage (default: human).
--corrections       JSON file with correction mappings. Format:
                    {"obs_column": {"original": "corrected", ...}}.
--report-dir        Directory for markdown report (default: resolver_reports
                    in the input CSV's parent directory).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lancell.standardization import (
    OntologyEntity,
    detect_control_labels,
    resolve_ontology_terms,
)
from lancell.standardization.types import CellLineResolution, OntologyResolution


def _get_term_id(res) -> str | None:
    """Extract the ontology/identifier CURIE from a resolution object.

    OntologyResolution has ontology_term_id; CellLineResolution has
    cellosaurus_id. Both inherit from Resolution.
    """
    if isinstance(res, OntologyResolution):
        return res.ontology_term_id
    if isinstance(res, CellLineResolution):
        return res.cellosaurus_id
    return None


def _parse_field(field_str: str) -> tuple[str, str, OntologyEntity]:
    """Parse a --field argument into (obs_column, schema_field, entity)."""
    parts = field_str.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"--field must be obs_column:schema_field:ENTITY_TYPE, got: {field_str!r}"
        )
    obs_col, schema_field, entity_name = parts
    try:
        entity = OntologyEntity[entity_name.upper()]
    except KeyError:
        valid = [e.name for e in OntologyEntity]
        raise ValueError(
            f"Unknown entity type {entity_name!r}. Valid: {valid}"
        ) from None
    return obs_col, schema_field, entity


def _resolve_field(
    raw_obs: pd.DataFrame,
    fragment: pd.DataFrame,
    obs_col: str,
    schema_field: str,
    entity: OntologyEntity,
    organism: str,
    corrections: dict[str, str] | None,
) -> dict:
    """Resolve a single ontology field. Returns stats dict."""
    unique_values = raw_obs[obs_col].dropna().unique().tolist()
    unique_values = [str(v) for v in unique_values]

    # Control detection
    control_mask = detect_control_labels(unique_values)
    control_labels = [v for v, is_ctrl in zip(unique_values, control_mask) if is_ctrl]
    actual_values = [v for v, is_ctrl in zip(unique_values, control_mask) if not is_ctrl]

    # Apply corrections
    corrected_originals = {}
    if corrections:
        corrected_actual = []
        for v in actual_values:
            if v in corrections:
                corrected_actual.append(corrections[v])
                corrected_originals[v] = corrections[v]
            else:
                corrected_actual.append(v)
        actual_values_for_resolution = list(set(corrected_actual))
    else:
        actual_values_for_resolution = actual_values

    # Resolve
    report = resolve_ontology_terms(actual_values_for_resolution, entity, organism=organism)

    # Build lookup from resolved value back to resolution
    resolution_map = {}
    for res in report.results:
        resolution_map[res.input_value] = res

    # Build name/id maps for all original values
    name_map = {}
    id_map = {}

    for v in actual_values:
        lookup_key = corrected_originals.get(v, v)
        res = resolution_map.get(lookup_key)
        if res is not None and res.resolved_value is not None:
            name_map[v] = res.resolved_value
            id_map[v] = _get_term_id(res)
        else:
            # Keep original value for name, None for ontology_id
            name_map[v] = v
            id_map[v] = None

    for ctrl in control_labels:
        name_map[ctrl] = None
        id_map[ctrl] = None

    # Write the schema column to fragment (exact schema field name, no suffix)
    fragment[schema_field] = raw_obs[obs_col].astype(str).map(name_map)

    # Rows where obs_col was NaN should stay NaN in the fragment
    null_mask = raw_obs[obs_col].isna()
    fragment.loc[null_mask, schema_field] = None

    # Track which rows resolved (have an ontology ID) for the ontology_resolved boolean
    resolved_set = {v for v in actual_values if id_map.get(v) is not None}
    control_set = {str(c) for c in control_labels}
    ok_set = resolved_set | control_set
    fragment[f"_resolved_{schema_field}"] = raw_obs[obs_col].astype(str).isin(ok_set)
    fragment.loc[null_mask, f"_resolved_{schema_field}"] = True  # NaN rows are not failures

    # Collect unresolved
    unresolved = [v for v in actual_values if id_map.get(v) is None]

    # Build resolved mappings for the report (canonical name → CURIE)
    resolved_mappings = {
        name_map[v]: id_map[v]
        for v in actual_values
        if id_map.get(v) is not None
    }

    stats = {
        "obs_col": obs_col,
        "schema_field": schema_field,
        "entity": entity.name,
        "total_unique": len(unique_values),
        "actual_values": len(actual_values),
        "controls": control_labels,
        "resolved": report.resolved,
        "unresolved_count": report.unresolved,
        "ambiguous": report.ambiguous,
        "unresolved_values": unresolved,
        "corrections_applied": corrected_originals,
        "resolved_mappings": resolved_mappings,
    }
    return stats


def _write_report(
    report_dir: Path,
    input_path: Path,
    output_path: Path,
    organism: str,
    field_stats: list[dict],
) -> Path:
    """Write markdown report summarizing resolution results."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ontology-resolver.md"

    lines = [
        "# Ontology Resolver Report",
        "",
        f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"**Input:** `{input_path}`",
        f"**Output:** `{output_path}`",
        f"**Organism:** {organism}",
        "",
        "## Field Resolution Summary",
        "",
        "| Field | Entity | Unique | Resolved | Unresolved | Controls |",
        "|---|---|---|---|---|---|",
    ]

    for s in field_stats:
        lines.append(
            f"| {s['schema_field']} | {s['entity']} | {s['total_unique']} "
            f"| {s['resolved']} | {s['unresolved_count']} | {len(s['controls'])} |"
        )

    lines.append("")

    for s in field_stats:
        if s["unresolved_values"]:
            lines.append(f"### Unresolved: {s['schema_field']} ({s['entity']})")
            lines.append("")
            for v in sorted(s["unresolved_values"]):
                lines.append(f"- `{v}`")
            lines.append("")

        if s["controls"]:
            lines.append(f"### Controls detected: {s['schema_field']}")
            lines.append("")
            for v in s["controls"]:
                lines.append(f"- `{v}`")
            lines.append("")

        if s["resolved_mappings"]:
            lines.append(f"### Resolved CURIEs: {s['schema_field']} ({s['entity']})")
            lines.append("")
            for name, curie in sorted(s["resolved_mappings"].items()):
                lines.append(f"- `{name}` → `{curie}`")
            lines.append("")

        if s["corrections_applied"]:
            lines.append(f"### Corrections applied: {s['schema_field']}")
            lines.append("")
            for orig, fixed in sorted(s["corrections_applied"].items()):
                lines.append(f"- `{orig}` → `{fixed}`")
            lines.append("")

    report_path.write_text("\n".join(lines))
    return report_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ontology resolution for raw obs CSVs",
    )
    parser.add_argument("input_csv", type=Path, help="Path to raw obs CSV (index_col=0)")
    parser.add_argument("output_csv", type=Path, help="Path to output fragment CSV")
    parser.add_argument(
        "--field",
        action="append",
        required=True,
        help="obs_column:schema_field:ENTITY_TYPE (repeatable)",
    )
    parser.add_argument("--organism", default="human", help="Organism context (default: human)")
    parser.add_argument("--corrections", type=Path, default=None, help="JSON corrections file")
    parser.add_argument("--report-dir", type=Path, default=None, help="Report output directory")
    args = parser.parse_args(argv)

    # Parse fields
    fields = []
    for f in args.field:
        fields.append(_parse_field(f))

    # Load input
    raw_obs = pd.read_csv(args.input_csv, index_col=0)
    print(f"Loaded {len(raw_obs)} rows from {args.input_csv}")

    # Validate columns exist
    for obs_col, schema_field, entity in fields:
        if obs_col not in raw_obs.columns:
            print(
                f"ERROR: column '{obs_col}' not found. Available: {list(raw_obs.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    # Load corrections
    all_corrections = {}
    if args.corrections:
        with open(args.corrections) as fh:
            all_corrections = json.load(fh)
        print(f"Loaded corrections for columns: {list(all_corrections.keys())}")

    # Initialize fragment
    fragment = pd.DataFrame(index=raw_obs.index)

    # Resolve each field
    all_stats = []
    for obs_col, schema_field, entity in fields:
        print(f"\n--- {obs_col} -> {schema_field} ({entity.name}) ---")
        corrections = all_corrections.get(obs_col, None)

        stats = _resolve_field(
            raw_obs, fragment, obs_col, schema_field, entity, args.organism, corrections
        )
        all_stats.append(stats)

        # Save after each pair
        fragment.to_csv(args.output_csv)

        print(f"  Resolved: {stats['resolved']}/{stats['actual_values']}")
        if stats["controls"]:
            print(f"  Controls: {stats['controls']}")
        if stats["unresolved_values"]:
            print(f"  Unresolved: {stats['unresolved_values']}")
        if stats["corrections_applied"]:
            print(f"  Corrections applied: {stats['corrections_applied']}")

    # Write ontology_resolved boolean from internal tracking columns
    schema_fields = [sf for _, sf, _ in fields]
    resolved_cols = [f"_resolved_{sf}" for sf in schema_fields]
    fragment["ontology_resolved"] = True
    for rc in resolved_cols:
        fragment.loc[~fragment[rc].astype(bool), "ontology_resolved"] = False

    # Drop internal tracking columns — only schema columns + ontology_resolved in output
    fragment.drop(columns=resolved_cols, inplace=True)

    fragment.to_csv(args.output_csv)

    # Write report
    report_dir = args.report_dir or (args.input_csv.parent / "resolver_reports")
    report_path = _write_report(report_dir, args.input_csv, args.output_csv, args.organism, all_stats)

    # Summary
    resolved_count = fragment["ontology_resolved"].sum()
    total_count = len(fragment)
    print(f"\nWrote {args.output_csv}: {total_count} rows")
    print(f"  ontology_resolved: {resolved_count}/{total_count}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
