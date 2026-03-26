"""Fetch publication metadata from PubMed/PMC and write validated parquet files.

Produces:
- PublicationSchema.parquet — single-row table with UID, DOI, PMID, etc.
- PublicationSectionSchema.parquet — one row per text section (optional)
- publication.json — backward-compatible sidecar (includes generated publication_uid)

Three modes of operation (same as write_publication_json.py):

1. From metadata JSON (default): reads <data_dir>/{accession}_metadata.json or
   <data_dir>/metadata.json and extracts PMIDs.
2. --pmid: fetch metadata for a specific PMID.
3. --title: search PubMed by title to find the PMID, then fetch metadata.

Usage:
    python scripts/write_publication_parquet.py <data_dir> \
        <schema_module> <publication_schema_class> \
        [--section-schema <section_schema_class>] \
        [--pmid PMID] [--title TITLE]
"""

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

import pandas as pd
from pydantic_core import PydanticUndefined

from lancell.schema import make_stable_uid
from lancell.standardization import (
    fetch_geo_series,
    fetch_publication,
    fetch_publication_metadata,
    fetch_publication_text,
    search_pubmed_by_title,
)

# ---------------------------------------------------------------------------
# PMID resolution helpers (shared with write_publication_json.py)
# ---------------------------------------------------------------------------


def _resolve_pmids_from_gse(gse_accession: str) -> list[str]:
    series = fetch_geo_series(gse_accession)
    return series.pmids


def _extract_pmids_from_metadata(metadata: dict) -> list[str]:
    pmids: list[str] = []
    gse_accessions: list[str] = []

    for entry in metadata.values():
        series = entry.get("series_metadata")
        if series and series.get("pmids"):
            pmids.extend(series["pmids"])
        sample = entry.get("sample_metadata")
        if sample and sample.get("gse"):
            gse_list = sample["gse"]
            if isinstance(gse_list, str):
                gse_list = [gse_list]
            gse_accessions.extend(gse_list)

    if not pmids and gse_accessions:
        seen = set()
        for gse in gse_accessions:
            if gse not in seen:
                seen.add(gse)
                print(f"Resolving PMIDs for {gse}...")
                pmids.extend(_resolve_pmids_from_gse(gse))

    seen_pmids: set[str] = set()
    unique: list[str] = []
    for pmid in pmids:
        if pmid not in seen_pmids:
            seen_pmids.add(pmid)
            unique.append(pmid)
    return unique


def _find_metadata_path(data_dir: Path) -> Path | None:
    accession_metadata = sorted(data_dir.glob("*_metadata.json"))
    if accession_metadata:
        return accession_metadata[0]
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        return metadata_path
    return None


# ---------------------------------------------------------------------------
# Type coercion helpers (same pattern as finalize_features.py)
# ---------------------------------------------------------------------------

_ATLAS_AUTO_FIELDS = {"global_index"}


def _get_field_type_category(annotation: type) -> str:
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
    if annotation is datetime:
        return "datetime"
    return "str"


def _is_nullable(annotation: type) -> bool:
    origin = get_origin(annotation)
    if origin is Union or isinstance(annotation, UnionType):
        return type(None) in get_args(annotation)
    return False


def _get_schema_fields(schema_class: type) -> dict[str, dict]:
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


def _coerce_column(series: pd.Series, category: str) -> pd.Series:
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

    if category == "datetime":

        def _to_datetime(v):
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return None
            if isinstance(v, datetime):
                return v
            return datetime.fromisoformat(str(v))

        return series.apply(_to_datetime)

    # str
    return series.apply(
        lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
    )


def _finalize_df(df: pd.DataFrame, schema_class: type) -> pd.DataFrame:
    """Strip non-schema columns, fill missing nullable fields, coerce types."""
    schema_fields = _get_schema_fields(schema_class)

    # Fill missing nullable/defaulted columns with None
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
        sys.exit(1)

    # Keep only schema columns
    extra = set(df.columns) - set(schema_fields)
    if extra:
        print(f"  Dropping non-schema columns: {sorted(extra)}")
    out = df[[name for name in schema_fields if name in df.columns]].copy()

    # Coerce types
    for name in out.columns:
        info = schema_fields[name]
        out[name] = _coerce_column(out[name], info["category"])

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_pmid(data_dir: Path, pmid: str | None, title: str | None) -> str:
    """Resolve a PMID from arguments or metadata files."""
    if pmid:
        return pmid
    if title:
        print(f"Searching PubMed for title: {title}")
        resolved = search_pubmed_by_title(title)
        assert resolved, f"No PubMed result found for title: {title}"
        print(f"Found PMID: {resolved}")
        return str(resolved)

    metadata_path = _find_metadata_path(data_dir)
    assert metadata_path is not None, (
        f"No metadata JSON found in {data_dir}. Use --pmid or --title."
    )
    metadata = json.loads(metadata_path.read_text())
    pmids = _extract_pmids_from_metadata(metadata)
    assert pmids, f"No PMIDs found in {metadata_path.name}. Use --pmid or --title."
    if len(pmids) > 1:
        print(f"Multiple PMIDs found: {pmids}. Using first: {pmids[0]}")
    return pmids[0]


def write_publication_parquet(
    data_dir: str,
    schema_module: str,
    publication_schema_class: str,
    section_schema_class: str | None = None,
    pmid: str | None = None,
    title: str | None = None,
) -> dict[str, Path]:
    """Fetch publication data and write validated parquet files.

    Returns a dict of output file paths.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Import schema classes
    mod = importlib.import_module(schema_module)
    pub_schema = getattr(mod, publication_schema_class)
    section_schema = getattr(mod, section_schema_class) if section_schema_class else None

    # Resolve PMID
    resolved_pmid = resolve_pmid(data_dir, pmid, title)

    # Fetch from PubMed/PMC
    print(f"Fetching publication metadata for PMID {resolved_pmid}...")
    publication = fetch_publication(str(resolved_pmid))
    pub_dict = fetch_publication_metadata(resolved_pmid)
    text_data = fetch_publication_text(publication.pmid, publication.pmc_id)

    # Generate stable UID from PMID
    publication_uid = make_stable_uid(str(resolved_pmid))

    outputs: dict[str, Path] = {}

    # --- PublicationSchema.parquet ---
    pub_row = {
        "uid": publication_uid,
        "doi": publication.doi or "",
        "pmid": publication.pmid,
        "title": publication.title,
        "journal": publication.journal,
        "publication_date": publication.publication_date,
    }
    pub_df = pd.DataFrame([pub_row])
    pub_df = _finalize_df(pub_df, pub_schema)

    pub_parquet_path = data_dir / f"{publication_schema_class}.parquet"
    pub_df.to_parquet(pub_parquet_path, index=False)
    print(f"Wrote {pub_parquet_path.name}: 1 row, {len(pub_df.columns)} columns")
    outputs["publication_parquet"] = pub_parquet_path

    # --- PublicationSectionSchema.parquet (optional) ---
    if section_schema and text_data.sections:
        section_rows = [
            {
                "publication_uid": publication_uid,
                "section_text": section.section_text,
                "section_title": section.section_title,
            }
            for section in text_data.sections
        ]
        section_df = pd.DataFrame(section_rows)
        section_df = _finalize_df(section_df, section_schema)

        section_parquet_path = data_dir / f"{section_schema_class}.parquet"
        section_df.to_parquet(section_parquet_path, index=False)
        print(
            f"Wrote {section_parquet_path.name}: {len(section_df)} rows, {len(section_df.columns)} columns"
        )
        outputs["section_parquet"] = section_parquet_path
    elif section_schema:
        print("No text sections available (no PMC full text or abstract), skipping section parquet")

    # --- publication.json (backward compat) ---
    pub_dict["authors"] = publication.authors
    pub_dict["text_source"] = text_data.source
    pub_dict["publication_uid"] = publication_uid

    json_path = data_dir / "publication.json"
    json_path.write_text(json.dumps(pub_dict, indent=2))
    print(f"Wrote {json_path.name}")
    outputs["publication_json"] = json_path

    # --- Summary ---
    print(f"\nPublication: {publication.title[:80]}...")
    print(f"  UID: {publication_uid}")
    print(f"  PMID: {publication.pmid}, DOI: {publication.doi}")
    print(f"  Text source: {text_data.source}, Sections: {len(text_data.sections)}")

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch publication metadata and write validated parquet files"
    )
    parser.add_argument("data_dir", help="Directory to write output files to")
    parser.add_argument(
        "schema_module", help="Dotted module path (e.g. lancell_examples.foo.schema)"
    )
    parser.add_argument(
        "publication_schema_class", help="Publication schema class name (e.g. PublicationSchema)"
    )
    parser.add_argument(
        "--section-schema",
        dest="section_schema_class",
        default=None,
        help="Section schema class name (e.g. PublicationSectionSchema). Omit if schema has no section table.",
    )
    parser.add_argument("--pmid", help="PubMed ID to fetch directly")
    parser.add_argument("--title", help="Paper title to search PubMed for")
    args = parser.parse_args()

    assert not (args.pmid and args.title), "Cannot specify both --pmid and --title"

    write_publication_parquet(
        data_dir=args.data_dir,
        schema_module=args.schema_module,
        publication_schema_class=args.publication_schema_class,
        section_schema_class=args.section_schema_class,
        pmid=args.pmid,
        title=args.title,
    )
