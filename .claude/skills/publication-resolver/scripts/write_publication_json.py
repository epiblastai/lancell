"""Fetch publication metadata from PubMed/PMC and write publication.json.

Three modes of operation:

1. From metadata JSON (default): reads <data_dir>/{accession}_metadata.json if
   present, otherwise <data_dir>/metadata.json, and extracts PMIDs from
   series_metadata.pmids (or resolves GSE from sample_metadata.gse).
2. --pmid: fetch metadata for a specific PMID.
3. --title: search PubMed by title to find the PMID, then fetch metadata.

Usage:
    python scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]
"""

import argparse
import json
from pathlib import Path

from lancell.standardization import (
    fetch_geo_series,
    fetch_publication,
    fetch_publication_metadata,
    fetch_publication_text,
    search_pubmed_by_title,
)


def _resolve_pmids_from_gse(gse_accession: str) -> list[str]:
    """Fetch PMIDs for a GSE accession via GEO series metadata."""
    series = fetch_geo_series(gse_accession)
    return series.pmids


def _extract_pmids_from_metadata(metadata: dict) -> list[str]:
    """Extract PMIDs from a metadata.json dict.

    Looks through all entries for series_metadata.pmids first, then falls back
    to resolving GSE accessions found in sample_metadata.gse.
    """
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

    # Deduplicate while preserving order
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


def write_publication_json(
    data_dir: str, pmid: str | None = None, title: str | None = None
) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if pmid:
        resolved_pmid = pmid
    elif title:
        print(f"Searching PubMed for title: {title}")
        resolved_pmid = search_pubmed_by_title(title)
        assert resolved_pmid, f"No PubMed result found for title: {title}"
        print(f"Found PMID: {resolved_pmid}")
    else:
        metadata_path = _find_metadata_path(data_dir)
        assert metadata_path is not None, (
            f"No metadata JSON found in {data_dir}. Use --pmid or --title."
        )
        metadata = json.loads(metadata_path.read_text())
        pmids = _extract_pmids_from_metadata(metadata)
        assert pmids, f"No PMIDs found in {metadata_path.name}. Use --pmid or --title."
        resolved_pmid = pmids[0]
        if len(pmids) > 1:
            print(f"Multiple PMIDs found: {pmids}. Using first: {resolved_pmid}")

    print(f"Fetching publication metadata for PMID {resolved_pmid}...")
    pub = fetch_publication_metadata(resolved_pmid)
    publication = fetch_publication(str(resolved_pmid))
    text_data = fetch_publication_text(publication.pmid, publication.pmc_id)
    pub["authors"] = publication.authors
    pub["text_source"] = text_data.source

    output_path = data_dir / "publication.json"
    output_path.write_text(json.dumps(pub, indent=2))
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch publication metadata and write publication.json"
    )
    parser.add_argument("data_dir", help="Directory to write publication.json to")
    parser.add_argument("--pmid", help="PubMed ID to fetch directly")
    parser.add_argument("--title", help="Paper title to search PubMed for")
    args = parser.parse_args()

    assert not (args.pmid and args.title), "Cannot specify both --pmid and --title"
    write_publication_json(args.data_dir, pmid=args.pmid, title=args.title)
