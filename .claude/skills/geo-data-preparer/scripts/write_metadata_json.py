"""Fetch and save GEO metadata for a GSE (series) or GSM (sample) accession.

Usage:
    python scripts/write_metadata_json.py <data_dir> <accession>

Writes <data_dir>/<accession>_metadata.json with the fetched metadata.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from Bio import Entrez

Entrez.email = "ryan@epiblast.ai"

GEO_QUERY_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"

ACCESSION_RE = re.compile(r"^GS[EM]\d+$")


# --- GSE (series) ---


def _fetch_doi_from_pubmed(pmid: str) -> str | None:
    with Entrez.efetch(db="pubmed", id=pmid, rettype="xml") as handle:
        xml_data = handle.read()
    root = ET.fromstring(xml_data)
    doi_elem = root.find('.//ArticleId[@IdType="doi"]')
    if doi_elem is not None:
        return doi_elem.text
    return None


def _fetch_gse_metadata(accession: str) -> dict:
    with Entrez.esearch(db="gds", term=f"{accession}[ACCN]", retmax=5) as handle:
        search = Entrez.read(handle)

    assert search["IdList"], f"No GDS record found for accession {accession}"

    uid = search["IdList"][0]
    with Entrez.esummary(db="gds", id=uid) as handle:
        summaries = Entrez.read(handle)

    assert summaries, f"No summary returned for UID {uid}"
    doc = summaries[0]

    result = {
        "accession": doc.get("Accession", accession),
        "title": doc.get("title", ""),
        "summary": doc.get("summary", ""),
        "organism": doc.get("taxon", ""),
        "n_samples": int(doc.get("n_samples", 0)),
        "platform": doc.get("GPL", ""),
        "ftp_link": doc.get("FTPLink", ""),
        "pmids": [str(int(x)) for x in doc.get("PubMedIds", [])],
        "doi": None,
    }

    if result["pmids"]:
        result["doi"] = _fetch_doi_from_pubmed(result["pmids"][0])

    return result


# --- GSM (sample) ---


def _parse_soft(text: str) -> dict[str, str | list[str]]:
    parsed: dict[str, str | list[str]] = {}
    for line in text.splitlines():
        if not line.startswith("!"):
            continue
        key, _, value = line[1:].partition(" = ")
        key = key.strip()
        value = value.strip()
        if key in parsed:
            existing = parsed[key]
            if isinstance(existing, list):
                existing.append(value)
            else:
                parsed[key] = [existing, value]
        else:
            parsed[key] = value
    return parsed


def _fetch_gsm_metadata(accession: str) -> dict:
    resp = requests.get(
        GEO_QUERY_URL,
        params={"acc": accession, "targ": "self", "form": "text", "view": "full"},
        timeout=30,
    )
    resp.raise_for_status()

    soft = _parse_soft(resp.text)
    assert soft, f"No SOFT record returned for {accession}"

    characteristics = {}
    raw_chars = soft.get("Sample_characteristics_ch1", [])
    if isinstance(raw_chars, str):
        raw_chars = [raw_chars]
    for ch in raw_chars:
        if ": " in ch:
            k, v = ch.split(": ", 1)
            characteristics[k.strip()] = v.strip()
        else:
            characteristics[ch] = ch

    raw_series = soft.get("Sample_series_id", [])
    if isinstance(raw_series, str):
        raw_series = [raw_series]

    return {
        "accession": soft.get("Sample_geo_accession", accession),
        "title": soft.get("Sample_title", ""),
        "source": soft.get("Sample_source_name_ch1", ""),
        "organism": soft.get("Sample_organism_ch1", ""),
        "characteristics": characteristics,
        "molecule": soft.get("Sample_molecule_ch1", ""),
        "platform": soft.get("Sample_platform_id", ""),
        "description": soft.get("Sample_description", ""),
        "treatment_protocol": soft.get("Sample_treatment_protocol_ch1", ""),
        "growth_protocol": soft.get("Sample_growth_protocol_ch1", ""),
        "extract_protocol": soft.get("Sample_extract_protocol_ch1", ""),
        "data_processing": soft.get("Sample_data_processing", ""),
        "gse": raw_series,
    }


# --- Entry point ---


def _fetch_metadata(accession: str) -> dict:
    if accession.startswith("GSE"):
        return _fetch_gse_metadata(accession)
    elif accession.startswith("GSM"):
        return _fetch_gsm_metadata(accession)
    else:
        raise ValueError(f"Unsupported accession prefix: {accession}. Expected GSE or GSM.")


def write_metadata_json(data_dir: str, accession: str) -> Path:
    assert ACCESSION_RE.match(accession), (
        f"Invalid accession: {accession}. Expected GSE or GSM followed by digits."
    )

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fetching metadata for {accession}...")
    metadata = _fetch_metadata(accession)

    output_path = data_dir / f"{accession}_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {output_path}")
    return output_path


if __name__ == "__main__":
    assert len(sys.argv) == 3, f"Usage: {sys.argv[0]} <data_dir> <accession>"
    write_metadata_json(sys.argv[1], sys.argv[2])
