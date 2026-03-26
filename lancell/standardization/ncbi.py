"""NCBI metadata fetching — GEO, BioSample, BioProject, PubMed, PMC.

Fetches raw metadata from NCBI E-utilities, GEO SOFT files, PubMed,
and PMC Open Access. Resolution through ontology resolvers is the
caller's responsibility.

Rate limiting: 10 req/s with NCBI_API_KEY, 3 req/s without.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from xml.etree import ElementTree

import requests

from lancell.standardization._rate_limit import rate_limited

# ---------------------------------------------------------------------------
# Rate limit — dynamic based on API key
# ---------------------------------------------------------------------------

_NCBI_API_KEY: str | None = os.environ.get("NCBI_API_KEY")
_NCBI_RATE: float = 10.0 if _NCBI_API_KEY else 3.0

_EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_GEO_QUERY_BASE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GeoSeriesMetadata:
    """Metadata from a GEO Series (GSE) record."""

    accession: str
    title: str
    summary: str
    organism: str
    n_samples: int
    platform_ids: list[str]
    ftp_link: str
    pmids: list[str]
    doi: str | None
    bioproject: str | None
    sra_accession: str | None
    samples: list[dict[str, str]]


@dataclass
class GeoSampleMetadata:
    """Metadata from a GEO Sample (GSM) record."""

    accession: str
    title: str
    source: str
    organism: str
    characteristics: dict[str, str]
    molecule: str
    platform: str
    description: str
    treatment_protocol: str
    growth_protocol: str
    extract_protocol: str
    data_processing: str
    series_ids: list[str]
    biosample_accession: str | None


@dataclass
class BioSampleMetadata:
    """Metadata from a BioSample record."""

    accession: str
    title: str
    organism: str
    taxonomy_id: str
    attributes: dict[str, str]


@dataclass
class BioProjectMetadata:
    """Metadata from a BioProject record."""

    accession: str
    title: str
    description: str
    organism: str | None
    data_type: str | None
    scope: str | None


@dataclass
class PublicationMetadata:
    """Metadata for a single publication from PubMed."""

    pmid: int
    doi: str | None
    title: str
    journal: str | None
    publication_date: datetime | None
    authors: list[str] = field(default_factory=list)
    pmc_id: str | None = None


@dataclass
class PublicationSection:
    """A single section from a publication's full text or abstract."""

    section_title: str | None
    section_text: str


@dataclass
class PublicationFullText:
    """Full text of a publication, broken into sections."""

    pmid: int
    pmc_id: str | None
    sections: list[PublicationSection]
    source: str  # "pmc" or "abstract_only"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _entrez_params(**kw: str) -> dict[str, str]:
    """Build E-utilities query params, injecting API key if available."""
    params = dict(kw)
    if _NCBI_API_KEY:
        params["api_key"] = _NCBI_API_KEY
    return params


@rate_limited("ncbi", max_per_second=_NCBI_RATE)
def _entrez_get(endpoint: str, params: dict[str, str]) -> requests.Response:
    """Rate-limited GET to an E-utilities endpoint."""
    url = f"{_EUTILS_BASE}/{endpoint}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp


@rate_limited("ncbi", max_per_second=_NCBI_RATE)
def _geo_soft_get(accession: str) -> str:
    """Rate-limited GET for GEO SOFT-format text."""
    params = {"acc": accession, "targ": "self", "form": "text", "view": "full"}
    resp = requests.get(_GEO_QUERY_BASE, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_soft(text: str) -> dict[str, list[str]]:
    """Parse GEO SOFT format into {field_name: [values]}."""
    result: dict[str, list[str]] = {}
    for line in text.splitlines():
        if line.startswith("!"):
            # Format: !Field_name = value
            match = re.match(r"^!(\S+)\s*=\s*(.*)", line)
            if match:
                key, val = match.group(1), match.group(2).strip()
                result.setdefault(key, []).append(val)
    return result


def _parse_characteristics(values: list[str]) -> dict[str, str]:
    """Parse 'key: value' characteristic lines into a dict."""
    chars: dict[str, str] = {}
    for v in values:
        if ": " in v:
            k, _, val = v.partition(": ")
            chars[k.strip().lower()] = val.strip()
        elif v.strip():
            chars[v.strip().lower()] = ""
    return chars


def _parse_biosample_xml(xml_text: str) -> BioSampleMetadata | None:
    """Parse BioSample XML (efetch result) into BioSampleMetadata."""
    root = ElementTree.fromstring(xml_text)
    bs_elem = root.find(".//BioSample")
    if bs_elem is None:
        return None

    accession = bs_elem.get("accession", "")

    # Title
    title_elem = bs_elem.find(".//Description/Title")
    title = title_elem.text if title_elem is not None and title_elem.text else ""

    # Organism
    org_elem = bs_elem.find(".//Description/Organism")
    organism = ""
    taxonomy_id = ""
    if org_elem is not None:
        organism = org_elem.get("taxonomy_name", "")
        taxonomy_id = org_elem.get("taxonomy_id", "")

    # Attributes — prefer harmonized_name over attribute_name
    attributes: dict[str, str] = {}
    for attr in bs_elem.findall(".//Attribute"):
        name = attr.get("harmonized_name") or attr.get("attribute_name") or ""
        value = attr.text or ""
        if name:
            attributes[name] = value.strip()

    return BioSampleMetadata(
        accession=accession,
        title=title,
        organism=organism,
        taxonomy_id=taxonomy_id,
        attributes=attributes,
    )


def _parse_bioproject_xml(xml_text: str) -> BioProjectMetadata | None:
    """Parse BioProject XML (efetch result) into BioProjectMetadata."""
    root = ElementTree.fromstring(xml_text)

    # Navigate to the project descriptor
    pkg = root.find(".//DocumentSummary") or root.find(".//Package")
    proj = root.find(".//Project/ProjectDescr")
    if proj is None and pkg is not None:
        proj = pkg.find(".//Project/ProjectDescr")
    if proj is None:
        proj = root.find(".//ProjectDescr")

    if proj is None:
        return None

    # Accession
    acc_elem = root.find(".//Project/ProjectID/ArchiveID")
    if acc_elem is None and pkg is not None:
        acc_elem = pkg.find(".//Project/ProjectID/ArchiveID")
    accession = acc_elem.get("accession", "") if acc_elem is not None else ""

    title_elem = proj.find("Title")
    title = title_elem.text if title_elem is not None and title_elem.text else ""

    desc_elem = proj.find("Description")
    description = desc_elem.text if desc_elem is not None and desc_elem.text else ""

    # Organism (from ProjectType)
    org_elem = root.find(".//ProjectType/ProjectTypeSubmission/Target")
    if org_elem is None and pkg is not None:
        org_elem = pkg.find(".//ProjectType/ProjectTypeSubmission/Target")
    organism = org_elem.get("organism", "") if org_elem is not None else None
    organism = organism or None

    # Data type and scope
    data_type_elem = root.find(".//ProjectType/ProjectTypeSubmission/ProjectDataTypeSet/DataType")
    if data_type_elem is None and pkg is not None:
        data_type_elem = pkg.find(
            ".//ProjectType/ProjectTypeSubmission/ProjectDataTypeSet/DataType"
        )
    data_type = data_type_elem.text if data_type_elem is not None and data_type_elem.text else None

    scope_elem = root.find(".//ProjectType/ProjectTypeSubmission/Target/Scope")
    if scope_elem is None and pkg is not None:
        scope_elem = pkg.find(".//ProjectType/ProjectTypeSubmission/Target/Scope")
    if scope_elem is None:
        # Try the Target attribute
        scope = org_elem.get("scope") if org_elem is not None else None
    else:
        scope = scope_elem.text if scope_elem.text else None

    return BioProjectMetadata(
        accession=accession,
        title=title,
        description=description,
        organism=organism,
        data_type=data_type,
        scope=scope,
    )


def _fetch_doi_from_pmid(pmid: str) -> str | None:
    """Fetch DOI for a PubMed ID via efetch."""
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="pubmed", id=pmid, rettype="xml", retmode="xml"),
    )
    try:
        root = ElementTree.fromstring(resp.text)
        for aid in root.findall(".//ArticleId"):
            if aid.get("IdType") == "doi" and aid.text:
                return aid.text
    except ElementTree.ParseError:
        pass
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_geo_metadata(
    accession: str,
) -> GeoSeriesMetadata | GeoSampleMetadata:
    """Dispatch to fetch_geo_series or fetch_geo_sample based on prefix."""
    acc = accession.strip().upper()
    if acc.startswith("GSE"):
        return fetch_geo_series(acc)
    elif acc.startswith("GSM"):
        return fetch_geo_sample(acc)
    else:
        raise ValueError(
            f"Unrecognized GEO accession prefix: {accession!r}. "
            "Expected GSE (series) or GSM (sample)."
        )


def fetch_geo_series(accession: str) -> GeoSeriesMetadata:
    """Fetch GEO Series metadata via E-utilities esearch + esummary."""
    # esearch to get UID
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(db="gds", term=f"{accession}[ACCN]", retmode="json", retmax="1"),
    )
    search_result = resp.json().get("esearchresult", {})
    id_list = search_result.get("idlist", [])
    if not id_list:
        raise ValueError(f"GEO series not found: {accession}")

    uid = id_list[0]

    # esummary to get details
    resp = _entrez_get(
        "esummary.fcgi",
        _entrez_params(db="gds", id=uid, retmode="json"),
    )
    summary_data = resp.json()
    doc = summary_data.get("result", {}).get(uid, {})
    if not doc:
        raise ValueError(f"No esummary data for GEO series {accession} (UID {uid})")

    # Extract fields
    title = doc.get("title", "")
    summary = doc.get("summary", "")
    organism = doc.get("taxon", "")
    n_samples = int(doc.get("n_samples", 0))
    gpl_str = doc.get("gpl", "")
    platform_ids = [f"GPL{g}" for g in gpl_str.split(";") if g.strip()]
    ftp_link = doc.get("ftplink", "")
    pmids = [str(p) for p in doc.get("pubmedids", []) if p]
    bioproject = doc.get("bioproject", "") or None
    sra_accession = None

    # Extract SRA from extrelations
    for rel in doc.get("extrelations", []):
        if rel.get("relationtype") == "SRA" and rel.get("targetobject"):
            sra_accession = rel["targetobject"]
            break

    # Extract samples
    samples = []
    for s in doc.get("samples", []):
        samples.append({"accession": s.get("accession", ""), "title": s.get("title", "")})

    # Resolve DOI from first PMID
    doi = None
    if pmids:
        doi = _fetch_doi_from_pmid(pmids[0])

    result = GeoSeriesMetadata(
        accession=accession,
        title=title,
        summary=summary,
        organism=organism,
        n_samples=n_samples,
        platform_ids=platform_ids,
        ftp_link=ftp_link,
        pmids=pmids,
        doi=doi,
        bioproject=bioproject,
        sra_accession=sra_accession,
        samples=samples,
    )
    return result


def fetch_geo_sample(accession: str) -> GeoSampleMetadata:
    """Fetch GEO Sample metadata from SOFT format."""
    text = _geo_soft_get(accession)
    soft = _parse_soft(text)

    def _first(key: str) -> str:
        vals = soft.get(key, [])
        return vals[0] if vals else ""

    def _all(key: str) -> list[str]:
        return soft.get(key, [])

    title = _first("Sample_title")
    source = _first("Sample_source_name_ch1")
    organism = _first("Sample_organism_ch1")
    characteristics = _parse_characteristics(_all("Sample_characteristics_ch1"))
    molecule = _first("Sample_molecule_ch1")
    platform = _first("Sample_platform_id")
    description = _first("Sample_description")
    treatment_protocol = _first("Sample_treatment_protocol_ch1")
    growth_protocol = _first("Sample_growth_protocol_ch1")
    extract_protocol = _first("Sample_extract_protocol_ch1")
    data_processing = _first("Sample_data_processing")
    series_ids = _all("Sample_series_id")

    # Parse BioSample accession from Sample_relation
    biosample_accession = None
    for rel in _all("Sample_relation"):
        m = re.search(r"SAMN\d+", rel)
        if m:
            biosample_accession = m.group(0)
            break

    result = GeoSampleMetadata(
        accession=accession,
        title=title,
        source=source,
        organism=organism,
        characteristics=characteristics,
        molecule=molecule,
        platform=platform,
        description=description,
        treatment_protocol=treatment_protocol,
        growth_protocol=growth_protocol,
        extract_protocol=extract_protocol,
        data_processing=data_processing,
        series_ids=series_ids,
        biosample_accession=biosample_accession,
    )
    return result


def fetch_biosample(accession: str) -> BioSampleMetadata:
    """Fetch BioSample metadata.

    Accepts a SAMN accession, numeric UID, or GSM accession (searches biosample DB).
    """
    # Determine search term
    acc = accession.strip()
    if acc.startswith("GSM"):
        search_term = acc
    elif acc.startswith("SAMN"):
        search_term = f"{acc}[ACCN]"
    elif acc.isdigit():
        # Numeric UID — go straight to efetch
        search_term = None
        uid = acc
    else:
        search_term = f"{acc}[ACCN]"

    if search_term is not None:
        resp = _entrez_get(
            "esearch.fcgi",
            _entrez_params(db="biosample", term=search_term, retmode="json", retmax="1"),
        )
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            raise ValueError(f"BioSample not found: {accession}")
        uid = id_list[0]

    # efetch
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="biosample", id=uid, rettype="xml", retmode="xml"),
    )
    result = _parse_biosample_xml(resp.text)
    if result is None:
        raise ValueError(f"Failed to parse BioSample XML for {accession}")

    return result


def fetch_bioproject(accession: str) -> BioProjectMetadata:
    """Fetch BioProject metadata."""
    # esearch — bioproject uses [Project Accession] field, not [ACCN]
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(
            db="bioproject",
            term=f"{accession}[Project Accession]",
            retmode="json",
            retmax="1",
        ),
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        raise ValueError(f"BioProject not found: {accession}")
    uid = id_list[0]

    # efetch
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="bioproject", id=uid, rettype="xml", retmode="xml"),
    )
    result = _parse_bioproject_xml(resp.text)
    if result is None:
        raise ValueError(f"Failed to parse BioProject XML for {accession}")

    return result


def link_accessions(accession: str, source_db: str, target_db: str) -> list[str]:
    """Generic elink wrapper — find linked records across NCBI databases.

    Returns a list of target UIDs.
    """
    # esearch source DB
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(db=source_db, term=f"{accession}[ACCN]", retmode="json", retmax="1"),
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        raise ValueError(f"Accession not found in {source_db}: {accession}")
    source_uid = id_list[0]

    # elink (XML mode — more reliable than JSON)
    resp = _entrez_get(
        "elink.fcgi",
        _entrez_params(dbfrom=source_db, db=target_db, id=source_uid, retmode="xml"),
    )

    target_ids: list[str] = []
    try:
        root = ElementTree.fromstring(resp.text)
        for link_set in root.findall(".//LinkSetDb"):
            for link in link_set.findall("Link/Id"):
                if link.text:
                    target_ids.append(link.text)
    except ElementTree.ParseError:
        pass

    return target_ids


def fetch_geo_biosample_attrs(gsm_accession: str) -> dict[str, str]:
    """Convenience: GSM -> BioSample attributes.

    Fetches the GEO sample, follows the BioSample link, and returns the
    BioSample attributes dict.
    """
    sample = fetch_geo_sample(gsm_accession)

    biosample_acc = sample.biosample_accession
    if not biosample_acc:
        # Fallback: try esearch biosample by GSM
        resp = _entrez_get(
            "esearch.fcgi",
            _entrez_params(db="biosample", term=gsm_accession, retmode="json", retmax="1"),
        )
        id_list = resp.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            raise ValueError(f"No BioSample link found for {gsm_accession}")
        # Fetch by UID
        bs = fetch_biosample(id_list[0])
        return bs.attributes

    bs = fetch_biosample(biosample_acc)
    return bs.attributes


# ---------------------------------------------------------------------------
# Publication helpers (private)
# ---------------------------------------------------------------------------

_PMC_ID_CONVERTER = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"


def _detect_identifier_type(identifier: str) -> str:
    """Classify a publication identifier as 'pmid', 'doi', or 'title'."""
    cleaned = identifier.strip()
    if cleaned.startswith("PMID:"):
        return "pmid"
    if cleaned.isdigit():
        return "pmid"
    if cleaned.startswith("10.") or "/10." in cleaned:
        return "doi"
    if cleaned.lower().startswith("doi:"):
        return "doi"
    return "title"


def _search_pmid_by_doi(doi: str) -> int | None:
    """Search PubMed for a DOI and return the PMID."""
    doi = doi.strip().removeprefix("doi:").removeprefix("DOI:").strip()
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(db="pubmed", term=f"{doi}[DOI]", retmode="json", retmax="1"),
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    if id_list:
        return int(id_list[0])
    return None


def _search_pmid_by_title(title: str) -> int | None:
    """Search PubMed by title and return the top-ranked PMID."""
    # Try exact title field search first
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(db="pubmed", term=f"{title}[Title]", retmode="json", retmax="1"),
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    if id_list:
        return int(id_list[0])

    # Fall back to broader search without field tag
    resp = _entrez_get(
        "esearch.fcgi",
        _entrez_params(db="pubmed", term=title, retmode="json", retmax="1"),
    )
    id_list = resp.json().get("esearchresult", {}).get("idlist", [])
    if id_list:
        return int(id_list[0])

    return None


def _parse_pubmed_article(xml_text: str) -> PublicationMetadata:
    """Parse PubMed efetch XML into PublicationMetadata."""
    root = ElementTree.fromstring(xml_text)

    article = root.find(".//PubmedArticle")
    if article is None:
        raise ValueError("No PubmedArticle element found in XML response")

    # PMID
    pmid_elem = article.find(".//MedlineCitation/PMID")
    if pmid_elem is None or not pmid_elem.text:
        raise ValueError("No PMID found in PubMed XML")
    pmid = int(pmid_elem.text)

    # DOI and PMC ID from ArticleIdList
    doi = None
    pmc_id = None
    for aid in article.findall(".//PubmedData/ArticleIdList/ArticleId"):
        id_type = aid.get("IdType", "")
        if id_type == "doi" and aid.text:
            doi = aid.text
        elif id_type == "pmc" and aid.text:
            pmc_id = aid.text

    # Title
    title_elem = article.find(".//Article/ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # Journal
    journal_elem = article.find(".//Article/Journal/Title")
    if journal_elem is None or not journal_elem.text:
        journal_elem = article.find(".//MedlineJournalInfo/MedlineTA")
    journal = journal_elem.text.strip() if journal_elem is not None and journal_elem.text else None

    # Publication date
    publication_date = _parse_pubmed_date(article)

    # Authors
    authors: list[str] = []
    for author in article.findall(".//Article/AuthorList/Author"):
        last = author.find("LastName")
        fore = author.find("ForeName")
        if last is not None and last.text:
            name = last.text
            if fore is not None and fore.text:
                name = f"{fore.text} {last.text}"
            authors.append(name)
        else:
            collective = author.find("CollectiveName")
            if collective is not None and collective.text:
                authors.append(collective.text)

    return PublicationMetadata(
        pmid=pmid,
        doi=doi,
        title=title,
        journal=journal,
        publication_date=publication_date,
        authors=authors,
        pmc_id=pmc_id,
    )


def _parse_pubmed_date(article: ElementTree.Element) -> datetime | None:
    """Extract publication date from a PubMed article element."""
    # Try ArticleDate first (electronic publication date)
    for date_elem in article.findall(".//Article/ArticleDate"):
        year = date_elem.find("Year")
        month = date_elem.find("Month")
        day = date_elem.find("Day")
        if year is not None and year.text:
            return _build_date(year.text, month, day)

    # Fall back to Journal PubDate
    pubdate = article.find(".//Article/Journal/JournalIssue/PubDate")
    if pubdate is not None:
        year = pubdate.find("Year")
        month = pubdate.find("Month")
        day = pubdate.find("Day")
        if year is not None and year.text:
            return _build_date(year.text, month, day)
        # Some articles only have MedlineDate
        medline_date = pubdate.find("MedlineDate")
        if medline_date is not None and medline_date.text:
            # e.g. "2020 Jan-Feb" or "2020 Spring"
            match = re.match(r"(\d{4})", medline_date.text)
            if match:
                return datetime(int(match.group(1)), 1, 1)

    return None


_MONTH_MAP: dict[str, int] = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _build_date(
    year_str: str,
    month_elem: ElementTree.Element | None,
    day_elem: ElementTree.Element | None,
) -> datetime:
    """Build a datetime from year/month/day elements."""
    year = int(year_str)
    month = 1
    day = 1

    if month_elem is not None and month_elem.text:
        m = month_elem.text.strip()
        if m.isdigit():
            month = int(m)
        else:
            month = _MONTH_MAP.get(m.lower()[:3], 1)

    if day_elem is not None and day_elem.text and day_elem.text.strip().isdigit():
        day = int(day_elem.text.strip())

    return datetime(year, month, day)


@rate_limited("ncbi", max_per_second=_NCBI_RATE)
def _convert_pmid_to_pmcid(pmid: int) -> str | None:
    """Convert PMID to PMC ID via the PMC ID converter API."""
    resp = requests.get(
        _PMC_ID_CONVERTER,
        params={"ids": str(pmid), "format": "json"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    records = data.get("records", [])
    if records and records[0].get("pmcid"):
        return records[0]["pmcid"]
    return None


def _fetch_pmc_sections(pmc_id: str) -> list[PublicationSection]:
    """Fetch full text from PMC and parse into sections."""
    # Strip "PMC" prefix for the API call
    numeric_id = pmc_id.removeprefix("PMC")
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="pmc", id=numeric_id, rettype="xml", retmode="xml"),
    )

    root = ElementTree.fromstring(resp.text)
    body = root.find(".//body")
    if body is None:
        return []

    sections: list[PublicationSection] = []
    _walk_sections(body, parent_title=None, sections=sections)
    return sections


def _walk_sections(
    elem: ElementTree.Element,
    parent_title: str | None,
    sections: list[PublicationSection],
) -> None:
    """Recursively walk <sec> elements and collect sections."""
    for sec in elem.findall("sec"):
        title_elem = sec.find("title")
        section_title = (
            title_elem.text.strip() if title_elem is not None and title_elem.text else None
        )

        if parent_title and section_title:
            section_title = f"{parent_title} > {section_title}"
        elif parent_title:
            section_title = parent_title

        # Collect paragraph text from direct <p> children
        paragraphs: list[str] = []
        for p in sec.findall("p"):
            text = "".join(p.itertext()).strip()
            if text:
                paragraphs.append(text)

        if paragraphs:
            sections.append(
                PublicationSection(
                    section_title=section_title,
                    section_text="\n\n".join(paragraphs),
                )
            )

        # Recurse into nested <sec> elements
        _walk_sections(sec, parent_title=section_title, sections=sections)


def _extract_abstract_sections(xml_text: str) -> list[PublicationSection]:
    """Extract abstract sections from PubMed efetch XML."""
    root = ElementTree.fromstring(xml_text)

    sections: list[PublicationSection] = []

    # Handle structured abstracts with labeled sections
    abstract_texts = root.findall(".//Abstract/AbstractText")
    if abstract_texts:
        for at in abstract_texts:
            label = at.get("Label")
            text = "".join(at.itertext()).strip()
            if text:
                section_title = label if label else "Abstract"
                sections.append(
                    PublicationSection(
                        section_title=section_title,
                        section_text=text,
                    )
                )
    else:
        # Try unstructured abstract
        abstract_elem = root.find(".//Abstract")
        if abstract_elem is not None:
            text = "".join(abstract_elem.itertext()).strip()
            if text:
                sections.append(
                    PublicationSection(
                        section_title="Abstract",
                        section_text=text,
                    )
                )

    return sections


# ---------------------------------------------------------------------------
# Publication API (public)
# ---------------------------------------------------------------------------


def fetch_publication(identifier: str) -> PublicationMetadata:
    """Fetch publication metadata from PubMed.

    Accepts a PMID (numeric string), DOI, or paper title.
    Auto-detects the identifier type and resolves to a PubMed record.

    Raises ValueError if the identifier cannot be resolved.
    """
    id_type = _detect_identifier_type(identifier)

    if id_type == "pmid":
        pmid_str = identifier.strip().removeprefix("PMID:").strip()
        pmid = int(pmid_str)
    elif id_type == "doi":
        pmid = _search_pmid_by_doi(identifier)
        if pmid is None:
            raise ValueError(
                f"Could not find PubMed record for DOI: {identifier}. "
                "The paper may not be indexed in PubMed yet."
            )
    else:
        pmid = _search_pmid_by_title(identifier)
        if pmid is None:
            raise ValueError(
                f"Could not find PubMed record for title: {identifier!r}. "
                "Try searching with a more specific title or use a PMID/DOI instead."
            )

    # Fetch full PubMed record
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="pubmed", id=str(pmid), rettype="xml", retmode="xml"),
    )
    pub = _parse_pubmed_article(resp.text)

    # If no PMC ID from PubMed, try the converter API
    if pub.pmc_id is None:
        pub.pmc_id = _convert_pmid_to_pmcid(pmid)

    return pub


def fetch_publication_text(pmid: int, pmc_id: str | None = None) -> PublicationFullText:
    """Fetch full text sections for a publication.

    Tries PMC full text first (if pmc_id is available), then falls back
    to the PubMed abstract.
    """
    # Try PMC full text
    if pmc_id:
        sections = _fetch_pmc_sections(pmc_id)
        if sections:
            return PublicationFullText(pmid=pmid, pmc_id=pmc_id, sections=sections, source="pmc")

    # Fall back to abstract from PubMed
    resp = _entrez_get(
        "efetch.fcgi",
        _entrez_params(db="pubmed", id=str(pmid), rettype="xml", retmode="xml"),
    )
    sections = _extract_abstract_sections(resp.text)
    return PublicationFullText(pmid=pmid, pmc_id=pmc_id, sections=sections, source="abstract_only")


def search_pubmed_by_title(title: str) -> int | None:
    """Search PubMed by paper title and return the PMID, or None."""
    return _search_pmid_by_title(title)


def fetch_publication_metadata(pmid: str | int) -> dict:
    """Fetch publication metadata as a dict (backward-compatible).

    Returns a dict with keys: pmid, doi, title, journal, publication_date,
    full_text. The full_text field concatenates all sections with headers.
    """
    pub = fetch_publication(str(pmid))
    text_data = fetch_publication_text(pub.pmid, pub.pmc_id)

    # Build full_text string from sections
    parts: list[str] = []
    for section in text_data.sections:
        if section.section_title:
            parts.append(f"## {section.section_title}\n\n{section.section_text}")
        else:
            parts.append(section.section_text)
    full_text = "\n\n".join(parts)

    return {
        "pmid": pub.pmid,
        "doi": pub.doi,
        "title": pub.title,
        "journal": pub.journal,
        "publication_date": (pub.publication_date.isoformat() if pub.publication_date else None),
        "full_text": full_text,
    }
