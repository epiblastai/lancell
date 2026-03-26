---
name: publication-resolver
description: Fetch publication metadata (title, DOI, journal, date) from PubMed and full text sections from PMC for PublicationSchema and PublicationSectionSchema. Accepts PMID, DOI, or paper title. Use when a dataset needs publication metadata for its publication.json sidecar or when populating publication records in a LanceDB table.
---

# Publication Resolver

Fetch publication metadata from PubMed and full text from PMC Open Access. Produces validated parquet files for `PublicationSchema` and (optionally) `PublicationSectionSchema` — ready for direct LanceDB ingestion.

## Interface

**Input:** A publication identifier — PMID (numeric), DOI, or paper title. A target schema module and publication schema class name.

**Output:**

*Accession-level (global foreign key tables):*
- `PublicationSchema.parquet` — finalized against the target schema with correct types. Contains exactly the schema fields, one row per publication. Parquet preserves types so the file can be loaded directly into LanceDB.
- `PublicationSectionSchema.parquet` — one row per text section (abstract paragraphs or PMC full-text sections), with `publication_uid` FK. Only produced when `--section-schema` is provided and the target schema includes a section table.
- `publication.json` — backward-compatible sidecar for human inspection and downstream use. Includes the generated `publication_uid` so the curator can reference it when building `DatasetSchema` records.
- `resolver_reports/publication-resolver.md` — markdown report summarizing the identifier used, fetched metadata, full-text availability, output paths, field completeness audit, and any missing fields with reasons.

**Column naming:** No `validated_` prefix. Use schema field names directly.

## Reporting

Each run must write a markdown report to `resolver_reports/` in the working directory.

- Create the directory if it does not exist.
- Default report path: `resolver_reports/publication-resolver.md`
- Overwrite the report for the current run unless the caller asks for a different naming scheme.
- Include:
  - input identifier(s)
  - output file path(s)
  - PMID/DOI/title/journal/date found
  - whether PMC full text was available
  - number of text sections produced
  - schema field completeness audit, including reasons for blanks

## Scripts

Run these via Bash from the **repository root**.

| Script | Usage | Purpose |
|--------|-------|---------|
| `scripts/write_publication_parquet.py` | `python .claude/skills/publication-resolver/scripts/write_publication_parquet.py <data_dir> <schema_module> <pub_schema_class> [--section-schema <section_class>] [--pmid PMID] [--title TITLE]` | **Primary.** Fetch publication metadata, write validated parquet files + publication.json |
| `scripts/write_publication_json.py` | `python .claude/skills/publication-resolver/scripts/write_publication_json.py <data_dir> [--pmid PMID] [--title TITLE]` | **Legacy.** Write only publication.json (no schema validation or parquet) |

### `write_publication_parquet.py`

The primary script. Prefer `--pmid` when you have one. Supports three identifier modes:

1. **From metadata JSON** (default): reads `<data_dir>/{accession}_metadata.json` if present, otherwise `<data_dir>/metadata.json`, and extracts PMIDs.
2. **`--pmid PMID`**: fetch metadata for a specific PubMed ID.
3. **`--title TITLE`**: search PubMed by title, then fetch metadata.

| Argument | Description |
|---|---|
| `data_dir` | Directory to write output files to |
| `schema_module` | Dotted module path (e.g. `lancell_examples.multimodal_perturbation_atlas.schema`) |
| `pub_schema_class` | Publication schema class name (e.g. `PublicationSchema`) |
| `--section-schema` | Section schema class name (e.g. `PublicationSectionSchema`). Omit if schema has no section table. |
| `--pmid` | PubMed ID to fetch directly |
| `--title` | Paper title to search PubMed for |

Example:

```bash
python .claude/skills/publication-resolver/scripts/write_publication_parquet.py \
    /tmp/geo_agent/GSE123456 \
    lancell_examples.multimodal_perturbation_atlas.schema \
    PublicationSchema \
    --section-schema PublicationSectionSchema \
    --pmid 31806696
```

Outputs:
- `PublicationSchema.parquet` — 1 row with uid, doi, pmid, title, journal, publication_date
- `PublicationSectionSchema.parquet` — N rows (one per section) with publication_uid, section_text, section_title
- `publication.json` — backward-compatible JSON with `publication_uid` included

## Imports

```python
from lancell.standardization import (
    fetch_publication,
    fetch_publication_text,
    fetch_publication_metadata,
    search_pubmed_by_title,
    PublicationMetadata,
    PublicationFullText,
    PublicationSection,
)
from lancell.schema import make_uid
```

## Workflow

### 1. Identify the publication

Determine the identifier type. `fetch_publication()` auto-detects:
- Pure digits or `PMID:` prefix → PMID
- Starts with `10.` or contains `/10.` → DOI
- Otherwise → title search

```python
pub = fetch_publication("31806696")         # by PMID
pub = fetch_publication("10.1016/j.cell.2017.10.049")  # by DOI
pub = fetch_publication("Massively multiplex chemical transcriptomics")  # by title
```

### 2. Fetch metadata

`fetch_publication()` returns a `PublicationMetadata` dataclass:

```python
pub = fetch_publication(identifier)
print(f"PMID: {pub.pmid}")
print(f"DOI: {pub.doi}")
print(f"Title: {pub.title}")
print(f"Journal: {pub.journal}")
print(f"Date: {pub.publication_date}")
print(f"Authors: {pub.authors}")
print(f"PMC ID: {pub.pmc_id}")
```

### 3. Fetch full text (if needed by schema)

Only fetch full text if the target schema includes `PublicationSectionSchema`. If the schema only has `PublicationSchema`, the metadata from step 2 is sufficient.

```python
text = fetch_publication_text(pub.pmid, pub.pmc_id)
print(f"Source: {text.source}")  # "pmc" or "abstract_only"
for section in text.sections:
    print(f"  [{section.section_title}] {section.section_text[:100]}...")
```

PMC full text is attempted first. If the article is not in PMC Open Access, the abstract is returned as a single section (or multiple sections for structured abstracts with labeled parts like "Background", "Methods", "Results").

### 4. Assign UID and build schema records

Generate a publication UID via `make_uid()` — this UID is used as the primary key in `PublicationSchema` and as the foreign key in `PublicationSectionSchema`.

```python
from lancell.schema import make_uid

publication_uid = make_uid()

# PublicationSchema record
pub_record = {
    "uid": publication_uid,
    "doi": pub.doi or "",
    "pmid": pub.pmid,
    "title": pub.title,
    "journal": pub.journal,
    "publication_date": pub.publication_date,
}

# PublicationSectionSchema records (one per section)
section_records = [
    {
        "publication_uid": publication_uid,
        "section_text": section.section_text,
        "section_title": section.section_title,
    }
    for section in text.sections
]
```

### 5. Finalize and write parquet

Build DataFrames, coerce types against the target schema, and write parquet. The script handles this automatically:

```bash
python .claude/skills/publication-resolver/scripts/write_publication_parquet.py \
    /tmp/geo_agent/GSE123456 \
    lancell_examples.multimodal_perturbation_atlas.schema \
    PublicationSchema \
    --section-schema PublicationSectionSchema \
    --pmid 31806696
```

The script also writes `publication.json` with the `publication_uid` included for backward compatibility.

### 6. Write the markdown report

After finalization, write `resolver_reports/publication-resolver.md` in the working directory with:
- Input identifier and resolution method
- Output file paths
- PMID, DOI, title, journal, publication date
- PMC full-text availability and section count
- Schema field completeness audit (for each field: populated or reason for blank)

## Rules

- **PMC before abstract.** Always attempt PMC full text before falling back to abstract. Many biology papers have PMC full text (~40% of PubMed-indexed papers).
- **DOI is best-effort.** If no DOI is found in the PubMed record, the field will be None. Do not fail on missing DOI.
- **Multiple PMIDs.** For datasets with multiple associated publications (multiple PMIDs in metadata.json), process each one. The shared script uses the first PMID by default.
- **Title mismatches are not necessarily errors.** If the caller provides a title and PubMed returns a slightly different canonical published title, prefer the PubMed title.
- **Identifier auto-detection.** `fetch_publication()` auto-detects PMID vs DOI vs title. No need to classify the identifier manually.
- **PubMed-only.** Resolution goes through PubMed. Papers not yet indexed in PubMed (e.g., very recent preprints with only a DOI) will fail with a clear error message.
- **Section titles for PMC.** PMC full text sections use the article's own headings (e.g., "Introduction", "Methods", "Results"). Nested subsections are flattened with `>` separators (e.g., "Methods > Cell Culture").
- **Structured abstracts.** Some PubMed abstracts have labeled sections (e.g., "Background", "Methods", "Conclusions"). These are returned as separate `PublicationSection` entries rather than a single block.
- **UID ownership.** The publication resolver generates and owns the `publication_uid`. The curator reads it from the parquet (or from `publication.json`) — it does not generate its own.
