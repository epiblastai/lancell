"""List supplementary files available for a GEO accession.

Supports both series (GSE) and sample (GSM) accessions, detected
automatically from the prefix.

Usage:
    python scripts/list_geo_files.py GSE123456
    python scripts/list_geo_files.py GSM654321
"""

import ftplib
import sys

GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"


def _build_ftp_path(accession: str) -> str:
    prefix = accession[:-3] + "nnn"
    if accession.startswith("GSE"):
        return f"/geo/series/{prefix}/{accession}/suppl/"
    elif accession.startswith("GSM"):
        return f"/geo/samples/{prefix}/{accession}/suppl/"
    else:
        raise ValueError(f"Unsupported accession prefix: {accession}. Expected GSE or GSM.")


def list_geo_files(accession: str) -> list[str]:
    ftp_path = _build_ftp_path(accession)
    ftp = ftplib.FTP(GEO_FTP_HOST)
    ftp.login()
    try:
        files = ftp.nlst(ftp_path)
    except ftplib.error_perm:
        return []
    finally:
        ftp.quit()
    # nlst returns full paths; extract just the filenames
    return [f.rsplit("/", 1)[-1] for f in files if f.strip()]


if __name__ == "__main__":
    assert len(sys.argv) == 2, f"Usage: {sys.argv[0]} <GEO_ACCESSION>"
    accession = sys.argv[1]
    files = list_geo_files(accession)
    if files:
        for f in files:
            print(f)
    else:
        print(f"No supplementary files found for {accession}", file=sys.stderr)
        sys.exit(1)
