"""Download a supplementary file from GEO via FTP.

Usage:
    python scripts/download_geo_file.py GSE123456 filename.h5ad [dest_dir]

Default dest_dir: /tmp/geo_agent/<accession>/
"""

import ftplib
import os
import sys

GEO_FTP_HOST = "ftp.ncbi.nlm.nih.gov"


def _build_ftp_path(accession: str, filename: str) -> str:
    prefix = accession[:-3] + "nnn"
    if accession.startswith("GSE"):
        return f"/geo/series/{prefix}/{accession}/suppl/{filename}"
    elif accession.startswith("GSM"):
        return f"/geo/samples/{prefix}/{accession}/suppl/{filename}"
    else:
        raise ValueError(f"Invalid GEO accession: {accession}")


def download_geo_file(accession: str, filename: str, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    local_path = os.path.join(dest_dir, filename)

    if os.path.exists(local_path):
        print(f"Already exists: {local_path}")
        return local_path

    ftp_path = _build_ftp_path(accession, filename)
    print(f"Downloading ftp://{GEO_FTP_HOST}{ftp_path} -> {local_path}")

    ftp = ftplib.FTP(GEO_FTP_HOST)
    ftp.login()
    try:
        with open(local_path, "wb") as f:
            ftp.retrbinary(f"RETR {ftp_path}", f.write)
    except Exception:
        # Clean up partial download on failure
        if os.path.exists(local_path):
            os.remove(local_path)
        raise
    finally:
        ftp.quit()

    print(f"Downloaded: {local_path}")
    return local_path


if __name__ == "__main__":
    assert len(sys.argv) >= 3, f"Usage: {sys.argv[0]} <accession> <filename> [dest_dir]"
    accession = sys.argv[1]
    filename = sys.argv[2]
    dest_dir = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/geo_agent/{accession}/"
    download_geo_file(accession, filename, dest_dir)
