"""Bulk download scBaseCount h5ad files from GCS.

Usage:
    python -m examples.scbasecount.download \
        --output-dir ./data/scbasecount \
        --max-download 50 \
        --feature-type Velocyto \
        --release-date 2026-01-12 \
        --dry-run
"""

import argparse
import shutil
import subprocess
from pathlib import Path

import gcsfs
import pyarrow.parquet as pq

GCS_BASE = "gs://arc-institute-virtual-cell-atlas/scbasecount"


def read_sample_metadata(
    release_date: str,
    feature_type: str,
) -> list[dict]:
    """Read sample metadata parquet from GCS and return list of row dicts."""
    gcs_path = (
        f"arc-institute-virtual-cell-atlas/scbasecount/{release_date}"
        f"/metadata/{feature_type}/Homo_sapiens/sample_metadata.parquet"
    )
    fs = gcsfs.GCSFileSystem(token="anon")
    with fs.open(gcs_path, "rb") as f:
        table = pq.read_table(f)
    return table.to_pylist()


def build_file_list(
    rows: list[dict],
    release_date: str,
    feature_type: str,
    output_dir: Path,
    max_download: int | None,
) -> list[tuple[str, Path]]:
    """Return list of (gcs_uri, local_path) pairs, skipping already-downloaded files."""
    pairs = []
    for row in rows:
        srx = row.get("srx_accession") or row["SRX_accession"]
        gcs_uri = (
            f"{GCS_BASE}/{release_date}/h5ad/{feature_type}"
            f"/Homo_sapiens/{srx}.h5ad"
        )
        local_path = output_dir / f"{srx}.h5ad"
        if local_path.exists():
            continue
        pairs.append((gcs_uri, local_path))
        if max_download is not None and len(pairs) >= max_download:
            break
    return pairs


def download_files(pairs: list[tuple[str, Path]]) -> None:
    """Download files using gcloud storage cp."""
    for i, (gcs_uri, local_path) in enumerate(pairs, 1):
        print(f"  [{i}/{len(pairs)}] {gcs_uri} -> {local_path}")
        subprocess.run(
            ["gcloud", "storage", "cp", gcs_uri, str(local_path)],
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download scBaseCount h5ad files from GCS"
    )
    parser.add_argument("--output-dir", required=True, help="Local download directory")
    parser.add_argument("--feature-type", default="Velocyto", help="Feature type (default: Velocyto)")
    parser.add_argument("--release-date", default="2026-01-12", help="Release date (default: 2026-01-12)")
    parser.add_argument("--max-download", type=int, default=None, help="Limit number of h5ad files")
    parser.add_argument("--dry-run", action="store_true", help="List files without downloading")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.dry_run and shutil.which("gcloud") is None:
        raise RuntimeError("gcloud CLI not found. Install it from https://cloud.google.com/sdk/docs/install")

    print(f"Reading sample metadata for {args.feature_type} / {args.release_date}...")
    rows = read_sample_metadata(args.release_date, args.feature_type)
    print(f"  Found {len(rows)} samples")

    # Save sample metadata locally for the ingest script
    metadata_path = output_dir / "sample_metadata.parquet"
    if not metadata_path.exists():
        import pyarrow as pa

        pa.parquet.write_table(pa.Table.from_pylist(rows), str(metadata_path))
        print(f"  Saved sample metadata to {metadata_path}")

    pairs = build_file_list(rows, args.release_date, args.feature_type, output_dir, args.max_download)
    print(f"  {len(pairs)} files to download")

    if args.dry_run:
        for gcs_uri, local_path in pairs:
            print(f"    {gcs_uri}")
        return

    download_files(pairs)
    print("Done!")


if __name__ == "__main__":
    main()
