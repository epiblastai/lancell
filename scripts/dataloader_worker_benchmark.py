"""Benchmark: CellDataset throughput with varying DataLoader worker counts.

Opens a lancell atlas, builds a CellDataset, then times one full epoch
for each worker count in [0, 1, 2, 4] (0 = in-process, no spawn overhead).

Reports total epoch time, ms/batch, and cells/sec for each setting.

Usage::

    uv run scripts/dataloader_worker_benchmark.py
    uv run scripts/dataloader_worker_benchmark.py --atlas s3://bucket/prefix/ --query "tissue = 'lung'"
    uv run scripts/dataloader_worker_benchmark.py --workers 0 1 2 4 --batch-size 512 --repeats 3
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from tqdm.auto import tqdm

# Allow imports from the repo root (lancell + examples)
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))


ATLAS_DEFAULT = "s3://epiblast/ragged_atlases/cellxgene_mini_bp/"
QUERY_DEFAULT = "sex == 'male'"


def open_atlas(atlas_dir: str):
    import obstore.store

    from examples.cellxgene_census.schema import CellObs
    from lancell.atlas import RaggedAtlas

    if atlas_dir.startswith("s3://"):
        parsed = urlparse(atlas_dir)
        bucket = parsed.netloc
        prefix = os.path.join(parsed.path.strip("/"), "zarr_store")
        region = os.environ.get("AWS_REGION", "us-east-2")
        store = obstore.store.S3Store(bucket, prefix=prefix, region=region)
        db_uri = atlas_dir.rstrip("/") + "/lance_db"
    else:
        store = obstore.store.LocalStore(str(Path(atlas_dir) / "zarr_store"))
        db_uri = str(Path(atlas_dir) / "lance_db")

    return RaggedAtlas.open(
        db_uri=db_uri,
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={"gene_expression": "gene_expression_registry"},
    )


def run_epoch(dataset, n_workers: int) -> tuple[float, int, list[float]]:
    """Run one full epoch; return (total_s, total_cells, per_batch_ms)."""
    from lancell.dataloader import make_loader

    loader = make_loader(dataset, num_workers=n_workers)

    batch_times = []
    total_cells = 0
    t_epoch_start = time.perf_counter()

    for batch in tqdm(loader):
        t_batch_end = time.perf_counter()
        batch_times.append((t_batch_end - t_epoch_start) * 1000)
        total_cells += len(batch.offsets) - 1
        t_epoch_start = t_batch_end  # measure next batch from here

    total_s = sum(batch_times) / 1000
    return total_s, total_cells, batch_times


def main():
    parser = argparse.ArgumentParser(description="Benchmark CellDataset worker throughput")
    parser.add_argument("--atlas", default=ATLAS_DEFAULT, help="Atlas directory (S3 or local)")
    parser.add_argument("--query", default=QUERY_DEFAULT, help="LanceDB WHERE clause")
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        default=[0, 2, 4],
        help="Worker counts to benchmark (default: 0 2 4)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Cells per batch (default: 256)"
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Timed epochs per worker count (default: 1)"
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup epochs (default: 1)")
    args = parser.parse_args()

    print(f"Atlas    : {args.atlas}")
    print(f"Query    : {args.query}")
    print(f"Workers  : {args.workers}")
    print(f"Batch sz : {args.batch_size}")
    print(f"Warmup   : {args.warmup} epoch(s) | Repeats: {args.repeats}")
    print()

    print("Opening atlas...", flush=True)
    atlas = open_atlas(args.atlas)

    cells_pl = atlas.query().where(args.query).to_polars()
    print(f"Query returned {cells_pl.height:,} cells")
    print()

    results = []

    for n_workers in args.workers:
        from lancell.dataloader import CellDataset

        label = f"workers={n_workers}" + (" (in-process)" if n_workers == 0 else "")
        print(f"--- {label} ---", flush=True)

        dataset = CellDataset(
            atlas=atlas,
            cells_pl=cells_pl,
            feature_space="gene_expression",
            layer="counts",
            batch_size=args.batch_size,
            shuffle=True,
            seed=args.seed,
            num_workers=max(1, n_workers),  # planning always uses >=1
        )
        n_batches = len(dataset)
        print(
            f"  {n_batches} batches x {args.batch_size} cells = {dataset.n_cells:,} cells, "
            f"{dataset.n_features:,} features"
        )

        # Warmup
        for ep in range(args.warmup):
            dataset.set_epoch(ep)
            _, _, _ = run_epoch(dataset, n_workers)
            print(f"  warmup {ep + 1}/{args.warmup} done", flush=True)

        # Timed repeats
        epoch_times = []
        all_batch_ms = []
        for rep in range(args.repeats):
            dataset.set_epoch(args.warmup + rep)
            t, n_cells, batch_ms = run_epoch(dataset, n_workers)
            epoch_times.append(t)
            all_batch_ms.extend(batch_ms)
            cells_sec = n_cells / t if t > 0 else 0
            print(
                f"  rep {rep + 1}: {t:.3f}s total | "
                f"{t / n_batches * 1000:.1f} ms/batch | "
                f"{cells_sec:,.0f} cells/s",
                flush=True,
            )

        med_epoch = float(np.median(epoch_times))
        med_batch_ms = float(np.median(all_batch_ms))
        p95_batch_ms = float(np.percentile(all_batch_ms, 95))
        med_cells_sec = dataset.n_cells / med_epoch if med_epoch > 0 else 0

        results.append(
            {
                "n_workers": n_workers,
                "label": label,
                "med_epoch_s": med_epoch,
                "med_batch_ms": med_batch_ms,
                "p95_batch_ms": p95_batch_ms,
                "med_cells_sec": med_cells_sec,
            }
        )
        print()

    # Summary table
    col = 22
    print("=" * (col + 60))
    print(
        f"{'workers':<{col}} {'epoch (med)'!s:>12} {'ms/batch (med)'!s:>16} {'ms/batch (p95)'!s:>16} {'cells/s'!s:>12}"
    )
    print("-" * (col + 60))
    for r in results:
        print(
            f"{r['label']:<{col}} "
            f"{r['med_epoch_s']:>11.3f}s "
            f"{r['med_batch_ms']:>15.1f}ms "
            f"{r['p95_batch_ms']:>15.1f}ms "
            f"{r['med_cells_sec']:>11,.0f}"
        )
    print("=" * (col + 60))

    if len(results) > 1:
        baseline = results[0]["med_epoch_s"]
        print("\nSpeedup vs baseline (workers=0):")
        for r in results[1:]:
            speedup = baseline / r["med_epoch_s"] if r["med_epoch_s"] > 0 else 0
            print(f"  {r['label']}: {speedup:.2f}x")


if __name__ == "__main__":
    main()
