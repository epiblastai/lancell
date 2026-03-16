"""Ingest a single cellxgene census h5ad file into a lancell ragged atlas.

Usage:
    python -m examples.cellxgene_census.ingest \
        --h5ad /path/to/dataset.h5ad \
        --atlas-dir /path/to/atlas

Designed to be run in parallel for multiple h5ad files against the same atlas.
"""

import argparse
import os
from pathlib import Path

import anndata as ad
import obstore.store

from examples.cellxgene_census.schema import (
    CellObs,
    CensusDatasetRecord,
    GeneFeatureSpace,
)
from lancell.atlas import RaggedAtlas
from lancell.ingestion import add_anndata_batch
from lancell.schema import make_uid
from lancell.tools.add_csc import add_csc

FEATURE_SPACE = "gene_expression"
LAYER_NAME = "counts"
CHUNK_SIZE = 5_000
SHARD_SIZE = 50_000_000


def make_store(atlas_dir: str) -> obstore.store.ObjectStore:
    """Build an obstore for the zarr data, choosing backend from the path."""
    if atlas_dir.startswith("s3://"):
        # e.g. "s3://bucket/prefix/atlas"
        from urllib.parse import urlparse

        parsed = urlparse(atlas_dir)
        bucket = parsed.netloc
        prefix = os.path.join(parsed.path.strip("/"), "zarr_store")
        region = os.environ.get("AWS_REGION")
        if not region:
            raise ValueError("AWS_REGION environment variable must be set for S3 access")
        return obstore.store.S3Store(bucket, prefix=prefix, region=region)
    else:
        zarr_path = Path(atlas_dir) / "zarr_store"
        zarr_path.mkdir(parents=True, exist_ok=True)
        return obstore.store.LocalStore(str(zarr_path))


def db_uri_for(atlas_dir: str) -> str:
    """Return the LanceDB URI for the given atlas directory."""
    if atlas_dir.startswith("s3://"):
        return atlas_dir.rstrip("/") + "/lance_db"
    return str(Path(atlas_dir) / "lance_db")


def atlas_exists(atlas_dir: str) -> bool:
    """Check whether an atlas already exists at the given path."""
    if atlas_dir.startswith("s3://"):
        import lancedb

        db = lancedb.connect(db_uri_for(atlas_dir))
        return "cells" in db.list_tables().tables
    return (Path(atlas_dir) / "lance_db").exists()


def create_atlas(atlas_dir: str) -> RaggedAtlas:
    """Create a new atlas at the given directory."""
    if not atlas_dir.startswith("s3://"):
        Path(atlas_dir).mkdir(parents=True, exist_ok=True)
    store = make_store(atlas_dir)
    return RaggedAtlas.create(
        db_uri=db_uri_for(atlas_dir),
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        dataset_schema=CensusDatasetRecord,
        store=store,
        registry_schemas={FEATURE_SPACE: GeneFeatureSpace},
    )


def open_atlas(atlas_dir: str) -> RaggedAtlas:
    """Open an existing atlas."""
    store = make_store(atlas_dir)
    return RaggedAtlas.open(
        db_uri=db_uri_for(atlas_dir),
        cell_table_name="cells",
        cell_schema=CellObs,
        dataset_table_name="datasets",
        store=store,
        registry_tables={FEATURE_SPACE: "gene_expression_registry"},
    )


def register_genes(atlas: RaggedAtlas, adata: ad.AnnData) -> dict[str, str]:
    """Register gene features and return ensembl_id -> registry uid mapping."""
    var = adata.var
    ensembl_ids = list(var.index)

    # Load existing registry to find genes already registered
    registry_table = atlas._registry_tables[FEATURE_SPACE]
    existing_df = registry_table.search().select(["uid", "ensembl_id"]).to_polars()
    existing_ensembl_to_uid = dict(
        zip(existing_df["ensembl_id"].to_list(), existing_df["uid"].to_list(), strict=False)
    )

    # Only create new features for genes not already in the registry
    ensembl_to_uid: dict[str, str] = {}
    new_features: list[GeneFeatureSpace] = []
    for ensembl_id in ensembl_ids:
        if ensembl_id in existing_ensembl_to_uid:
            ensembl_to_uid[ensembl_id] = existing_ensembl_to_uid[ensembl_id]
        elif ensembl_id in ensembl_to_uid:
            # Already seen in this file (duplicate var index), skip
            pass
        else:
            row = var.loc[ensembl_id]
            feature = GeneFeatureSpace(
                ensembl_id=ensembl_id,
                feature_name=str(row["feature_name"]),
                feature_reference=str(row["feature_reference"]),
                feature_biotype=str(row["feature_biotype"]),
                feature_length=int(row["feature_length"]),
                feature_type=str(row["feature_type"]),
                feature_is_filtered=bool(row["feature_is_filtered"]),
            )
            new_features.append(feature)
            ensembl_to_uid[ensembl_id] = feature.uid

    if new_features:
        n_new = atlas.register_features(FEATURE_SPACE, new_features)
        print(
            f"  Registered {n_new} new genes ({len(ensembl_ids)} total in this file, "
            f"{len(existing_ensembl_to_uid)} already existed)"
        )
    else:
        print(f"  All {len(ensembl_ids)} genes already registered")

    return ensembl_to_uid


def ingest_backed(
    atlas: RaggedAtlas,
    adata: ad.AnnData,
    h5ad_path: str,
    ensembl_to_uid: dict[str, str],
) -> tuple[int, str]:
    """Ingest a backed h5ad into the atlas using batched streaming writes."""
    n_cells = adata.n_obs
    print(f"  n_cells={n_cells:,}")

    cellxgene_dataset_id = Path(h5ad_path).stem
    zarr_group = make_uid()

    # Attach global_feature_uid to adata.var so write_var_sidecar can use it
    ensembl_ids = list(adata.var.index)
    adata.var["global_feature_uid"] = [ensembl_to_uid[eid] for eid in ensembl_ids]

    dataset_record = CensusDatasetRecord(
        uid=zarr_group,
        zarr_group=zarr_group,
        feature_space=FEATURE_SPACE,
        n_cells=n_cells,
        cellxgene_dataset_id=cellxgene_dataset_id,
    )

    add_anndata_batch(
        atlas,
        adata,
        feature_space=FEATURE_SPACE,
        zarr_layer=LAYER_NAME,
        dataset_record=dataset_record,
        chunk_shape=(CHUNK_SIZE,),
        shard_shape=(SHARD_SIZE,),
    )

    print(f"    Inserted {n_cells:,} cell records")
    return n_cells, zarr_group


def main():
    parser = argparse.ArgumentParser(
        description="Ingest a cellxgene census h5ad into a lancell atlas"
    )
    parser.add_argument("--h5ad", required=True, help="Path to .h5ad file")
    parser.add_argument("--atlas-dir", required=True, help="Path to atlas directory")
    parser.add_argument("--no-csc", action="store_true", help="Skip adding CSC layout")
    args = parser.parse_args()

    h5ad_path = args.h5ad
    atlas_dir = args.atlas_dir

    print(f"Loading {h5ad_path} (backed mode)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    if atlas_exists(atlas_dir):
        print(f"Opening existing atlas at {atlas_dir}")
        atlas = open_atlas(atlas_dir)
    else:
        print(f"Creating new atlas at {atlas_dir}")
        atlas = create_atlas(atlas_dir)

    print("Registering genes...")
    ensembl_to_uid = register_genes(atlas, adata)

    print("Ingesting data...")
    n_cells, zarr_group = ingest_backed(atlas, adata, h5ad_path, ensembl_to_uid)

    if not args.no_csc:
        print("Adding CSC layout...")
        add_csc(
            atlas,
            zarr_group=zarr_group,
            feature_space=FEATURE_SPACE,
            layer_name=LAYER_NAME,
            chunk_size=CHUNK_SIZE,
            shard_size=SHARD_SIZE,
        )

    print(f"Done! Ingested {n_cells:,} cells from {Path(h5ad_path).name}")


if __name__ == "__main__":
    main()
