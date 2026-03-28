"""Migrate Tahoe-100M from old S3 LanceDB to a lancell RaggedAtlas.

Reads pre-ingested Tahoe-100M data from the old epiblast LanceDB on S3
(per-cell binary blobs of gene_indices + counts) and writes it into
lancell's zarr-based sparse format with the new CellIndex schema.

This is much faster than re-streaming from HuggingFace because the data
is already in S3 in the same region.

Old format (epiblast):
  - gene_expression table: one row per cell with gene_indices (int32 bytes)
    and counts (float32 bytes) as binary blobs
  - gene_index values are positional into a global genes table
  - metadata is denormalized on each row

New format (lancell):
  - Zarr CSR arrays (indices + values) in a single zarr group
  - CellIndex rows in LanceDB with SparseZarrPointer structs
  - Feature UIDs from GenomicFeatureSchema

Usage:
    python -m lancell_examples.multimodal_perturbation_atlas.scripts.migrate_Tahoe100M_from_s3 \
        --atlas-path /path/to/atlas \
        [--max-batches 2] \
        [--batch-size 50000]
"""

import argparse
import json
import warnings
from pathlib import Path
from queue import Queue
from threading import Thread

import lancedb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import scipy.sparse as sp
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Transforming to str index")

from lancell.atlas import create_or_open_atlas
from lancell.ingestion import (
    SparseZarrWriter,
    insert_cell_records,
)
from lancell.schema import make_uid

from lancell_examples.multimodal_perturbation_atlas.schema import (
    CellIndex,
    DatasetSchema,
    GenomicFeatureSchema,
    PublicationSchema,
    PublicationSectionSchema,
    REGISTRY_SCHEMAS,
    SmallMoleculeSchema,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACCESSION_DIR = Path("/home/ubuntu/geo_agent_resolution/Tahoe100M")
FEATURE_SPACE = "gene_expression"
OLD_S3_URI = "s3://epiblast/lancell/"
OLD_DATASET_UID = "cfb577ba-0319-4ddc-be93-4b81cc1cc198"
# Tahoe rows start at this offset in the old gene_expression table
# (total rows = 100,936,851; Tahoe = 95,624,334; start = 5,312,517)
OLD_TABLE_OFFSET = 5_312_517
OLD_TOTAL_CELLS = 95_624_334
CHECKPOINT_FILENAME = "tahoe_migrate_checkpoint.json"


# ---------------------------------------------------------------------------
# Gene index remapping
# ---------------------------------------------------------------------------


def build_gene_remap(
    old_db: lancedb.DBConnection,
    new_feature_df: pd.DataFrame,
) -> np.ndarray:
    """Build a dense array mapping old gene_index → new column position.

    Returns an int32 array where arr[old_gene_index] = new_col, or -1.
    """
    # Get old gene_index → ensembl_id from S3
    old_genes = (
        old_db.open_table("genes")
        .search()
        .where("organism = 'human'")
        .select(["gene_index", "ensembl_id"])
        .to_pandas()
    )
    old_idx_to_ensembl = dict(zip(old_genes["gene_index"], old_genes["ensembl_id"]))

    # Build new ensembl_id → column position
    new_df = new_feature_df.drop_duplicates(subset="ensembl_gene_id", keep="first")
    new_sorted = new_df.sort_values("ensembl_gene_id").reset_index(drop=True)
    new_ensembl_to_col = {
        eid: idx for idx, eid in enumerate(new_sorted["ensembl_gene_id"])
    }

    # Build dense remap array
    max_old_idx = max(old_idx_to_ensembl.keys())
    remap = np.full(max_old_idx + 1, -1, dtype=np.int32)
    mapped = 0
    for old_idx, eid in old_idx_to_ensembl.items():
        if eid and eid in new_ensembl_to_col:
            remap[old_idx] = new_ensembl_to_col[eid]
            mapped += 1

    print(f"  Gene remap: {mapped} mapped, {len(old_idx_to_ensembl) - mapped} unmapped")
    return remap, len(new_sorted)


# ---------------------------------------------------------------------------
# Molecule UID remapping
# ---------------------------------------------------------------------------


def build_molecule_remap() -> dict[str, str]:
    """Build drug_name → new molecule UID mapping."""
    with open(ACCESSION_DIR / "drug_to_molecule_uid.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Batch conversion: old rows → CSR + obs
# ---------------------------------------------------------------------------


def _build_csr_from_blobs(
    gene_indices_col: pd.Series,
    counts_col: pd.Series,
    gene_remap: np.ndarray,
    n_features: int,
) -> sp.csr_matrix:
    """Vectorized conversion of binary blob columns to a single CSR matrix.

    Unpacks all binary blobs, concatenates, remaps gene indices in bulk,
    and builds a single CSR matrix for the whole batch.
    """
    n_cells = len(gene_indices_col)

    # Unpack all blobs into flat arrays + build indptr
    all_old_indices = []
    all_counts = []
    indptr = np.empty(n_cells + 1, dtype=np.int64)
    indptr[0] = 0

    for i in range(n_cells):
        gi = np.frombuffer(gene_indices_col.iloc[i], dtype=np.int32)
        ct = np.frombuffer(counts_col.iloc[i], dtype=np.float32)
        all_old_indices.append(gi)
        all_counts.append(ct)
        indptr[i + 1] = indptr[i] + len(gi)

    if indptr[-1] == 0:
        return sp.csr_matrix((n_cells, n_features), dtype=np.float32)

    flat_old = np.concatenate(all_old_indices)
    flat_counts = np.concatenate(all_counts)

    # Vectorized remap — single numpy fancy-index operation
    valid_range = flat_old < len(gene_remap)
    flat_old[~valid_range] = 0  # will be filtered by -1
    flat_new = gene_remap[flat_old]
    flat_new[~valid_range] = -1
    valid = flat_new >= 0

    # Build CSR with only valid entries
    if not valid.all():
        # Recompute indptr for the filtered entries
        valid_per_cell = np.diff(indptr)  # original counts per cell
        cell_ids = np.repeat(np.arange(n_cells), valid_per_cell)
        valid_cell_ids = cell_ids[valid]
        flat_new = flat_new[valid]
        flat_counts = flat_counts[valid]

        # Rebuild indptr from filtered cell assignments
        new_counts = np.bincount(valid_cell_ids, minlength=n_cells)
        indptr = np.zeros(n_cells + 1, dtype=np.int64)
        np.cumsum(new_counts, out=indptr[1:])

    # Sort indices within each cell (required for CSR)
    # Process cells where indices are unsorted
    for i in range(n_cells):
        start, end = indptr[i], indptr[i + 1]
        if end - start > 1:
            chunk = flat_new[start:end]
            if not np.all(chunk[:-1] <= chunk[1:]):
                order = np.argsort(chunk)
                flat_new[start:end] = chunk[order]
                flat_counts[start:end] = flat_counts[start:end][order]

    return sp.csr_matrix(
        (flat_counts, flat_new.astype(np.int32), indptr),
        shape=(n_cells, n_features),
        dtype=np.float32,
    )


def _build_obs_from_old_rows(
    batch_df: pd.DataFrame,
    drug_to_new_uid: dict[str, str],
    cell_line_map: dict[str, str],
) -> pd.DataFrame:
    """Build new CellIndex-compatible obs from old denormalized rows."""
    n_cells = len(batch_df)
    obs_records = []

    for i in range(n_cells):
        row = batch_df.iloc[i]

        # Parse additional_metadata
        additional_raw = row.get("additional_metadata", "{}")
        try:
            additional = json.loads(additional_raw) if additional_raw else {}
        except (json.JSONDecodeError, TypeError):
            additional = {}

        drug = additional.get("drug", "")
        cellosaurus_id = additional.get("cellosaurus_id", "")
        plate = additional.get("plate", "")
        sample = additional.get("sample", "")
        barcode = additional.get("barcode", "")
        moa_fine = additional.get("moa_fine", "")
        drug_targets = additional.get("drug_targets", "")

        cell_line = cell_line_map.get(cellosaurus_id, row.get("cell_line", ""))
        is_control = bool(row.get("is_control", False))

        # Perturbation fields
        if is_control:
            p_uids = None
            p_types = None
            p_conc = None
            p_dur = None
            p_meta = None
            neg_type = "DMSO"
        else:
            old_chem_conc = row.get("chemical_perturbation_concentration")
            old_chem_meta = row.get("chemical_perturbation_additional_metadata")

            # Get compound name from chemical_perturbation_additional_metadata
            compound_name = drug
            if old_chem_meta and len(old_chem_meta) > 0:
                try:
                    chem_info = json.loads(old_chem_meta[0])
                    compound_name = chem_info.get("compound_name", compound_name)
                except (json.JSONDecodeError, TypeError):
                    pass

            new_uid = drug_to_new_uid.get(compound_name)
            if new_uid:
                p_uids = [new_uid]
                p_types = ["small_molecule"]
                p_conc = list(old_chem_conc) if old_chem_conc else [-1.0]
                p_dur = [-1.0]
                p_meta = [json.dumps({"compound_name": compound_name})]
            else:
                p_uids = None
                p_types = None
                p_conc = None
                p_dur = None
                p_meta = None
            neg_type = None

        new_additional = {
            "barcode": barcode,
            "plate": plate,
            "sample": sample,
            "cellosaurus_id": cellosaurus_id,
            "drug": drug,
        }
        if moa_fine and moa_fine != "unclear":
            new_additional["moa_fine"] = moa_fine
        if drug_targets:
            new_additional["drug_targets"] = drug_targets

        obs_records.append({
            "assay": "single-cell RNA sequencing",
            "organism": "Homo sapiens",
            "cell_line": cell_line,
            "cell_type": None,
            "development_stage": None,
            "disease": None,
            "tissue": None,
            "donor_uid": None,
            "days_in_vitro": None,
            "additional_metadata": json.dumps(new_additional),
            "replicate": None,
            "batch_id": plate if plate else None,
            "well_position": None,
            "is_negative_control": is_control,
            "negative_control_type": neg_type,
            "perturbation_uids": p_uids,
            "perturbation_types": p_types,
            "perturbation_concentrations_um": p_conc,
            "perturbation_durations_hr": p_dur,
            "perturbation_additional_metadata": p_meta,
        })

    obs_df = pd.DataFrame(obs_records)
    obs_df = CellIndex.compute_auto_fields(obs_df)
    return obs_df


def convert_batch(
    batch_df: pd.DataFrame,
    gene_remap: np.ndarray,
    n_features: int,
    drug_to_new_uid: dict[str, str],
    cell_line_map: dict[str, str],
) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Convert a batch of old LanceDB rows to CSR matrix + new obs DataFrame."""
    csr = _build_csr_from_blobs(
        batch_df["gene_indices"], batch_df["counts"],
        gene_remap, n_features,
    )
    obs_df = _build_obs_from_old_rows(batch_df, drug_to_new_uid, cell_line_map)
    return csr, obs_df


# ---------------------------------------------------------------------------
# FK tables and feature registration (same as ingest script)
# ---------------------------------------------------------------------------


def populate_fk_tables(db_uri: str) -> str:
    """Create publication and small_molecules tables. Returns publication_uid."""
    db = lancedb.connect(db_uri)
    existing = db.list_tables().tables

    pub_df = pd.read_parquet(ACCESSION_DIR / "PublicationSchema.parquet")
    publication_uid = pub_df["uid"].iloc[0]
    if "publications" not in existing:
        pub_table = db.create_table(
            "publications", schema=PublicationSchema.to_arrow_schema()
        )
    else:
        pub_table = db.open_table("publications")
    pub_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(pub_df, schema=PublicationSchema.to_arrow_schema())
    )
    print(f"  Publication: {publication_uid}")

    section_parquet = ACCESSION_DIR / "PublicationSectionSchema.parquet"
    if section_parquet.exists():
        section_df = pd.read_parquet(section_parquet)
        if "publication_sections" not in existing:
            sec_table = db.create_table(
                "publication_sections",
                schema=PublicationSectionSchema.to_arrow_schema(),
            )
        else:
            sec_table = db.open_table("publication_sections")
        sec_table.add(
            pa.Table.from_pandas(
                section_df, schema=PublicationSectionSchema.to_arrow_schema()
            )
        )

    sm_df = pd.read_parquet(ACCESSION_DIR / "SmallMoleculeSchema.parquet")
    if "small_molecules" not in existing:
        sm_table = db.create_table(
            "small_molecules", schema=SmallMoleculeSchema.to_arrow_schema()
        )
    else:
        sm_table = db.open_table("small_molecules")
    sm_table.merge_insert(on="uid").when_not_matched_insert_all().execute(
        pa.Table.from_pandas(sm_df, schema=SmallMoleculeSchema.to_arrow_schema())
    )
    print(f"  Small molecules: {len(sm_df)} records")

    return publication_uid


def register_features(atlas):
    """Register genomic features from the finalized parquet."""
    feature_df = pd.read_parquet(ACCESSION_DIR / "GenomicFeatureSchema.parquet")
    records = [
        GenomicFeatureSchema(**row.to_dict()) for _, row in feature_df.iterrows()
    ]
    n_new = atlas.register_features("gene_expression", records)
    print(f"  Registered {n_new} new features ({len(records)} total)")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _checkpoint_path(atlas_path: Path) -> Path:
    return atlas_path / CHECKPOINT_FILENAME


def write_checkpoint(atlas_path: Path, state: dict) -> None:
    cp = _checkpoint_path(atlas_path)
    tmp = cp.with_suffix(".tmp")
    tmp.write_text(json.dumps(state))
    tmp.rename(cp)


def load_checkpoint(atlas_path: Path) -> dict | None:
    cp = _checkpoint_path(atlas_path)
    if cp.exists():
        return json.loads(cp.read_text())
    return None


def clear_checkpoint(atlas_path: Path) -> None:
    _checkpoint_path(atlas_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Tahoe-100M from old S3 LanceDB to lancell atlas"
    )
    parser.add_argument(
        "--atlas-path", type=str, required=True,
        help="Directory for the atlas (created if needed)",
    )
    parser.add_argument(
        "--max-batches", type=int, default=None,
        help="Maximum number of batches to process (for testing)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=50_000,
        help="Number of cells per batch read from S3",
    )
    args = parser.parse_args()

    atlas_path = Path(args.atlas_path)

    # --- Load lookups ---
    print("Loading lookup tables...")
    with open(ACCESSION_DIR / "cellosaurus_to_cell_name_resolved.json") as f:
        cell_line_map = json.load(f)
    drug_to_new_uid = build_molecule_remap()
    print(f"  Cell lines: {len(cell_line_map)}, Drugs: {len(drug_to_new_uid)}")

    # --- Build gene remap from old S3 DB ---
    print("\nBuilding gene index remap from S3...")
    old_db = lancedb.connect(OLD_S3_URI)
    feature_df = pd.read_parquet(ACCESSION_DIR / "GenomicFeatureSchema.parquet")
    gene_remap, n_features = build_gene_remap(old_db, feature_df)

    # Build var_df for feature layout
    new_df = feature_df.drop_duplicates(subset="ensembl_gene_id", keep="first")
    new_sorted = new_df.sort_values("ensembl_gene_id").reset_index(drop=True)
    var_df = pd.DataFrame(
        {"global_feature_uid": new_sorted["uid"].values},
        index=pd.Index(new_sorted["ensembl_gene_id"].values, name="ensembl_gene_id"),
    )

    # --- Create or open atlas ---
    print(f"\n{'='*60}")
    print("Creating/opening atlas")
    print(f"{'='*60}")
    atlas = create_or_open_atlas(
        str(atlas_path),
        cell_table_name="cells",
        cell_schema=CellIndex,
        dataset_table_name="datasets",
        dataset_schema=DatasetSchema,
        registry_schemas=REGISTRY_SCHEMAS,
    )

    # --- Populate FK tables ---
    print(f"\n{'='*60}")
    print("Populating foreign key tables")
    print(f"{'='*60}")
    db_uri = str(atlas_path / "lance_db")
    publication_uid = populate_fk_tables(db_uri)

    # --- Register features ---
    print(f"\n{'='*60}")
    print("Registering features")
    print(f"{'='*60}")
    register_features(atlas)

    # --- Check for checkpoint vs fresh start ---
    checkpoint = load_checkpoint(atlas_path)

    if checkpoint is not None:
        dataset_uid = checkpoint["dataset_uid"]
        zarr_group_name = dataset_uid
        total_cells = checkpoint["total_cells"]
        total_nnz = checkpoint["total_nnz"]
        batch_count = checkpoint["batch_count"]

        print(f"\n  RESUMING from checkpoint:")
        print(f"    Dataset UID: {dataset_uid}")
        print(f"    Batches completed: {batch_count}")
        print(f"    Cells written: {total_cells:,}")
        print(f"    NNZ written: {total_nnz:,}")

        group = atlas._root[zarr_group_name]
        writer = SparseZarrWriter.open(
            group, "counts",
            feature_space=FEATURE_SPACE,
            written=total_nnz,
        )
    else:
        dataset_uid = make_uid()
        zarr_group_name = dataset_uid

        dataset_record = DatasetSchema(
            uid=dataset_uid,
            zarr_group=zarr_group_name,
            feature_space=FEATURE_SPACE,
            n_cells=OLD_TOTAL_CELLS,
            publication_uid=publication_uid,
            accession_database="HuggingFace",
            accession_id="tahoebio/Tahoe-100M",
            dataset_description=(
                "Tahoe-100M: giga-scale single-cell perturbation atlas. "
                "100M+ transcriptomic profiles from 50 cancer cell lines "
                "exposed to ~379 small-molecule perturbations at 3 concentrations."
            ),
            organism=["Homo sapiens"],
            tissue=None,
            cell_line=sorted(set(cell_line_map.values())),
            disease=None,
        )

        dataset_arrow = pa.Table.from_pylist(
            [dataset_record.model_dump()],
            schema=type(dataset_record).to_arrow_schema(),
        )
        atlas._dataset_table.add(dataset_arrow)
        print(f"\n  Dataset UID: {dataset_uid}")

        group = atlas._root.create_group(zarr_group_name)
        writer = SparseZarrWriter.create(
            group, "counts",
            data_dtype=np.float32,
            feature_space=FEATURE_SPACE,
        )

        var_pl = pl.DataFrame({
            "ensembl_gene_id": var_df.index.tolist(),
            "global_feature_uid": var_df["global_feature_uid"].tolist(),
        })
        atlas.add_or_reuse_layout(var_pl, dataset_uid, FEATURE_SPACE)

        total_cells = 0
        total_nnz = 0
        batch_count = 0

    # --- Stream from S3 with prefetching ---
    print(f"\n{'='*60}")
    print(f"Migrating from {OLD_S3_URI}")
    print(f"{'='*60}")

    old_ge = old_db.open_table("gene_expression")
    s3_offset = OLD_TABLE_OFFSET + total_cells

    remaining = OLD_TOTAL_CELLS - total_cells
    if args.max_batches:
        remaining = min(remaining, args.max_batches * args.batch_size)

    # Prefetch thread: reads S3 batches ahead of the main loop.
    # Uses a thread (not process) because lance table handles aren't
    # fork/pickle-safe, and S3 reads release the GIL anyway.
    prefetch_queue: Queue = Queue(maxsize=2)

    def _prefetch_s3(table, start_offset, batch_size, max_batches, total_limit):
        """Read batches from S3 lance table into a queue."""
        offset = start_offset
        count = 0
        cells_read = 0
        while cells_read < total_limit:
            if max_batches is not None and count >= max_batches:
                break
            df = table.search().offset(offset).limit(batch_size).to_pandas()
            if df.empty:
                break
            # Verify we're still reading Tahoe data
            if df["dataset_uid"].iloc[0] != OLD_DATASET_UID:
                break
            prefetch_queue.put(df)
            offset += len(df)
            cells_read += len(df)
            count += 1
        prefetch_queue.put(None)  # sentinel

    prefetch_thread = Thread(
        target=_prefetch_s3,
        args=(old_ge, s3_offset, args.batch_size, args.max_batches, remaining),
        daemon=True,
    )
    prefetch_thread.start()

    pbar = tqdm(total=remaining, unit="cells", desc="Migrating")

    batches_this_run = 0
    while True:
        batch_df = prefetch_queue.get()
        if batch_df is None:
            break

        # Convert old format → new CSR + obs
        csr, obs_df = convert_batch(
            batch_df, gene_remap, n_features, drug_to_new_uid, cell_line_map
        )

        # Append sparse data to zarr
        starts, ends = writer.append_csr(csr)

        # Insert cell records
        n_inserted = insert_cell_records(
            atlas, obs_df,
            feature_space=FEATURE_SPACE,
            zarr_group=zarr_group_name,
            dataset_uid=dataset_uid,
            starts=starts,
            ends=ends,
            zarr_row_offset=total_cells,
        )

        total_cells += n_inserted
        total_nnz += csr.nnz
        batch_count += 1
        batches_this_run += 1

        # Checkpoint
        write_checkpoint(atlas_path, {
            "dataset_uid": dataset_uid,
            "batch_count": batch_count,
            "total_cells": total_cells,
            "total_nnz": total_nnz,
        })

        pbar.update(n_inserted)
        pbar.set_postfix(
            batch=batch_count,
            nnz=f"{total_nnz/1e9:.2f}B",
        )

        del batch_df, csr, obs_df

    prefetch_thread.join()
    pbar.close()

    # Trim zarr arrays
    writer.trim()

    # Clear checkpoint on completion
    if total_cells >= OLD_TOTAL_CELLS:
        clear_checkpoint(atlas_path)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("Migration complete" if total_cells >= OLD_TOTAL_CELLS else "Migration paused")
    print(f"{'='*60}")
    print(f"  Source: {OLD_S3_URI}")
    print(f"  Dataset UID: {dataset_uid}")
    print(f"  Batches processed: {batch_count}")
    print(f"  Total cells migrated: {total_cells:,} / {OLD_TOTAL_CELLS:,}")
    print(f"  Total nonzeros: {total_nnz:,}")
    print(f"  Atlas path: {atlas_path}")


if __name__ == "__main__":
    main()
