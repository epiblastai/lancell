# Lancell Data Model

## Overview

Lancell uses a hybrid storage architecture:
- **LanceDB** for queryable metadata tables (cells, datasets, feature registries, versions)
- **Zarr** for bulk numerical arrays (gene expression, protein abundance, etc.)
- **Parquet** sidecars alongside zarr groups for per-dataset feature metadata and index remaps

---

## Entity Relationship Diagram

```mermaid
erDiagram
    CELL_TABLE {
        str uid PK "random 16-char hex"
        str dataset_uid FK "-> DatasetRecord.uid"
        SparseZarrPointer gene_expression "nullable pointer to zarr"
        DenseZarrPointer protein_abundance "nullable pointer to zarr"
        DenseZarrPointer image_features "nullable pointer to zarr"
        str cell_type "user-defined obs metadata"
        str tissue "user-defined obs metadata"
        str disease "..."
    }

    SPARSE_ZARR_POINTER {
        str feature_space "e.g. gene_expression"
        str zarr_group "path to zarr group"
        int start "CSR indices start offset"
        int end "CSR indices end offset"
        int zarr_row "cell position in group (for CSC)"
    }

    DENSE_ZARR_POINTER {
        str feature_space "e.g. protein_abundance"
        str zarr_group "path to zarr group"
        int position "row index in 2D array"
    }

    DATASET_TABLE {
        str uid PK "random 16-char hex"
        str zarr_group "zarr group path"
        str feature_space "e.g. gene_expression"
        int n_cells "cell count"
        str created_at "ISO timestamp"
    }

    FEATURE_REGISTRY {
        str uid PK "stable canonical ID"
        int global_index "contiguous 0..N-1 compute index"
        str ensembl_id "modality-specific fields..."
        str feature_name "modality-specific fields..."
    }

    FEATURE_DATASET_INDEX {
        str feature_uid FK "-> FeatureRegistry.uid"
        str dataset_uid FK "-> DatasetRecord.uid"
    }

    ATLAS_VERSIONS {
        int version PK "snapshot number"
        str cell_table_name "LanceDB table name"
        int cell_table_version "LanceDB table version"
        str dataset_table_name "LanceDB table name"
        int dataset_table_version "LanceDB table version"
        str registry_table_names "JSON map"
        str registry_table_versions "JSON map"
        int total_cells "total cell count"
        str created_at "ISO timestamp"
    }

    ZARR_GROUP_SPARSE {
        array csr_indices "uint32 (nnz,) flattened col indices"
        array csr_layers_counts "float/int (nnz,) values"
        array csr_layers_log_normalized "float (nnz,) optional"
        array csr_layers_tpm "float (nnz,) optional"
    }

    ZARR_GROUP_DENSE {
        array layers_counts "float32 (n_cells, n_features)"
        array layers_clr "float32 optional"
        array layers_dsb "float32 optional"
    }

    VAR_PARQUET {
        str global_feature_uid FK "-> FeatureRegistry.uid"
        int csc_start "optional CSC col offset"
        int csc_end "optional CSC col offset"
        ___ ___ "plus modality-specific columns"
    }

    REMAP_PARQUET {
        int global_index "remap local_i -> global_index"
    }

    CELL_TABLE ||--o{ SPARSE_ZARR_POINTER : "embeds (gene_expression)"
    CELL_TABLE ||--o{ DENSE_ZARR_POINTER : "embeds (protein, image)"
    CELL_TABLE }o--|| DATASET_TABLE : "dataset_uid"
    SPARSE_ZARR_POINTER ||--|| ZARR_GROUP_SPARSE : "zarr_group points to"
    DENSE_ZARR_POINTER ||--|| ZARR_GROUP_DENSE : "zarr_group points to"
    DATASET_TABLE ||--|| ZARR_GROUP_SPARSE : "zarr_group"
    DATASET_TABLE ||--|| ZARR_GROUP_DENSE : "zarr_group"
    ZARR_GROUP_SPARSE ||--|| VAR_PARQUET : "sidecar at {group}/var.parquet"
    ZARR_GROUP_SPARSE ||--|| REMAP_PARQUET : "sidecar at {group}/local_to_global_index.parquet"
    ZARR_GROUP_DENSE ||--|| VAR_PARQUET : "sidecar at {group}/var.parquet"
    ZARR_GROUP_DENSE ||--|| REMAP_PARQUET : "sidecar at {group}/local_to_global_index.parquet"
    VAR_PARQUET }o--|| FEATURE_REGISTRY : "global_feature_uid -> uid"
    REMAP_PARQUET }o--|| FEATURE_REGISTRY : "global_index"
    FEATURE_DATASET_INDEX }o--|| FEATURE_REGISTRY : "feature_uid"
    FEATURE_DATASET_INDEX }o--|| DATASET_TABLE : "dataset_uid"
    ATLAS_VERSIONS ||..|| CELL_TABLE : "snapshots table version"
    ATLAS_VERSIONS ||..|| DATASET_TABLE : "snapshots table version"
    ATLAS_VERSIONS ||..|| FEATURE_REGISTRY : "snapshots table version"
```

## Storage Backends

```mermaid
graph TB
    subgraph LanceDB["LanceDB (lancedb/)"]
        cells["cells table<br/><i>Cell obs + embedded zarr pointers</i>"]
        datasets["datasets table<br/><i>One row per ingested dataset</i>"]
        gene_reg["gene_expression_registry<br/><i>One row per unique gene</i>"]
        prot_reg["protein_abundance_registry<br/><i>One row per unique protein</i>"]
        img_reg["image_features_registry<br/><i>One row per unique image feature</i>"]
        fdpairs["_feature_dataset_pairs<br/><i>FTS-indexed inverted index</i>"]
        versions["atlas_versions<br/><i>Snapshot records</i>"]
    end

    subgraph ZarrStore["Object Store (S3 / GCS / local)"]
        subgraph ZG1["zarr_group_1/ (sparse)"]
            indices1["csr/indices (uint32)"]
            layers1["csr/layers/counts"]
            var1["var.parquet"]
            remap1["local_to_global_index.parquet"]
        end
        subgraph ZG2["zarr_group_2/ (dense)"]
            data2["layers/counts (float32 2D)"]
            var2["var.parquet"]
            remap2["local_to_global_index.parquet"]
        end
    end

    cells -->|"SparseZarrPointer<br/>start, end, zarr_row"| ZG1
    cells -->|"DenseZarrPointer<br/>position"| ZG2
    cells -->|"dataset_uid"| datasets
    datasets -->|"zarr_group"| ZG1
    datasets -->|"zarr_group"| ZG2
    var1 -->|"global_feature_uid"| gene_reg
    var2 -->|"global_feature_uid"| prot_reg
    remap1 -->|"global_index"| gene_reg
    remap2 -->|"global_index"| prot_reg
    fdpairs -->|"feature_uid"| gene_reg
    fdpairs -->|"dataset_uid"| datasets
```

## Feature Spaces

| Feature Space | Pointer Type | Zarr Layout | Required Layers | Optional Layers |
|---|---|---|---|---|
| `gene_expression` | `SparseZarrPointer` | `csr/indices` (1D uint32) + `csr/layers/*` (1D, same shape) | `counts` | `log_normalized`, `tpm` |
| `protein_abundance` | `DenseZarrPointer` | `layers/*` (2D float32, uniform shape) | `counts` | `clr`, `dsb`, `log_normalized` |
| `image_features` | `DenseZarrPointer` | `layers/*` (2D float32, uniform shape) | `raw` | `log_normalized`, `ctrl_standardized` |

## Key Design Properties

- **Indptr is not stored** -- CSR row boundaries are reconstructed from each cell's `(start, end)` in the pointer
- **Feature indices are local per zarr group** -- the `local_to_global_index.parquet` remap translates local feature index `i` to the global contiguous index in the feature registry
- **Remap freshness** -- the zarr group stores `remap_registry_version` in attrs; stale remaps are rebuilt on read
- **One feature registry per feature space** -- `global_index` values are reassigned by `reindex_registry()` (deterministic sort by `uid`)
- **Versioning** -- `AtlasVersionRecord` snapshots LanceDB table versions for time-travel/checkout
