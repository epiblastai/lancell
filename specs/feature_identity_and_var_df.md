# Feature Identity And `var_df` Sidecars

## Summary

Use a hybrid design:

- Keep expensive assay arrays in Zarr, with dataset-local feature ordering.
- Store per-dataset feature metadata in parquet sidecars analogous to AnnData `var`.
- Maintain a global feature registry with:
  - a stable `uid` for canonical identity
  - a contiguous `global_index` integer for compute
- Optionally materialize a compiled per-dataset `local_to_global_index` artifact for fast ML loading.

This separates:

- storage layout of a dataset
- global feature identity and metadata
- compute-oriented remapping for training and retrieval

## Goals

- Avoid rewriting expensive Zarr arrays when feature metadata changes.
- Support dataset-local feature orderings naturally.
- Allow joins into a global feature space for cross-dataset analysis.
- Keep training-time remapping cheap and vectorized.
- Support both sparse and dense feature spaces without forcing one representation.

## Non-Goals

- Do not require all feature spaces to use the same registry semantics.
- Do not force raw fragment/event data into an AnnData-like `var` model.
- Do not make runtime training depend on database joins or string-key lookups.

## Global Feature Registry

`FeatureBaseSchema` should be extended to include both:

- `uid: str`
  - canonical stable identifier
  - should not change if rows are reordered or if the registry is rebuilt
- `global_index: int`
  - dense integer ID used in compute paths
  - intended for remapping local feature indices into a global aligned space

The meaning of `global_index` is:

- unique within one global feature registry
- contiguous or near-contiguous
- fast to gather/scatter against in NumPy, PyTorch, Arrow, or Rust

The meaning of `uid` is:

- stable identity for references, joins, and migrations
- safe to preserve even if `global_index` assignment changes in a future rebuild

## Per-Dataset `var_df` Sidecar

Each dataset may store a parquet sidecar representing its local feature axis.

This sidecar is the equivalent of AnnData `var`:

- one row per local feature
- row order matches the dataset-local feature axis used by the Zarr group

Required properties:

- row order is authoritative
- row `i` corresponds to local feature index `i`
- there must be a way to resolve the row into the global feature registry

Required columns:

- `global_feature_uid` or equivalent stable key into the global registry

Recommended columns:

- `global_index`
- dataset-local display fields such as `feature_name`
- any original source annotations needed for debugging or provenance

Optional columns:

- modality-specific metadata such as gene symbols, transcript IDs, peak coordinates, antibody names, imaging channel metadata, etc.

Why store both `global_feature_uid` and `global_index`:

- `global_feature_uid` is the durable reference
- `global_index` avoids runtime resolution work

If only one is stored, prefer:

- parquet sidecar stores `global_feature_uid`
- build step resolves it into a compiled `local_to_global_index`

If both are stored, ingestion must validate that they agree.

## Which Feature Spaces Use `var_df`

Feature spaces with a stable feature axis should use a `var_df` sidecar.

Examples:

- gene expression
- protein abundance
- image features
- chromatin peaks

For these spaces:

- local column/feature index is meaningful
- sidecar row order is sufficient to define the mapping
- no separate sparse mapping structure is needed beyond the ordered rows

Dense feature spaces do not need explicit per-observation indices merely because they participate in the global registry. The feature axis ordering plus `var_df` is enough.

## Raw Fragments And Other Event Data

Raw fragment data is not a `var_df`-shaped feature space.

Fragments are event records, not features on a stable feature axis. They should be represented directly with coordinate arrays such as:

- `reference_sequence_uid` or equivalent contig/chromosome identifier
- `start`
- `end`
- optional `count`

For raw fragments, the genomic interval itself is the natural key. These do not need to be forced through `FeatureBaseSchema` unless a separate registry is useful for downstream indexing or search.

Called peaks are different:

- peaks are dataset features
- they do have a stable feature axis
- they should use a `var_df` sidecar

## Compiled Remap Artifact

`var_df` is the metadata layer, not the hot-path training layer.

For fast loading, each dataset may also materialize a compact remap artifact:

- `local_to_global_index[i] = global_index of local feature i`

Properties:

- one-dimensional integer array
- length equals the number of local features
- can be stored as NumPy memmap, Arrow, parquet, or Zarr array
- should be cheap to gather from in vectorized code

This artifact is optional but strongly recommended for ML training and repeated retrieval workloads.

## Training-Time Access Pattern

Do not do per-feature or per-cell database lookups during training.

Recommended path:

1. Load a dataset-local sparse or dense batch from Zarr.
2. Load or memory-map that dataset's `local_to_global_index`.
3. Remap local feature indices by vectorized gather.
4. Group batches by dataset when possible to avoid Python-level dispatch overhead.

This makes the hot path purely numeric.

Avoid:

- table joins keyed by `(zarr_group, local_feature_index)` in the training loop
- repeated string UID resolution in the loader
- Python loops over cells solely for remapping

## Storage Contract

For a dataset-backed feature space with a stable feature axis, the dataset contract is:

- Zarr group contains the actual assay arrays
- sidecar parquet contains local feature metadata in row order
- sidecar rows resolve to the global registry
- optional compiled remap stores `local_index -> global_index`

The sidecar and remap may live:

- adjacent to the Zarr group in object storage
- in a parallel dataset metadata directory
- in any location referenced by the manifest/schema layer

The exact storage path is an implementation detail. The contract is about row order and identity resolution, not a specific file layout.

## Ingestion Validation

Ingestion should validate:

- sidecar row count matches the local feature axis length
- sidecar row order is stable and intended
- each required global reference resolves
- `global_feature_uid` and `global_index` agree if both are present
- compiled remap length matches sidecar length
- compiled remap values match the resolved global registry

## Registry Evolution

The main reason to preserve both `uid` and `global_index` is to allow safe evolution.

Allowed changes:

- update metadata columns in the global registry
- add new feature rows
- rebuild `global_index` assignments if needed
- regenerate compiled remap artifacts without touching assay arrays

Assay arrays should not need rewriting when only registry metadata changes.

If `global_index` changes, the required rebuild scope is:

- compiled remap artifacts
- any cached aligned training representations that depend on old indices

The original Zarr arrays and `var_df` sidecars can remain unchanged if they still carry stable `uid` references.

## Optional Deduplication

If many datasets share the exact same ordered feature axis, the system may deduplicate the sidecar/remap pair into a shared feature-set object identified by a content hash.

This is an optimization, not a requirement.

Datasets can then point at:

- a shared feature-set definition
- their own assay arrays

This should only be added if repeated feature sets become a meaningful storage or maintenance cost.

## Recommendation

Adopt the following default:

- extend `FeatureBaseSchema` with `global_index: int`
- keep `uid` as the canonical stable identity
- store per-dataset parquet `var_df` sidecars for all feature spaces with a stable feature axis
- store raw fragments directly as coordinate/event arrays rather than forcing them into `var_df`
- precompute `local_to_global_index` for training and repeated retrieval

This is the best balance of:

- storage flexibility
- metadata maintainability
- runtime performance
- compatibility with AnnData-style dataset-local feature semantics
