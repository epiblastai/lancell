# Reconstructors

A reconstructor converts raw zarr data into an `AnnData` object. Every `ZarrGroupSpec` carries a `reconstructor` instance; at query time, `AtlasQuery` calls `spec.reconstructor.as_anndata(...)` for each feature space being loaded.

The reconstructor is responsible for:

- Loading the per-group local-to-global feature index remap from [`_feature_layouts`](feature_layouts.md)
- Reading zarr arrays for each dataset group
- Remapping per-group local feature indices to the global feature space
- Assembling the final `AnnData` with obs, var, and X (and any additional layers)

Lancell ships three built-in reconstructors. Most users will pick one of them when defining a custom `ZarrGroupSpec`; implementing a custom reconstructor from scratch is rarely needed.

---

## The `Reconstructor` protocol

```python
from lancell.protocols import Reconstructor
```

`Reconstructor` is a `runtime_checkable` protocol. Any class that implements `as_anndata` with the following signature satisfies it — no explicit inheritance needed.

```python
class Reconstructor(Protocol):
    def as_anndata(
        self,
        atlas: RaggedAtlas,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData: ...
```

Key parameters:

- `cells_pl` — Polars DataFrame of the queried cells. Includes the zarr pointer struct columns used to locate each cell's row within its zarr group.
- `pf` — `PointerFieldInfo` identifying which pointer field this reconstructor handles (field name, feature space, and pointer kind).
- `spec` — the `ZarrGroupSpec` for this feature space, carrying the declared array layout and layer names.
- `layer_overrides` — if set, read these layers instead of `spec.layers.required`.
- `feature_join` — `"union"` includes all features from any group; `"intersection"` includes only features present in every group. Ignored when `wanted_globals` is set.
- `wanted_globals` — if set, pin the output feature space to these global feature indices. Overrides `feature_join`.

---

## Built-in reconstructors

### `SparseCSRReconstructor`

**Use for:** high-dimensional sparse data — gene expression, chromatin accessibility, any assay stored in CSR format.

```python
from lancell.reconstruction import SparseCSRReconstructor

ZarrGroupSpec(
    feature_space="gene_expression",
    pointer_kind=PointerKind.SPARSE,
    reconstructor=SparseCSRReconstructor(),
    ...
)
```

Each sparse cell pointer encodes a byte range `[_start, _end)` into the `csr/indices` array and the corresponding layer arrays. The reconstructor reads the flat index and value segments for each cell, remaps the local feature indices to their global positions using the `_feature_layouts` remap, builds a `scipy.sparse.csr_matrix` per layer, and stacks them vertically across groups.

For union queries, cells from groups that don't measure a given feature simply have no entries in that column; the sparse format represents this without any fill. For intersection queries, the reconstructor masks out any feature index that does not appear in every group's remap before building the CSR, effectively discarding non-shared features.

When `wanted_globals` is provided and the number of queried cells exceeds the number of requested features, the reconstructor automatically delegates to `FeatureCSCReconstructor`. This heuristic — cells outnumber features — identifies the regime where CSC reads are cheaper: reading one column at a time across many cells costs less I/O than reading many full cell rows and then slicing. The delegation is transparent; you do not need to configure anything.

### `DenseReconstructor`

**Use for:** dense assays — protein abundance (CITE-seq ADT), image feature vectors, log-normalized expression where all values are non-zero (e.g., after HVG selection and normalization), any data stored as a 2D float array.

```python
from lancell.reconstruction import DenseReconstructor

ZarrGroupSpec(
    feature_space="protein_abundance",
    pointer_kind=PointerKind.DENSE,
    reconstructor=DenseReconstructor(),
    ...
)
```

Each dense cell pointer carries a single row index `_pos`. The reconstructor pre-allocates a float32 output array of shape `(n_cells, n_features)`, then fills it group by group: for each group it reads the relevant rows from `layers/{layer_name}`, remaps the local feature columns to their global positions, and scatters the values into the pre-allocated array.

For union queries, positions corresponding to features not measured in a group remain at their initialized value of zero. For intersection queries, only the columns present in every group's remap are written; all other columns remain zero. For `wanted_globals` queries, the same scatter logic applies but the output column set is pinned to the requested global indices.

Dense data does not support a CSC equivalent. Feature-filtered queries on dense data always read the relevant rows in full and then select columns during scatter.

### `FeatureCSCReconstructor`

**Use for:** sparse data where feature-filtered queries are performance-critical — i.e., when users often request a small set of genes or peaks (such as a marker gene panel) across a large number of cells.

```python
from lancell.reconstruction import FeatureCSCReconstructor

ZarrGroupSpec(
    feature_space="gene_expression",
    pointer_kind=PointerKind.SPARSE,
    reconstructor=FeatureCSCReconstructor(),
    ...
)
```

When `wanted_globals` is provided, the reconstructor looks up the CSC byte ranges from the zarr `csc/indptr` array for each requested feature. It then reads only those column segments from `csc/indices` and the corresponding `csc/layers/{layer}` arrays — O(nnz for wanted features) instead of O(nnz across all cells). This is a significant win when the atlas is large and only a handful of features are requested.

For groups that do not yet have CSC data — groups where `add_csc()` has not been called — the reconstructor falls back to reading CSR and filtering columns. This fallback produces the same output as `SparseCSRReconstructor` for those groups. The fallback means CSC is optional on a per-group basis: you can add it incrementally to existing groups without breaking reconstruction for groups that don't have it yet.

When `wanted_globals` is not provided, `FeatureCSCReconstructor` delegates entirely to `SparseCSRReconstructor`. The CSC path only activates when a feature filter is in play.

---

## Choosing a reconstructor

| Data type | Reconstructor | Notes |
|---|---|---|
| Sparse counts (gene expression, ATAC) | `SparseCSRReconstructor` | Default choice for sparse assays |
| Dense float arrays (protein, embeddings, log-normalized HVGs) | `DenseReconstructor` | Required for `PointerKind.DENSE` |
| Sparse + frequent feature-filtered queries | `FeatureCSCReconstructor` | Requires `add_csc()` on groups; falls back gracefully per group |

Use `FeatureCSCReconstructor` instead of `SparseCSRReconstructor` when:

- Users frequently query specific genes or peaks by UID (e.g., a fixed marker gene panel or a set of GWAS peaks).
- The atlas is large enough that reading all cell rows and then slicing columns is noticeably slower than reading the column slices directly.
- You have already run `add_csc()` on your groups, or plan to. Groups without CSC data incur no penalty — they silently fall back to CSR reads.

`SparseCSRReconstructor` is the simpler default and requires no post-processing step. `FeatureCSCReconstructor` is purely additive: switching the reconstructor on a registered spec requires no changes to ingestion code, schema definitions, or existing zarr data.

---

## Feature join semantics

Both sparse and dense reconstructors respect the `feature_join` argument passed down from `AtlasQuery.feature_join()`.

**Union (default):** the output feature space is the union of global indices across all groups. Each group contributes values only for the features it measures; positions for unmeasured features are zero (sparse: absent from the matrix; dense: zero-filled).

**Intersection:** the output feature space contains only global indices present in every group's remap. Cells from groups that don't have a feature in the intersection have that feature excluded from the output — not zero-filled, simply not in the matrix.

**`wanted_globals`:** when set (via `AtlasQuery.features()`), the feature space is pinned to exactly the requested global indices regardless of what any group measures. Groups that don't have a given feature contribute nothing to that column; `feature_join` has no effect.

---

## Implementing a custom reconstructor

If the built-in reconstructors don't match your data format, implement the `Reconstructor` protocol directly:

```python
import anndata as ad
import polars as pl
import numpy as np
from typing import Literal
from lancell.atlas import RaggedAtlas
from lancell.obs_alignment import PointerFieldInfo
from lancell.group_specs import ZarrGroupSpec

class MyCustomReconstructor:
    def as_anndata(
        self,
        atlas: RaggedAtlas,
        cells_pl: pl.DataFrame,
        pf: PointerFieldInfo,
        spec: ZarrGroupSpec,
        layer_overrides: list[str] | None = None,
        feature_join: Literal["union", "intersection"] = "union",
        wanted_globals: np.ndarray | None = None,
    ) -> ad.AnnData:
        # Load feature metadata, read zarr arrays, assemble AnnData
        ...
```

Pass the instance as the `reconstructor` argument to `ZarrGroupSpec`.

The following helpers from `lancell.reconstruction` handle the parts that are identical across all built-in reconstructors. Using them avoids reimplementing the feature space join logic.

### `_load_remaps_and_features(atlas, groups, spec, feature_join, wanted_globals)`

Loads the per-group local-to-global remap arrays from `_feature_layouts` and computes the joined feature space. Returns a four-tuple:

- `group_remaps` — `{zarr_group: remap_array}` where `remap[local_i] = global_index`.
- `joined_globals` — sorted array of unique global indices in the output feature space.
- `group_remap_to_joined` — `{zarr_group: positions_array}` where `positions[local_i]` is the column in the joined-space output matrix. For intersection or `wanted_globals` mode, local features not in the joined space are mapped to `-1`.
- `n_features` — length of `joined_globals`.

When `wanted_globals` is set, the union/intersection step is skipped and the returned `joined_globals` is exactly `wanted_globals`.

### `_build_obs_df(cells_pl)`

Strips zarr pointer struct columns and any internal `_`-prefixed columns from `cells_pl`, sets `uid` as the index if present, and returns a pandas DataFrame suitable for use as `adata.obs`.

### `_build_var(atlas, feature_space, joined_globals)`

Queries the feature registry table for the rows matching `joined_globals`, sorts them by `global_index`, and returns a pandas DataFrame with `uid` as the index. Used to build `adata.var`.

### `_build_feature_space(remaps, join)`

Lower-level helper used by `_load_remaps_and_features`. Computes the union or intersection of global indices from a `{group: remap_array}` dict and returns `(joined_globals, group_remap_to_joined)`.

---

## Imports

```python
from lancell.reconstruction import SparseCSRReconstructor, DenseReconstructor, FeatureCSCReconstructor
from lancell.protocols import Reconstructor
```
