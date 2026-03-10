# Lancell Data Structure

Single table for cell metadata with columns for zarr group path and a uint64 blob storing the range (use morton order to get the set of ranges for a bounding box in a multidimensional array). Pattern for zarr group path and blob range columns is f`_zarr_path_{feature_space}` and f`_zarr_ranges_{feature_space}`. The schema can have columns for all of the feature spaces. In practice this should be a relatively small set. Even if there are hundreds, it's not bad.
- Do we want to store icechunk version in addition to the group and ranges? Probably not now.

- Handling multimodal cells?
  - Just load the relevant zarr ath and feature space columns
- Handling chromosomes and bulk data?
- Linking perturbations directly to DNA (how possible is this?)
  - This is great for some kinds of perturbations which target enhancers and promoters and let's us query perturbations by genomic loci instead of plain gene name.
- Reference genome arrays (or tables with 1 base per?)
  - Can store the gene name annotations in the columns
- Add columns for each zarr feature space (lance handles column addition quite well). Extremely fast DNA read requests for contiguous regions, from zarr or from table.

custom var_dfs per dataset that can be merged to the main gene table for single dataset analysis, this also naturally tracks the measured features because we store the original var df as a parquet.

How do we handle multi-dataset concatenation when measured feature spaces might be different? Easy. Concat the COO arrays (more specifically the counts and indices arrays). Can always convert to CSR from there.

## Sparse Arrays

Zarr group -> indices and data_layers.
data_layers -> counts [could have log_counts, log_normalized_counts; all share the same `indices` array]

for fragments there is no var_indices unless there are called peaks, then there are var_index_starts and var_index_ends.


## Dense multidimensional arrays

- Morton order (Z-order curve) should be handled internally.