use std::borrow::Cow;
use std::num::NonZeroU64;
use std::sync::Arc;

use memmap2::MmapMut;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;
use tokio::runtime::Runtime;

use zarrs::array::{Array, ArrayToBytesCodecTraits, CodecOptions, FillValue};
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{ObjectStore, ObjectStoreExt};
use zarrs_object_store::AsyncObjectStore;

use crate::{ensure_bitpack_codec_registered, extract_sharding_meta, ShardingMeta};

// ---------------------------------------------------------------------------
// Sequential shard reader: fetch entire shard, decode all subchunks
// ---------------------------------------------------------------------------

/// Decoded flat data for one shard (all subchunks concatenated in order).
struct ShardData {
    /// Flat decoded u32 values for the entire shard, subchunks in order.
    values: Vec<u32>,
}

/// Fetch and decode one full shard from the object store.
async fn read_full_shard(
    store: &Arc<dyn ObjectStore>,
    array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    meta: &ShardingMeta,
    shard_idx: usize,
    codec_options: &CodecOptions,
) -> Result<Option<ShardData>, String> {
    let mut shard_indices = vec![0u64; meta.ndim];
    shard_indices[0] = shard_idx as u64;
    let store_key = array.chunk_key(&shard_indices);
    let path = ObjectPath::from(store_key.to_string());

    // Try to GET the entire shard file
    let shard_bytes = match store.get(&path).await {
        Ok(result) => result
            .bytes()
            .await
            .map_err(|e| format!("read shard {shard_idx} bytes: {e}"))?,
        Err(zarrs_object_store::object_store::Error::NotFound { .. }) => return Ok(None),
        Err(e) => return Err(format!("GET shard {shard_idx}: {e}")),
    };

    let shard_len = shard_bytes.len() as u64;

    // Decode shard index (last index_encoded_size bytes)
    let index_start = shard_len - meta.index_encoded_size as u64;
    let index_bytes = &shard_bytes[index_start as usize..];

    let index_shape: Vec<NonZeroU64> = vec![
        NonZeroU64::new(meta.chunks_per_shard as u64).unwrap(),
        NonZeroU64::new(2).unwrap(),
    ];
    let uint64_dt = zarrs::array::data_type::uint64();
    let uint64_fv = FillValue::from(u64::MAX);
    let decoded_index = meta
        .index_codecs
        .decode(
            Cow::Borrowed(index_bytes),
            &index_shape,
            &uint64_dt,
            &uint64_fv,
            codec_options,
        )
        .map_err(|e| format!("decode index shard {shard_idx}: {e}"))?;
    let raw_idx = decoded_index
        .into_fixed()
        .map_err(|e| format!("index into_fixed shard {shard_idx}: {e}"))?;
    let shard_index: Vec<u64> = raw_idx
        .as_ref()
        .chunks_exact(8)
        .map(|b: &[u8]| u64::from_ne_bytes(b.try_into().unwrap()))
        .collect();

    // Decode each subchunk
    let elems_per_sc = meta.chunk_size; // elements per subchunk (axis-0)
    let mut all_values = Vec::with_capacity(meta.chunks_per_shard * elems_per_sc);

    let fill_bytes = meta.fill_value.as_ne_bytes();

    for sc in 0..meta.chunks_per_shard {
        let offset = shard_index[sc * 2];
        let size = shard_index[sc * 2 + 1];

        if offset == u64::MAX && size == u64::MAX {
            // Fill-value subchunk
            let fill_u32 = if fill_bytes.len() == 4 {
                u32::from_ne_bytes(fill_bytes.try_into().unwrap())
            } else {
                0u32
            };
            all_values.resize(all_values.len() + elems_per_sc, fill_u32);
        } else {
            let sc_bytes = &shard_bytes[offset as usize..(offset + size) as usize];
            let decoded = meta
                .inner_codecs
                .decode(
                    Cow::Borrowed(sc_bytes),
                    &meta.subchunk_shape,
                    &meta.data_type,
                    &meta.fill_value,
                    codec_options,
                )
                .map_err(|e| format!("decode subchunk {sc} shard {shard_idx}: {e}"))?;
            let raw = decoded
                .into_fixed()
                .map_err(|e| format!("into_fixed sc {sc} shard {shard_idx}: {e}"))?;

            // Convert raw bytes to u32 values
            for chunk in raw.as_ref().chunks_exact(4) {
                let chunk: &[u8] = chunk;
                all_values.push(u32::from_ne_bytes(chunk.try_into().unwrap()));
            }
        }
    }

    Ok(Some(ShardData {
        values: all_values,
    }))
}

// ---------------------------------------------------------------------------
// Two-pass CSR-to-CSC conversion
// ---------------------------------------------------------------------------

/// Open a zarrs array from store + path.
fn open_zarrs_array(
    store: Arc<dyn ObjectStore>,
    path: &str,
    runtime: &Runtime,
) -> Result<Array<AsyncObjectStore<Arc<dyn ObjectStore>>>, String> {
    let zarrs_store = Arc::new(AsyncObjectStore::new(store));
    let store_path = if path.is_empty() {
        "/".to_string()
    } else if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    runtime
        .block_on(Array::async_open(zarrs_store, &store_path))
        .map_err(|e| format!("failed to open zarr array at '{path}': {e}"))
}

/// Compute number of shards for an array of `total_elems` elements.
fn n_shards(total_elems: u64, chunk_size: usize, chunks_per_shard: usize) -> usize {
    let shard_elems = (chunk_size * chunks_per_shard) as u64;
    ((total_elems + shard_elems - 1) / shard_elems) as usize
}

/// Pass 1: Count occurrences of each feature index across all CSR indices shards.
async fn pass1_count(
    store: &Arc<dyn ObjectStore>,
    indices_array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    indices_meta: &ShardingMeta,
    n_features: u32,
    total_nnz: u64,
    codec_options: &CodecOptions,
) -> Result<Vec<u64>, String> {
    let num_shards = n_shards(total_nnz, indices_meta.chunk_size, indices_meta.chunks_per_shard);
    let shard_elems = (indices_meta.chunk_size * indices_meta.chunks_per_shard) as u64;
    let mut col_counts = vec![0u64; n_features as usize];

    // Determine fill_value as u32 for handling missing (all-fill) shards
    let fill_bytes = indices_meta.fill_value.as_ne_bytes();
    let fill_u32 = if fill_bytes.len() == 4 {
        u32::from_ne_bytes(fill_bytes.try_into().unwrap())
    } else {
        0u32
    };

    for shard_idx in 0..num_shards {
        let shard_start = shard_idx as u64 * shard_elems;
        let valid_count = std::cmp::min(
            shard_elems,
            total_nnz.saturating_sub(shard_start),
        ) as usize;

        let shard = read_full_shard(store, indices_array, indices_meta, shard_idx, codec_options)
            .await?;
        match shard {
            Some(shard_data) => {
                let n = std::cmp::min(shard_data.values.len(), valid_count);
                for &val in &shard_data.values[..n] {
                    if (val as usize) < col_counts.len() {
                        col_counts[val as usize] += 1;
                    }
                }
            }
            None => {
                // Shard not written = all elements are fill_value
                if (fill_u32 as usize) < col_counts.len() {
                    col_counts[fill_u32 as usize] += valid_count as u64;
                }
            }
        }
    }

    Ok(col_counts)
}

/// Compute col_ptr (prefix sum) from col_counts.
fn build_col_ptr(col_counts: &[u64]) -> Vec<u64> {
    let mut col_ptr = Vec::with_capacity(col_counts.len() + 1);
    col_ptr.push(0);
    let mut acc = 0u64;
    for &c in col_counts {
        acc += c;
        col_ptr.push(acc);
    }
    col_ptr
}

/// Pass 2: Scatter-fill CSC indices and values into mmap'd temp files.
async fn pass2_scatter(
    store: &Arc<dyn ObjectStore>,
    indices_array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    values_array: &Array<AsyncObjectStore<Arc<dyn ObjectStore>>>,
    indices_meta: &ShardingMeta,
    values_meta: &ShardingMeta,
    starts: &[i64],
    ends: &[i64],
    col_ptr: &[u64],
    nnz: u64,
    csc_indices_mmap: &mut MmapMut,
    csc_values_mmap: &mut MmapMut,
    codec_options: &CodecOptions,
) -> Result<(), String> {
    let n_features = col_ptr.len() - 1;
    let mut write_pos: Vec<u64> = col_ptr[..n_features].to_vec();

    let num_idx_shards = n_shards(nnz, indices_meta.chunk_size, indices_meta.chunks_per_shard);

    let n_cells = starts.len();
    let mut cell_cursor: usize = 0;

    let shard_elems_idx =
        (indices_meta.chunk_size * indices_meta.chunks_per_shard) as u64;
    let shard_elems_val =
        (values_meta.chunk_size * values_meta.chunks_per_shard) as u64;

    let mut cur_val_shard: Option<(usize, Vec<u32>)> = None;

    // Advance cell_cursor past any empty cells at the start
    while cell_cursor < n_cells && starts[cell_cursor] == ends[cell_cursor] {
        cell_cursor += 1;
    }

    // Determine fill_value as u32 for indices (for missing shards)
    let idx_fill_bytes = indices_meta.fill_value.as_ne_bytes();
    let idx_fill_u32 = if idx_fill_bytes.len() == 4 {
        u32::from_ne_bytes(idx_fill_bytes.try_into().unwrap())
    } else {
        0u32
    };

    // Process shard by shard (indices array drives the outer loop)
    for idx_shard_idx in 0..num_idx_shards {
        let shard_start_elem = idx_shard_idx as u64 * shard_elems_idx;
        let valid_in_shard = std::cmp::min(
            shard_elems_idx,
            nnz.saturating_sub(shard_start_elem),
        ) as usize;

        // Fetch indices shard (None = all fill_value)
        let idx_shard = read_full_shard(
            store, indices_array, indices_meta, idx_shard_idx, codec_options
        ).await?;
        let idx_values: Vec<u32> = match idx_shard {
            Some(s) => s.values,
            None => vec![idx_fill_u32; valid_in_shard],
        };

        for (local_pos, &feature_idx) in idx_values.iter().enumerate() {
            let elem = shard_start_elem + local_pos as u64;
            if elem >= nnz {
                break;
            }

            // Advance cell cursor
            while cell_cursor < n_cells && elem >= ends[cell_cursor] as u64 {
                cell_cursor += 1;
            }
            if cell_cursor >= n_cells {
                break;
            }

            // Skip elements that don't belong to any cell (shouldn't happen with valid data)
            if elem < starts[cell_cursor] as u64 {
                continue;
            }

            let cell_id = cell_cursor as u32;

            // Fetch the corresponding value
            let val_shard_needed = (elem / shard_elems_val) as usize;
            if cur_val_shard.as_ref().map_or(true, |(si, _)| *si != val_shard_needed) {
                let vs = read_full_shard(
                    store, values_array, values_meta, val_shard_needed, codec_options
                ).await?;
                let val_fill_bytes = values_meta.fill_value.as_ne_bytes();
                let val_fill_u32 = if val_fill_bytes.len() == 4 {
                    u32::from_ne_bytes(val_fill_bytes.try_into().unwrap())
                } else {
                    0u32
                };
                let valid_val = std::cmp::min(
                    shard_elems_val,
                    nnz.saturating_sub(val_shard_needed as u64 * shard_elems_val),
                ) as usize;
                cur_val_shard = Some((
                    val_shard_needed,
                    vs.map_or_else(|| vec![val_fill_u32; valid_val], |s| s.values),
                ));
            }
            let val_local = (elem - val_shard_needed as u64 * shard_elems_val) as usize;
            let value = cur_val_shard
                .as_ref()
                .and_then(|(_, v)| v.get(val_local).copied())
                .unwrap_or(0);

            // Scatter-write
            let fi = feature_idx as usize;
            if fi < n_features {
                let wp = write_pos[fi] as usize;
                let byte_off = wp * 4;
                csc_indices_mmap[byte_off..byte_off + 4]
                    .copy_from_slice(&cell_id.to_ne_bytes());
                csc_values_mmap[byte_off..byte_off + 4]
                    .copy_from_slice(&value.to_ne_bytes());
                write_pos[fi] += 1;
            }
        }
    }

    // Consistency check
    for j in 0..n_features {
        if write_pos[j] != col_ptr[j + 1] {
            return Err(format!(
                "consistency check failed: write_pos[{j}]={} != col_ptr[{j}+1]={}",
                write_pos[j],
                col_ptr[j + 1]
            ));
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// PyO3 entry point
// ---------------------------------------------------------------------------

#[pyfunction]
pub fn csr_to_csc<'py>(
    py: Python<'py>,
    store: &Bound<'py, PyAny>,
    csr_indices_path: &str,
    csr_layer_path: &str,
    starts: &Bound<'py, PyArray1<i64>>,
    ends: &Bound<'py, PyArray1<i64>>,
    n_features: u32,
    tmp_dir: &str,
) -> PyResult<(String, String, Py<PyArray1<i64>>, Py<PyArray1<i64>>)> {
    ensure_bitpack_codec_registered();

    // Extract Arc<dyn ObjectStore> from obstore
    let any_store: AnyObjectStore = store.extract()?;
    let obj_store: Arc<dyn ObjectStore> = any_store.into_dyn();

    // Copy starts/ends to Vecs
    let starts_vec: Vec<i64> = unsafe { starts.as_slice()? }.to_vec();
    let ends_vec: Vec<i64> = unsafe { ends.as_slice()? }.to_vec();
    if starts_vec.len() != ends_vec.len() {
        return Err(PyRuntimeError::new_err(
            "starts and ends must have the same length",
        ));
    }

    // Compute total nnz from ends (last cell's end value = total elements)
    let nnz: u64 = if ends_vec.is_empty() {
        0
    } else {
        *ends_vec.iter().max().unwrap() as u64
    };

    let indices_path = csr_indices_path.to_string();
    let layer_path = csr_layer_path.to_string();
    let tmp = tmp_dir.to_string();
    let nf = n_features;

    // Release GIL for the heavy computation
    let result = py.detach(move || -> Result<(String, String, Vec<i64>, Vec<i64>), String> {
        let runtime = Runtime::new().map_err(|e| format!("tokio runtime: {e}"))?;

        // Open zarrs arrays for metadata extraction
        let indices_array = open_zarrs_array(Arc::clone(&obj_store), &indices_path, &runtime)?;
        let values_array = open_zarrs_array(Arc::clone(&obj_store), &layer_path, &runtime)?;

        let indices_meta = extract_sharding_meta(&indices_array)?;
        let values_meta = extract_sharding_meta(&values_array)?;

        let codec_options = CodecOptions::default();

        // Pass 1: Count
        let col_counts = runtime.block_on(pass1_count(
            &obj_store,
            &indices_array,
            &indices_meta,
            nf,
            nnz,
            &codec_options,
        ))?;

        let col_ptr = build_col_ptr(&col_counts);
        let total_nnz_check = *col_ptr.last().unwrap();

        // Create temp files for mmap
        let indices_path = format!("{}/csc_indices.tmp", tmp);
        let values_path = format!("{}/csc_values.tmp", tmp);

        let total_bytes = total_nnz_check as usize * 4;

        // Create and size the temp files
        {
            let f = std::fs::File::create(&indices_path)
                .map_err(|e| format!("create indices tmp: {e}"))?;
            f.set_len(total_bytes as u64)
                .map_err(|e| format!("set_len indices tmp: {e}"))?;
        }
        {
            let f = std::fs::File::create(&values_path)
                .map_err(|e| format!("create values tmp: {e}"))?;
            f.set_len(total_bytes as u64)
                .map_err(|e| format!("set_len values tmp: {e}"))?;
        }

        // Memory-map the temp files
        let idx_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&indices_path)
            .map_err(|e| format!("open indices mmap: {e}"))?;
        let val_file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&values_path)
            .map_err(|e| format!("open values mmap: {e}"))?;

        let mut csc_indices_mmap = if total_bytes > 0 {
            unsafe {
                MmapMut::map_mut(&idx_file).map_err(|e| format!("mmap indices: {e}"))?
            }
        } else {
            MmapMut::map_anon(0).map_err(|e| format!("anon mmap: {e}"))?
        };

        let mut csc_values_mmap = if total_bytes > 0 {
            unsafe {
                MmapMut::map_mut(&val_file).map_err(|e| format!("mmap values: {e}"))?
            }
        } else {
            MmapMut::map_anon(0).map_err(|e| format!("anon mmap: {e}"))?
        };

        // Pass 2: Scatter-fill
        runtime.block_on(pass2_scatter(
            &obj_store,
            &indices_array,
            &values_array,
            &indices_meta,
            &values_meta,
            &starts_vec,
            &ends_vec,
            &col_ptr,
            nnz,
            &mut csc_indices_mmap,
            &mut csc_values_mmap,
            &codec_options,
        ))?;

        // Flush mmaps
        csc_indices_mmap.flush().map_err(|e| format!("flush indices: {e}"))?;
        csc_values_mmap.flush().map_err(|e| format!("flush values: {e}"))?;

        // Build csc_start / csc_end from col_ptr
        let n = nf as usize;
        let csc_start: Vec<i64> = col_ptr[..n].iter().map(|&v| v as i64).collect();
        let csc_end: Vec<i64> = col_ptr[1..=n].iter().map(|&v| v as i64).collect();

        Ok((indices_path, values_path, csc_start, csc_end))
    });

    let (idx_path, val_path, csc_start_vec, csc_end_vec) =
        result.map_err(|e| PyRuntimeError::new_err(e))?;

    let csc_start = PyArray1::from_vec(py, csc_start_vec).into();
    let csc_end = PyArray1::from_vec(py, csc_end_vec).into();

    Ok((idx_path, val_path, csc_start, csc_end))
}
