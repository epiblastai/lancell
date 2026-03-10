use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use tokio::runtime::Runtime;

/// A fast shard reader for zarr sharded arrays stored on S3.
///
/// Bypasses zarr's per-chunk codec pipeline and performs all I/O + zstd
/// decompression in Rust with true parallelism (no GIL).
#[pyclass]
#[allow(dead_code)]
struct RustShardReader {
    store: Arc<dyn ObjectStore>,
    runtime: Arc<Runtime>,
    prefix: String,
    chunk_size: usize,
    shard_size: usize,
    chunks_per_shard: usize,
    dtype_size: usize,
    index_raw_bytes: usize,
    index_total_bytes: usize,
    dtype_str: String,
}

const MAX_UINT64: u64 = u64::MAX;

#[pymethods]
impl RustShardReader {
    #[new]
    fn new(
        bucket: &str,
        prefix: &str,
        region: &str,
        chunk_size: usize,
        shard_size: usize,
        dtype: &str,
    ) -> PyResult<Self> {
        let dtype_size = match dtype {
            "int32" | "float32" => 4,
            "int64" | "float64" => 8,
            "int16" | "float16" => 2,
            "int8" | "uint8" => 1,
            _ => {
                return Err(PyRuntimeError::new_err(format!(
                    "unsupported dtype: {dtype}"
                )))
            }
        };

        let chunks_per_shard = shard_size / chunk_size;
        let index_raw_bytes = chunks_per_shard * 2 * 8;
        let index_total_bytes = index_raw_bytes + 4; // +4 for crc32c

        let store = AmazonS3Builder::new()
            .with_bucket_name(bucket)
            .with_region(region)
            .with_allow_http(false)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create S3 store: {e}")))?;

        let runtime = Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create tokio runtime: {e}")))?;

        // Normalize prefix: strip leading/trailing slashes
        let prefix = prefix.trim_matches('/').to_string();

        Ok(Self {
            store: Arc::new(store),
            runtime: Arc::new(runtime),
            prefix,
            chunk_size,
            shard_size,
            chunks_per_shard,
            dtype_size,
            index_raw_bytes,
            index_total_bytes,
            dtype_str: dtype.to_string(),
        })
    }

    /// Read element ranges from the sharded array.
    ///
    /// Parameters
    /// ----------
    /// starts, ends : 1-D int64 arrays of element start/end positions.
    ///
    /// Returns
    /// -------
    /// (flat_data, lengths) where flat_data is the concatenated raw bytes
    /// reinterpreted as the target dtype, and lengths[i] = ends[i] - starts[i].
    fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        starts: &Bound<'py, PyArray1<i64>>,
        ends: &Bound<'py, PyArray1<i64>>,
    ) -> PyResult<(Py<PyArray1<u8>>, Py<PyArray1<i64>>)> {
        // Copy numpy arrays into Rust Vecs before releasing the GIL
        let starts_vec: Vec<i64> = unsafe { starts.as_slice()? }.to_vec();
        let ends_vec: Vec<i64> = unsafe { ends.as_slice()? }.to_vec();

        let n_cells = starts_vec.len();
        if n_cells != ends_vec.len() {
            return Err(PyRuntimeError::new_err(
                "starts and ends must have the same length",
            ));
        }

        let store = self.store.clone();
        let runtime = self.runtime.clone();
        let chunk_size = self.chunk_size;
        let chunks_per_shard = self.chunks_per_shard;
        let dtype_size = self.dtype_size;
        let index_raw_bytes = self.index_raw_bytes;
        let index_total_bytes = self.index_total_bytes;
        let prefix = self.prefix.clone();
        let chunk_bytes = chunk_size * dtype_size;

        // Release the GIL for all heavy work
        let (flat_data, lengths_vec) = py.allow_threads(move || -> Result<(Vec<u8>, Vec<i64>), String> {
            // Step 1: Map ranges to chunks
            // chunk_requests: abs_chunk_idx -> Vec<(cell_idx, local_start_elem, local_end_elem)>
            let mut chunk_requests: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();

            for cell_idx in 0..n_cells {
                let s = starts_vec[cell_idx] as usize;
                let e = ends_vec[cell_idx] as usize;
                let mut pos = s;
                while pos < e {
                    let abs_chunk = pos / chunk_size;
                    let local_start = pos % chunk_size;
                    let chunk_end = std::cmp::min(e, (abs_chunk + 1) * chunk_size);
                    let local_end = local_start + (chunk_end - pos);
                    chunk_requests
                        .entry(abs_chunk)
                        .or_default()
                        .push((cell_idx, local_start, local_end));
                    pos = chunk_end;
                }
            }

            // Step 2: Group chunks by shard
            let mut shard_chunks: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
            for &abs_chunk in chunk_requests.keys() {
                let shard_idx = abs_chunk / chunks_per_shard;
                let local_chunk = abs_chunk % chunks_per_shard;
                shard_chunks.entry(shard_idx).or_default().insert(local_chunk);
            }

            // Step 3 & 4: Fetch shard indexes + chunk data concurrently via tokio
            let shard_ids: Vec<usize> = shard_chunks.keys().copied().collect();

            let all_compressed: Vec<(usize, Vec<u8>, bool)> = runtime.block_on(async {
                let mut tasks = Vec::new();

                for &shard_idx in &shard_ids {
                    let store = store.clone();
                    let local_chunks: Vec<usize> =
                        shard_chunks[&shard_idx].iter().copied().collect();
                    let shard_key =
                        ObjectPath::from(format!("{}/c/{}", prefix, shard_idx));

                    tasks.push(tokio::spawn(async move {
                        // Fetch shard index (last N bytes)
                        let meta = store.head(&shard_key).await.map_err(|e| {
                            format!("HEAD {shard_key} failed: {e}")
                        })?;
                        let fsize = meta.size;
                        let idx_start = fsize - index_total_bytes;
                        let idx_range = idx_start..fsize;
                        let idx_bytes = store
                            .get_range(&shard_key, idx_range)
                            .await
                            .map_err(|e| format!("GET index {shard_key} failed: {e}"))?;

                        // Parse index: 2 x u64 per chunk, little-endian
                        let idx_data = &idx_bytes[..index_raw_bytes];

                        // Build byte ranges for needed chunks
                        let mut ranges: Vec<std::ops::Range<usize>> = Vec::new();
                        let mut valid_locals: Vec<usize> = Vec::new();
                        let mut empty_chunks: Vec<(usize, Vec<u8>)> = Vec::new();

                        for &lc in &local_chunks {
                            let base = lc * 16;
                            let offset = u64::from_le_bytes(
                                idx_data[base..base + 8].try_into().unwrap(),
                            );
                            let length = u64::from_le_bytes(
                                idx_data[base + 8..base + 16].try_into().unwrap(),
                            );

                            if offset == MAX_UINT64 {
                                // Empty chunk - zeros
                                let abs_chunk = shard_idx * chunks_per_shard + lc;
                                empty_chunks
                                    .push((abs_chunk, vec![0u8; chunk_bytes]));
                                continue;
                            }

                            ranges.push(offset as usize..(offset + length) as usize);
                            valid_locals.push(lc);
                        }

                        // Fetch all byte ranges for this shard
                        let buffers = if !ranges.is_empty() {
                            store
                                .get_ranges(&shard_key, &ranges)
                                .await
                                .map_err(|e| {
                                    format!("GET ranges {shard_key} failed: {e}")
                                })?
                        } else {
                            vec![]
                        };

                        // Collect compressed buffers with their abs_chunk_idx
                        let mut result: Vec<(usize, Vec<u8>, bool)> = Vec::new();
                        for (lc, buf) in valid_locals.iter().zip(buffers.into_iter()) {
                            let abs_chunk = shard_idx * chunks_per_shard + lc;
                            result.push((abs_chunk, buf.to_vec(), true));
                        }
                        for (abs_chunk, data) in empty_chunks {
                            result.push((abs_chunk, data, false));
                        }

                        Ok::<Vec<(usize, Vec<u8>, bool)>, String>(result)
                    }));
                }

                let mut all_compressed: Vec<(usize, Vec<u8>, bool)> = Vec::new();
                for task in tasks {
                    let chunks = task.await.map_err(|e| format!("task join error: {e}")).unwrap()?;
                    all_compressed.extend(chunks);
                }

                Ok::<_, String>(all_compressed)
            }).map_err(|e| format!("async fetch failed: {e}"))?;

            // Step 5: Decompress with rayon (true CPU parallelism)
            let decoded: Vec<(usize, Vec<u8>)> = all_compressed
                .into_par_iter()
                .map(|(abs_chunk, data, is_compressed)| {
                    if !is_compressed {
                        return Ok((abs_chunk, data));
                    }
                    let decoded = zstd::decode_all(data.as_slice())
                        .map_err(|e| format!("zstd decode chunk {abs_chunk} failed: {e}"))?;
                    Ok::<_, String>((abs_chunk, decoded))
                })
                .collect::<Result<Vec<_>, _>>()?;

            let decoded_map: HashMap<usize, Vec<u8>> =
                decoded.into_iter().collect();

            // Step 6: Slice and assemble output
            let lengths_vec: Vec<i64> = (0..n_cells)
                .map(|i| ends_vec[i] - starts_vec[i])
                .collect();
            let total_elements: usize = lengths_vec.iter().map(|&l| l as usize).sum();
            let total_bytes = total_elements * dtype_size;
            let mut out = vec![0u8; total_bytes];
            let mut out_pos: usize = 0;

            for cell_idx in 0..n_cells {
                let s = starts_vec[cell_idx] as usize;
                let e = ends_vec[cell_idx] as usize;
                let mut pos = s;
                while pos < e {
                    let abs_chunk = pos / chunk_size;
                    let local_start = pos % chunk_size;
                    let chunk_end = std::cmp::min(e, (abs_chunk + 1) * chunk_size);
                    let local_end = local_start + (chunk_end - pos);
                    let n_elem = local_end - local_start;
                    let n_bytes = n_elem * dtype_size;

                    let chunk_data = decoded_map.get(&abs_chunk).ok_or_else(|| {
                        format!("missing decoded chunk {abs_chunk}")
                    })?;

                    let src_start = local_start * dtype_size;
                    let src_end = local_end * dtype_size;
                    out[out_pos..out_pos + n_bytes]
                        .copy_from_slice(&chunk_data[src_start..src_end]);
                    out_pos += n_bytes;
                    pos = chunk_end;
                }
            }

            Ok((out, lengths_vec))
        }).map_err(|e| PyRuntimeError::new_err(e))?;

        // Convert to numpy arrays
        // flat_data is raw bytes - we'll return as u8 and let Python reinterpret via .view()
        // Actually, let's return properly typed. We know the dtype.
        let flat_array = PyArray1::from_vec(py, flat_data).into();
        let lengths_array = PyArray1::from_vec(py, lengths_vec).into();

        Ok((flat_array, lengths_array))
    }
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustShardReader>()?;
    Ok(())
}
