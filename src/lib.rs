use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::Arc;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rayon::prelude::*;
use tokio::runtime::Runtime;

use zarrs::array::codec::{CodecChain, ShardingCodecConfiguration};
use zarrs::array::{
    Array, ArrayShardedExt, ArrayToBytesCodecTraits, BytesRepresentation, CodecMetadataOptions,
    CodecOptions, DataType, FillValue,
};
use zarrs::metadata::ConfigurationSerialize;
use zarrs_object_store::object_store::aws::{AmazonS3, AmazonS3Builder};
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{ObjectStore, ObjectStoreExt};
use zarrs_object_store::AsyncObjectStore;

/// A fast shard reader for zarr sharded arrays stored on S3.
///
/// Uses zarrs at init time for metadata extraction and inner codec chain reconstruction.
/// At read time, performs batched S3 I/O via `object_store::get_ranges` (one call per shard)
/// and decodes subchunks with zarrs' `CodecChain::decode`.
#[pyclass]
struct RustShardReader {
    /// zarrs array (for chunk_key encoding)
    array: Arc<Array<AsyncObjectStore<AmazonS3>>>,
    /// Direct S3 access for batched I/O
    s3: Arc<AmazonS3>,
    /// Inner codec chain (reconstructed from sharding config)
    inner_codecs: Arc<CodecChain>,
    /// Index codec chain (for decoding shard indexes)
    index_codecs: Arc<CodecChain>,
    /// Subchunk shape as NonZeroU64 slice (for decode calls)
    subchunk_shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    /// Subchunk size in elements (1D)
    chunk_size: usize,
    /// Shard size in elements (1D)
    #[allow(dead_code)]
    shard_size: usize,
    chunks_per_shard: usize,
    dtype_size: usize,
    /// Encoded size of shard index in bytes
    index_encoded_size: usize,
    /// Shard index cache: shard_idx -> flat Vec<u64> of [offset, size, offset, size, ...]
    shard_index_cache: Arc<tokio::sync::Mutex<HashMap<usize, Vec<u64>>>>,
    runtime: Arc<Runtime>,
    codec_options: CodecOptions,
}

#[pymethods]
impl RustShardReader {
    #[new]
    fn new(py_zarr_array: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 1. Extract S3 config from arr.store.store (obstore S3Store)
        let store_wrapper = py_zarr_array.getattr("store")?;
        let s3_store = store_wrapper.getattr("store")?;
        let config = s3_store.getattr("config")?;
        let bucket: String = config
            .call_method1("get", ("bucket", ""))?
            .extract()?;
        let region: String = config
            .call_method1("get", ("region", "us-east-2"))?
            .extract()?;
        let prefix: String = s3_store
            .getattr("prefix")?
            .extract::<Option<String>>()?
            .unwrap_or_default();

        // 2. Build S3 (keep Arc for direct batched I/O)
        let s3 = AmazonS3Builder::new()
            .with_bucket_name(&bucket)
            .with_region(&region)
            .build()
            .map_err(|e| PyRuntimeError::new_err(format!("failed to create S3 store: {e}")))?;
        let s3 = Arc::new(s3);

        // 3. Build zarrs Array for metadata extraction
        let store = Arc::new(AsyncObjectStore::new((*s3).clone()));
        let runtime = Arc::new(
            Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?,
        );
        let prefix_normalized = prefix.trim_matches('/');
        let array = runtime
            .block_on(Array::async_open(store, &format!("/{prefix_normalized}")))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open zarr array: {e}")))?;

        // 4. Extract metadata
        let dtype_size = array
            .data_type()
            .fixed_size()
            .ok_or_else(|| PyRuntimeError::new_err("variable-length dtypes not supported"))?;
        let data_type = array.data_type().clone();
        let fill_value = array.fill_value().clone();

        // Shard shape = chunk shape of the array (1D)
        let shard_shape = array
            .chunk_shape(&[0])
            .map_err(|e| PyRuntimeError::new_err(format!("chunk_shape: {e}")))?;
        let shard_size = shard_shape[0].get() as usize;

        // Subchunk shape from sharding codec
        let subchunk_shape = array
            .subchunk_shape()
            .ok_or_else(|| PyRuntimeError::new_err("array is not sharded"))?;
        let chunk_size = subchunk_shape[0].get() as usize;
        let chunks_per_shard = shard_size / chunk_size;

        // 5. Extract inner codec chain from sharding configuration
        let codec_chain = array.codecs();
        let a2b_codec = codec_chain.array_to_bytes_codec();
        let configuration = a2b_codec
            .configuration_v3(&CodecMetadataOptions::default())
            .ok_or_else(|| PyRuntimeError::new_err("no v3 config for sharding codec"))?;
        let sharding_config = ShardingCodecConfiguration::try_from_configuration(configuration)
            .map_err(|e| PyRuntimeError::new_err(format!("parse sharding config: {e}")))?;
        let ShardingCodecConfiguration::V1(v1) = sharding_config
        else {
            return Err(PyRuntimeError::new_err(
                "unsupported sharding configuration variant",
            ));
        };

        let inner_codecs = Arc::new(
            CodecChain::from_metadata(&v1.codecs)
                .map_err(|e| PyRuntimeError::new_err(format!("inner codecs: {e}")))?,
        );
        let index_codecs = Arc::new(
            CodecChain::from_metadata(&v1.index_codecs)
                .map_err(|e| PyRuntimeError::new_err(format!("index codecs: {e}")))?,
        );

        // 6. Compute index encoded size
        // Index shape for 1D: [chunks_per_shard, 2]
        let index_shape: Vec<NonZeroU64> = vec![
            NonZeroU64::new(chunks_per_shard as u64).unwrap(),
            NonZeroU64::new(2).unwrap(),
        ];
        let uint64_dt = zarrs::array::data_type::uint64();
        let uint64_fv = FillValue::from(u64::MAX);
        let index_encoded_size = match index_codecs.encoded_representation(
            &index_shape,
            &uint64_dt,
            &uint64_fv,
        ) {
            Ok(BytesRepresentation::FixedSize(size)) => size as usize,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "index codecs must produce fixed-size output",
                ))
            }
        };

        let codec_options = CodecOptions::default();

        Ok(Self {
            array: Arc::new(array),
            s3,
            inner_codecs,
            index_codecs,
            subchunk_shape: subchunk_shape.to_vec(),
            data_type,
            fill_value,
            chunk_size,
            shard_size,
            chunks_per_shard,
            dtype_size,
            index_encoded_size,
            shard_index_cache: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            runtime,
            codec_options,
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
    /// and lengths[i] = number of elements in range i.
    fn read_ranges<'py>(
        &self,
        py: Python<'py>,
        starts: &Bound<'py, PyArray1<i64>>,
        ends: &Bound<'py, PyArray1<i64>>,
    ) -> PyResult<(Py<PyArray1<u8>>, Py<PyArray1<i64>>)> {
        let starts_vec: Vec<i64> = unsafe { starts.as_slice()? }.to_vec();
        let ends_vec: Vec<i64> = unsafe { ends.as_slice()? }.to_vec();
        let n = starts_vec.len();
        if n != ends_vec.len() {
            return Err(PyRuntimeError::new_err(
                "starts and ends must have the same length",
            ));
        }

        let array = self.array.clone();
        let s3 = self.s3.clone();
        let inner_codecs = self.inner_codecs.clone();
        let index_codecs = self.index_codecs.clone();
        let cache = self.shard_index_cache.clone();
        let runtime = self.runtime.clone();
        let codec_options = self.codec_options.clone();
        let subchunk_shape = self.subchunk_shape.clone();
        let data_type = self.data_type.clone();
        let fill_value = self.fill_value.clone();
        let chunk_size = self.chunk_size;
        let chunks_per_shard = self.chunks_per_shard;
        let dtype_size = self.dtype_size;
        let index_encoded_size = self.index_encoded_size;

        let (flat_data, lengths_vec) =
            py.allow_threads(move || -> Result<(Vec<u8>, Vec<i64>), String> {
                runtime.block_on(async {
                    // 1. Map ranges to subchunks and group by shard
                    // For each range i, record which subchunks are needed
                    // range_subchunks[i] = vec of (shard_idx, subchunk_within_shard, elem_start_in_subchunk, elem_end_in_subchunk)
                    struct SubchunkRef {
                        shard_idx: usize,
                        subchunk_in_shard: usize,
                        elem_start: usize, // start offset within decoded subchunk (elements)
                        elem_end: usize,   // end offset within decoded subchunk (elements)
                    }

                    let mut range_refs: Vec<Vec<SubchunkRef>> = Vec::with_capacity(n);
                    // shard_subchunks: shard_idx -> set of subchunk indices within shard
                    let mut shard_subchunks: HashMap<usize, Vec<usize>> = HashMap::new();

                    for i in 0..n {
                        let s = starts_vec[i] as usize;
                        let e = ends_vec[i] as usize;
                        let mut refs = Vec::new();

                        if s >= e {
                            range_refs.push(refs);
                            continue;
                        }

                        let first_subchunk = s / chunk_size;
                        let last_subchunk = (e - 1) / chunk_size;

                        for sc in first_subchunk..=last_subchunk {
                            let shard_idx = sc / chunks_per_shard;
                            let subchunk_in_shard = sc % chunks_per_shard;

                            // Element range within this subchunk
                            let sc_start_elem = sc * chunk_size;
                            let elem_start = if s > sc_start_elem { s - sc_start_elem } else { 0 };
                            let elem_end =
                                if e < sc_start_elem + chunk_size { e - sc_start_elem } else { chunk_size };

                            refs.push(SubchunkRef {
                                shard_idx,
                                subchunk_in_shard,
                                elem_start,
                                elem_end,
                            });

                            shard_subchunks
                                .entry(shard_idx)
                                .or_default()
                                .push(subchunk_in_shard);
                        }
                        range_refs.push(refs);
                    }

                    // Deduplicate subchunk lists per shard
                    for subchunks in shard_subchunks.values_mut() {
                        subchunks.sort_unstable();
                        subchunks.dedup();
                    }

                    // 2. For each shard: fetch index + compressed data (async I/O only)
                    let shard_tasks: Vec<_> = shard_subchunks
                        .into_iter()
                        .map(|(shard_idx, needed_subchunks)| {
                            let s3 = s3.clone();
                            let array = array.clone();
                            let cache = cache.clone();
                            let index_codecs = index_codecs.clone();
                            let codec_options = codec_options.clone();

                            tokio::spawn(async move {
                                // a. Get shard key via zarrs
                                let store_key = array.chunk_key(&[shard_idx as u64]);
                                let path = ObjectPath::from(store_key.to_string());

                                // b. Get shard index (cache check first)
                                let shard_index = {
                                    let cache_guard = cache.lock().await;
                                    if let Some(idx) = cache_guard.get(&shard_idx) {
                                        idx.clone()
                                    } else {
                                        drop(cache_guard);

                                        let meta = s3.head(&path).await.map_err(|e| {
                                            format!("HEAD shard {shard_idx}: {e}")
                                        })?;
                                        let shard_len = meta.size as u64;

                                        let index_start = shard_len - index_encoded_size as u64;
                                        let index_bytes = s3
                                            .get_range(&path, index_start..shard_len)
                                            .await
                                            .map_err(|e| {
                                                format!("GET index shard {shard_idx}: {e}")
                                            })?;

                                        let index_shape: Vec<NonZeroU64> = vec![
                                            NonZeroU64::new(chunks_per_shard as u64).unwrap(),
                                            NonZeroU64::new(2).unwrap(),
                                        ];
                                        let uint64_dt = zarrs::array::data_type::uint64();
                                        let uint64_fv = FillValue::from(u64::MAX);
                                        let decoded_index = index_codecs
                                            .decode(
                                                Cow::Owned(index_bytes.to_vec()),
                                                &index_shape,
                                                &uint64_dt,
                                                &uint64_fv,
                                                &codec_options,
                                            )
                                            .map_err(|e| {
                                                format!("decode index shard {shard_idx}: {e}")
                                            })?;
                                        let raw = decoded_index.into_fixed().map_err(|e| {
                                            format!("index into_fixed shard {shard_idx}: {e}")
                                        })?;
                                        let index_vec: Vec<u64> = raw
                                            .as_ref()
                                            .chunks_exact(8)
                                            .map(|b| u64::from_ne_bytes(b.try_into().unwrap()))
                                            .collect();

                                        let mut cache_guard = cache.lock().await;
                                        cache_guard.insert(shard_idx, index_vec.clone());
                                        index_vec
                                    }
                                };

                                // c. Build byte ranges for needed subchunks
                                let mut byte_ranges: Vec<Range<u64>> = Vec::new();
                                let mut subchunk_order: Vec<usize> = Vec::new();
                                let mut fill_subchunks: Vec<(usize, usize)> = Vec::new();

                                for &sc in &needed_subchunks {
                                    let offset = shard_index[sc * 2];
                                    let size = shard_index[sc * 2 + 1];
                                    if offset == u64::MAX && size == u64::MAX {
                                        fill_subchunks.push((shard_idx, sc));
                                    } else {
                                        byte_ranges.push(offset..offset + size);
                                        subchunk_order.push(sc);
                                    }
                                }

                                // d. Batched S3 fetch: ONE get_ranges call per shard
                                let fetched = if byte_ranges.is_empty() {
                                    vec![]
                                } else {
                                    s3.get_ranges(&path, &byte_ranges).await.map_err(|e| {
                                        format!("get_ranges shard {shard_idx}: {e}")
                                    })?
                                };

                                // Return compressed data — decoding happens later via rayon
                                let compressed: Vec<_> = subchunk_order
                                    .into_iter()
                                    .zip(fetched)
                                    .map(|(sc, data)| (shard_idx, sc, data))
                                    .collect();

                                Ok::<_, String>((compressed, fill_subchunks))
                            })
                        })
                        .collect();

                    // Wait for all I/O tasks, collect compressed data
                    let mut all_compressed = Vec::new();
                    let mut all_fills: Vec<(usize, usize)> = Vec::new();
                    for task in shard_tasks {
                        let (compressed, fills) = task
                            .await
                            .map_err(|e| format!("shard task join: {e}"))?
                            .map_err(|e| e)?;
                        all_compressed.extend(compressed);
                        all_fills.extend(fills);
                    }

                    // 3. Decode all subchunks in parallel with rayon (CPU-bound)
                    let decoded_results: Vec<_> = all_compressed
                        .par_iter()
                        .map(|(shard_idx, sc, compressed)| {
                            let decoded = inner_codecs
                                .decode(
                                    Cow::Borrowed(compressed.as_ref()),
                                    &subchunk_shape,
                                    &data_type,
                                    &fill_value,
                                    &codec_options,
                                )
                                .map_err(|e| {
                                    format!("decode subchunk {sc} shard {shard_idx}: {e}")
                                })?;
                            let raw = decoded.into_fixed().map_err(|e| {
                                format!("into_fixed subchunk {sc} shard {shard_idx}: {e}")
                            })?;
                            Ok::<_, String>(((*shard_idx, *sc), raw.into_owned()))
                        })
                        .collect::<Result<Vec<_>, _>>()?;

                    let mut decoded_map: HashMap<(usize, usize), Vec<u8>> =
                        decoded_results.into_iter().collect();

                    // Handle fill-value subchunks
                    for (shard_idx, sc) in all_fills {
                        let fill_bytes = fill_value.as_ne_bytes();
                        let mut buf = vec![0u8; chunk_size * dtype_size];
                        for elem_buf in buf.chunks_exact_mut(fill_bytes.len()) {
                            elem_buf.copy_from_slice(fill_bytes);
                        }
                        decoded_map.insert((shard_idx, sc), buf);
                    }

                    // 4. Assemble output with pre-allocated buffer
                    let total_bytes: usize = range_refs
                        .iter()
                        .flat_map(|refs| refs.iter())
                        .map(|r| (r.elem_end - r.elem_start) * dtype_size)
                        .sum();
                    let mut flat = Vec::with_capacity(total_bytes);
                    let mut lengths = Vec::with_capacity(n);

                    for refs in &range_refs {
                        let num_elements: usize =
                            refs.iter().map(|r| r.elem_end - r.elem_start).sum();
                        lengths.push(num_elements as i64);

                        for r in refs {
                            let decoded = decoded_map
                                .get(&(r.shard_idx, r.subchunk_in_shard))
                                .ok_or_else(|| {
                                    format!(
                                        "missing decoded subchunk ({}, {})",
                                        r.shard_idx, r.subchunk_in_shard
                                    )
                                })?;
                            let byte_start = r.elem_start * dtype_size;
                            let byte_end = r.elem_end * dtype_size;
                            flat.extend_from_slice(&decoded[byte_start..byte_end]);
                        }
                    }

                    Ok((flat, lengths))
                })
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

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
