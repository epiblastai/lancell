mod bitpack_codec;
mod bitpacking;

use std::borrow::Cow;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3_object_store::AnyObjectStore;
use rayon::prelude::*;
use tokio::runtime::Runtime;

use zarrs::array::codec::api::CodecRuntimeRegistryHandleV3;

/// Global handle for the bitpacking codec registration.
/// Kept alive for the lifetime of the process so the codec stays registered.
static BITPACK_CODEC_HANDLE: OnceLock<CodecRuntimeRegistryHandleV3> = OnceLock::new();

fn ensure_bitpack_codec_registered() {
    BITPACK_CODEC_HANDLE.get_or_init(bitpack_codec::register_bitpack_codec);
}

use zarrs::array::codec::{CodecChain, ShardingCodecConfiguration};
use zarrs::array::{
    Array, ArrayShardedExt, ArrayToBytesCodecTraits, BytesRepresentation, CodecMetadataOptions,
    CodecOptions, DataType, FillValue,
};
use zarrs::metadata::ConfigurationSerialize;
use zarrs_object_store::object_store::path::Path as ObjectPath;
use zarrs_object_store::object_store::{ObjectStore, ObjectStoreExt};
use zarrs_object_store::AsyncObjectStore;

// ---------------------------------------------------------------------------
// Shared sharding metadata extraction
// ---------------------------------------------------------------------------

/// Metadata extracted from a sharded zarr array.
pub(crate) struct ShardingMeta {
    pub inner_codecs: Arc<CodecChain>,
    pub index_codecs: Arc<CodecChain>,
    pub subchunk_shape: Vec<NonZeroU64>,
    pub data_type: DataType,
    pub fill_value: FillValue,
    pub chunk_size: usize,
    pub chunks_per_shard: usize,
    pub element_stride: usize,
    pub ndim: usize,
    pub index_encoded_size: usize,
}

/// Extract sharding metadata from a zarrs Array.
/// Works with any storage backend since it only accesses cached metadata.
pub(crate) fn extract_sharding_meta<T>(
    array: &Array<T>,
) -> Result<ShardingMeta, String> {
    let dtype_size = array
        .data_type()
        .fixed_size()
        .ok_or("variable-length dtypes not supported")?;
    let data_type = array.data_type().clone();
    let fill_value = array.fill_value().clone();

    let ndim = array.shape().len();
    let zero_indices: Vec<u64> = vec![0; ndim];
    let shard_shape = array
        .chunk_shape(&zero_indices)
        .map_err(|e| format!("chunk_shape: {e}"))?;

    let subchunk_shape = array
        .subchunk_shape()
        .ok_or("array is not sharded")?;
    let chunk_size = subchunk_shape[0].get() as usize;
    let chunks_per_shard: usize = shard_shape
        .iter()
        .zip(subchunk_shape.iter())
        .map(|(s, c)| s.get() as usize / c.get() as usize)
        .product();
    let element_stride: usize = subchunk_shape[1..]
        .iter()
        .map(|d| d.get() as usize)
        .product::<usize>()
        * dtype_size;

    let codec_chain = array.codecs();
    let a2b_codec = codec_chain.array_to_bytes_codec();
    let configuration = a2b_codec
        .configuration_v3(&CodecMetadataOptions::default())
        .ok_or("no v3 config for sharding codec")?;
    let sharding_config = ShardingCodecConfiguration::try_from_configuration(configuration)
        .map_err(|e| format!("parse sharding config: {e}"))?;
    let ShardingCodecConfiguration::V1(v1) = sharding_config
    else {
        return Err("unsupported sharding configuration variant".into());
    };

    let inner_codecs = Arc::new(
        CodecChain::from_metadata(&v1.codecs)
            .map_err(|e| format!("inner codecs: {e}"))?,
    );
    let index_codecs = Arc::new(
        CodecChain::from_metadata(&v1.index_codecs)
            .map_err(|e| format!("index codecs: {e}"))?,
    );

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
        _ => return Err("index codecs must produce fixed-size output".into()),
    };

    Ok(ShardingMeta {
        inner_codecs,
        index_codecs,
        subchunk_shape: subchunk_shape.to_vec(),
        data_type,
        fill_value,
        chunk_size,
        chunks_per_shard,
        element_stride,
        ndim,
        index_encoded_size,
    })
}

// ---------------------------------------------------------------------------
// Types used across read_ranges phases
// ---------------------------------------------------------------------------

/// Reference from one input range to a slice within a decoded subchunk.
struct SubchunkRef {
    shard_idx: usize,
    subchunk_in_shard: usize,
    elem_start: usize, // start offset within decoded subchunk (elements)
    elem_end: usize,   // end offset within decoded subchunk (elements)
}

/// A fast shard reader for zarr sharded arrays.
///
/// Uses zarrs at init time for metadata extraction and inner codec chain reconstruction.
/// At read time, performs batched I/O via `object_store::get_ranges` (one call per shard)
/// and decodes subchunks with zarrs' `CodecChain::decode`.
///
/// Works with any object store backend supported by obstore (S3, GCS, Azure, local, etc.).
#[pyclass]
struct RustBatchReader {
    /// zarrs array (for chunk_key encoding)
    array: Arc<Array<AsyncObjectStore<Arc<dyn ObjectStore>>>>,
    /// Direct store access for batched I/O
    store: Arc<dyn ObjectStore>,
    /// Inner codec chain (reconstructed from sharding config)
    inner_codecs: Arc<CodecChain>,
    /// Index codec chain (for decoding shard indexes)
    index_codecs: Arc<CodecChain>,
    /// Subchunk shape as NonZeroU64 slice (for decode calls)
    subchunk_shape: Vec<NonZeroU64>,
    data_type: DataType,
    fill_value: FillValue,
    /// Number of rows (axis-0 elements) per subchunk
    chunk_size: usize,
    chunks_per_shard: usize,
    /// Bytes per row along axis 0: product(subchunk_shape[1:]) * dtype_size.
    /// For 1D arrays this equals dtype_size.
    element_stride: usize,
    /// Number of dimensions (for chunk_key calls)
    ndim: usize,
    /// Encoded size of shard index in bytes
    index_encoded_size: usize,
    /// Shard index cache: shard_idx -> flat Vec<u64> of [offset, size, offset, size, ...]
    shard_index_cache: Arc<tokio::sync::Mutex<HashMap<usize, Vec<u64>>>>,
    runtime: Arc<Runtime>,
    codec_options: CodecOptions,
}

// ---------------------------------------------------------------------------
// Internal helpers (non-pymethod)
// ---------------------------------------------------------------------------

impl RustBatchReader {
    /// Phase 1: Map element ranges to subchunk references, grouped by shard.
    ///
    /// Returns per-range subchunk refs and a deduplicated shard→subchunk-indices map.
    fn map_ranges_to_subchunks(
        &self,
        starts: &[i64],
        ends: &[i64],
    ) -> (Vec<Vec<SubchunkRef>>, HashMap<usize, Vec<usize>>) {
        let chunk_size = self.chunk_size;
        let chunks_per_shard = self.chunks_per_shard;

        let mut range_refs: Vec<Vec<SubchunkRef>> = Vec::with_capacity(starts.len());
        let mut shard_subchunks: HashMap<usize, Vec<usize>> = HashMap::new();

        for i in 0..starts.len() {
            let s = starts[i] as usize;
            let e = ends[i] as usize;
            let mut refs = Vec::new();

            if s < e {
                let first_subchunk = s / chunk_size;
                let last_subchunk = (e - 1) / chunk_size;

                for sc in first_subchunk..=last_subchunk {
                    let shard_idx = sc / chunks_per_shard;
                    let subchunk_in_shard = sc % chunks_per_shard;

                    let sc_start_elem = sc * chunk_size;
                    let elem_start = if s > sc_start_elem { s - sc_start_elem } else { 0 };
                    let elem_end = if e < sc_start_elem + chunk_size {
                        e - sc_start_elem
                    } else {
                        chunk_size
                    };

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
            }
            range_refs.push(refs);
        }

        // Deduplicate subchunk lists per shard
        for subchunks in shard_subchunks.values_mut() {
            subchunks.sort_unstable();
            subchunks.dedup();
        }

        (range_refs, shard_subchunks)
    }

    /// Phase 2: Fetch shard indexes and compressed subchunk data from the store.
    ///
    /// For each shard, resolves the object key, fetches/caches the shard index,
    /// then issues a single `get_ranges` call for all needed subchunks.
    /// Returns (compressed_data, fill_subchunks).
    async fn fetch_shard_data(
        &self,
        shard_subchunks: HashMap<usize, Vec<usize>>,
    ) -> Result<(Vec<(usize, usize, Vec<u8>)>, Vec<(usize, usize)>), String> {
        let shard_tasks: Vec<_> = shard_subchunks
            .into_iter()
            .map(|(shard_idx, needed_subchunks)| {
                let store = self.store.clone();
                let array = self.array.clone();
                let cache = self.shard_index_cache.clone();
                let index_codecs = self.index_codecs.clone();
                let codec_options = self.codec_options.clone();
                let chunks_per_shard = self.chunks_per_shard;
                let index_encoded_size = self.index_encoded_size;
                let ndim = self.ndim;

                tokio::spawn(async move {
                    let mut shard_indices = vec![0u64; ndim];
                    shard_indices[0] = shard_idx as u64;
                    let store_key = array.chunk_key(&shard_indices);
                    let path = ObjectPath::from(store_key.to_string());

                    // Fetch or cache shard index
                    let shard_index = {
                        let cache_guard = cache.lock().await;
                        if let Some(idx) = cache_guard.get(&shard_idx) {
                            idx.clone()
                        } else {
                            drop(cache_guard);
                            let meta = store
                                .head(&path)
                                .await
                                .map_err(|e| format!("HEAD shard {shard_idx}: {e}"))?;
                            let shard_len = meta.size as u64;
                            let index_start = shard_len - index_encoded_size as u64;
                            let index_bytes = store
                                .get_range(&path, index_start..shard_len)
                                .await
                                .map_err(|e| format!("GET index shard {shard_idx}: {e}"))?;

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
                                .map_err(|e| format!("decode index shard {shard_idx}: {e}"))?;
                            let raw = decoded_index
                                .into_fixed()
                                .map_err(
                                    |e| format!("index into_fixed shard {shard_idx}: {e}")
                                )?;
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

                    // Build byte ranges for needed subchunks
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

                    // ONE get_ranges call per shard
                    let fetched = if byte_ranges.is_empty() {
                        vec![]
                    } else {
                        store.get_ranges(&path, &byte_ranges)
                            .await
                            .map_err(|e| format!("get_ranges shard {shard_idx}: {e}"))?
                    };

                    let compressed: Vec<_> = subchunk_order
                        .into_iter()
                        .zip(fetched)
                        .map(|(sc, data)| (shard_idx, sc, data.to_vec()))
                        .collect();

                    Ok::<_, String>((compressed, fill_subchunks))
                })
            })
            .collect();

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

        Ok((all_compressed, all_fills))
    }

    /// Phase 3: Decode compressed subchunks in parallel using rayon.
    fn decode_subchunks(
        &self,
        compressed: &[(usize, usize, Vec<u8>)],
        fill_subchunks: Vec<(usize, usize)>,
    ) -> Result<HashMap<(usize, usize), Vec<u8>>, String> {
        let decoded_results: Vec<_> = compressed
            .par_iter()
            .map(|(shard_idx, sc, data)| {
                let decoded = self
                    .inner_codecs
                    .decode(
                        Cow::Borrowed(data.as_ref()),
                        &self.subchunk_shape,
                        &self.data_type,
                        &self.fill_value,
                        &self.codec_options,
                    )
                    .map_err(|e| format!("decode subchunk {sc} shard {shard_idx}: {e}"))?;
                let raw = decoded
                    .into_fixed()
                    .map_err(|e| format!("into_fixed subchunk {sc} shard {shard_idx}: {e}"))?;
                Ok::<_, String>(((*shard_idx, *sc), raw.into_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let mut decoded_map: HashMap<(usize, usize), Vec<u8>> =
            decoded_results.into_iter().collect();

        // Handle fill-value subchunks
        let fill_bytes = self.fill_value.as_ne_bytes();
        for (shard_idx, sc) in fill_subchunks {
            let mut buf = vec![0u8; self.chunk_size * self.element_stride];
            for elem_buf in buf.chunks_exact_mut(fill_bytes.len()) {
                elem_buf.copy_from_slice(fill_bytes);
            }
            decoded_map.insert((shard_idx, sc), buf);
        }

        Ok(decoded_map)
    }

    /// Phase 4: Assemble decoded subchunks into a flat output buffer.
    fn assemble_output(
        &self,
        range_refs: &[Vec<SubchunkRef>],
        decoded_map: &HashMap<(usize, usize), Vec<u8>>,
    ) -> Result<(Vec<u8>, Vec<i64>), String> {
        let stride = self.element_stride;

        let total_bytes: usize = range_refs
            .iter()
            .flat_map(|refs| refs.iter())
            .map(|r| (r.elem_end - r.elem_start) * stride)
            .sum();
        let mut flat = Vec::with_capacity(total_bytes);
        let mut lengths = Vec::with_capacity(range_refs.len());

        for refs in range_refs {
            let num_elements: usize = refs.iter().map(|r| r.elem_end - r.elem_start).sum();
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
                let byte_start = r.elem_start * stride;
                let byte_end = r.elem_end * stride;
                flat.extend_from_slice(&decoded[byte_start..byte_end]);
            }
        }

        Ok((flat, lengths))
    }
}

#[pymethods]
impl RustBatchReader {
    #[new]
    fn new(py_zarr_array: &Bound<'_, PyAny>) -> PyResult<Self> {
        // 0. Ensure bitpacking codec is registered before opening any arrays
        ensure_bitpack_codec_registered();

        // 1. Extract obstore from zarr array and convert to Arc<dyn ObjectStore>
        let store_wrapper = py_zarr_array.getattr("store")?;
        let obstore = store_wrapper.getattr("store")?;
        let any_store: AnyObjectStore = obstore.extract()?;
        let store: Arc<dyn ObjectStore> = any_store.into_dyn();

        // 2. Extract the array's path within the store.
        let raw_path: String = py_zarr_array
            .getattr("store_path")
            .and_then(|sp| sp.getattr("path"))
            .and_then(|s| s.extract())
            .unwrap_or_else(|_| String::new());
        let store_path = if raw_path.is_empty() {
            "/".to_string()
        } else if raw_path.starts_with('/') {
            raw_path
        } else {
            format!("/{raw_path}")
        };

        // 3. Build zarrs Array for metadata extraction
        let zarrs_store = Arc::new(AsyncObjectStore::new(Arc::clone(&store)));
        let runtime = Arc::new(
            Runtime::new()
                .map_err(|e| PyRuntimeError::new_err(format!("tokio runtime: {e}")))?,
        );
        let array = runtime
            .block_on(Array::async_open(zarrs_store, &store_path))
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open zarr array: {e}")))?;

        // 4. Extract sharding metadata via shared helper
        let meta = extract_sharding_meta(&array)
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let codec_options = CodecOptions::default();

        Ok(Self {
            array: Arc::new(array),
            store,
            inner_codecs: meta.inner_codecs,
            index_codecs: meta.index_codecs,
            subchunk_shape: meta.subchunk_shape,
            data_type: meta.data_type,
            fill_value: meta.fill_value,
            chunk_size: meta.chunk_size,
            chunks_per_shard: meta.chunks_per_shard,
            element_stride: meta.element_stride,
            ndim: meta.ndim,
            index_encoded_size: meta.index_encoded_size,
            shard_index_cache: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            runtime,
            codec_options,
        })
    }

    /// Read ranges along axis 0 from the sharded array.
    ///
    /// Parameters
    /// ----------
    /// starts, ends : 1-D int64 arrays of axis-0 start/end positions.
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
        if starts_vec.len() != ends_vec.len() {
            return Err(PyRuntimeError::new_err(
                "starts and ends must have the same length",
            ));
        }

        // Phase 1: Map ranges to subchunks (pure computation)
        let (range_refs, shard_subchunks) =
            self.map_ranges_to_subchunks(&starts_vec, &ends_vec);

        // Phases 2-4 run without the GIL
        let runtime = self.runtime.clone();
        let (flat_data, lengths_vec) = py
            .detach(|| -> Result<(Vec<u8>, Vec<i64>), String> {
                // Phase 2: Fetch compressed data from the store
                let (compressed, fills) = runtime.block_on(
                    self.fetch_shard_data(shard_subchunks)
                )?;

                // Phase 3: Decode subchunks in parallel
                let decoded_map = self.decode_subchunks(&compressed, fills)?;

                // Phase 4: Assemble output
                self.assemble_output(&range_refs, &decoded_map)
            })
            .map_err(|e| PyRuntimeError::new_err(e))?;

        let flat_array = PyArray1::from_vec(py, flat_data).into();
        let lengths_array = PyArray1::from_vec(py, lengths_vec).into();
        Ok((flat_array, lengths_array))
    }
}

// ---------------------------------------------------------------------------
// Bitpacking pyo3 exports (for the Python write path)
// ---------------------------------------------------------------------------

/// Encode raw bytes (little-endian uint32) using BP-128 bitpacking.
///
/// Parameters
/// ----------
/// data : bytes
///     Raw little-endian uint32 data (length must be a multiple of 4).
/// transform : str
///     "none" or "delta".
///
/// Returns
/// -------
/// numpy array of encoded bytes.
#[pyfunction]
fn bitpack_encode<'py>(
    py: Python<'py>,
    data: &[u8],
    transform: &str,
) -> PyResult<Py<PyArray1<u8>>> {
    if data.len() % 4 != 0 {
        return Err(PyRuntimeError::new_err(format!(
            "bitpack_encode: input length {} is not a multiple of 4",
            data.len()
        )));
    }
    let t = bitpacking::Transform::from_str(transform)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let values: Vec<u32> = data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let encoded = bitpacking::encode(&values, t);
    Ok(PyArray1::from_vec(py, encoded).into())
}

/// Decode BP-128 bitpacked data back to raw bytes (little-endian uint32).
///
/// Parameters
/// ----------
/// data : bytes
///     Bitpacked encoded data.
///
/// Returns
/// -------
/// numpy array of decoded bytes (little-endian uint32).
#[pyfunction]
fn bitpack_decode<'py>(
    py: Python<'py>,
    data: &[u8],
) -> PyResult<Py<PyArray1<u8>>> {
    let values = bitpacking::decode(data)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let mut out = Vec::with_capacity(values.len() * 4);
    for v in &values {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Ok(PyArray1::from_vec(py, out).into())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBatchReader>()?;
    m.add_function(wrap_pyfunction!(bitpack_encode, m)?)?;
    m.add_function(wrap_pyfunction!(bitpack_decode, m)?)?;
    Ok(())
}
