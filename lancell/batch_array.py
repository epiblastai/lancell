import asyncio
from functools import cached_property

import numpy as np
from zarr.core.array import Array, AsyncArray
from zarr.core.sync import sync

from lancell._rust import RustBatchReader


class BatchAsyncArray(AsyncArray):
    """AsyncArray subclass with batched read methods."""

    @classmethod
    def from_array(cls, array: Array | AsyncArray) -> "BatchAsyncArray":
        """Wrap an existing :class:`zarr.Array` or :class:`zarr.AsyncArray`."""
        if isinstance(array, Array):
            async_array = array._async_array
        else:
            async_array = array
        obj = object.__new__(cls)
        obj.__dict__.update(async_array.__dict__)
        return obj

    @cached_property
    def _rust_reader(self) -> RustBatchReader:
        return RustBatchReader(self)

    @cached_property
    def _native_dtype(self) -> np.dtype:
        return np.dtype(self.metadata.dtype.to_native_dtype())

    async def read_ranges(
        self, starts: np.ndarray, ends: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read element ranges from the sharded array.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of element start/end positions.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = number of elements in range i.
        """
        loop = asyncio.get_running_loop()
        raw_bytes, lengths = await loop.run_in_executor(
            None,
            self._rust_reader.read_ranges,
            starts.astype(np.int64),
            ends.astype(np.int64),
        )
        return np.frombuffer(raw_bytes, dtype=self._native_dtype), lengths


class BatchArray(Array):
    """Array subclass with batched read methods.

    Drop-in replacement for :class:`zarr.Array` that adds :meth:`read_ranges`.

    Create via :meth:`from_array` to wrap an existing :class:`zarr.Array`::

        batch_arr = BatchArray.from_array(zarr.open_array("data.zarr"))
        data, lengths = batch_arr.read_ranges(starts, ends)
    """

    @classmethod
    def from_array(cls, array: Array) -> "BatchArray":
        """Wrap an existing :class:`zarr.Array`."""
        async_array = BatchAsyncArray.from_array(array)
        return cls(async_array)

    def read_ranges(self, starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Read element ranges from the sharded array.

        Parameters
        ----------
        starts, ends : 1-D int64 arrays of element start/end positions.

        Returns
        -------
        (flat_data, lengths) where flat_data is the concatenated result and
        lengths[i] = number of elements in range i.
        """
        return sync(self._async_array.read_ranges(starts, ends))
