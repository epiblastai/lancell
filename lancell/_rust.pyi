import numpy as np
import numpy.typing as npt
import zarr

class RustBatchReader:
    def __new__(cls, zarr_array: zarr.Array) -> RustBatchReader: ...
    def read_ranges(
        self,
        starts: npt.NDArray[np.int64],
        ends: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray, npt.NDArray[np.int64]]: ...
