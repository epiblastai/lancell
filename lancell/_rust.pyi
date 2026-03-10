import numpy as np
import numpy.typing as npt

class RustShardReader:
    def __new__(
        cls,
        bucket: str,
        prefix: str,
        region: str,
        chunk_size: int,
        shard_size: int,
        dtype: str,
    ) -> RustShardReader: ...
    def read_ranges(
        self,
        starts: npt.NDArray[np.int64],
        ends: npt.NDArray[np.int64],
    ) -> tuple[npt.NDArray, npt.NDArray[np.int64]]: ...
