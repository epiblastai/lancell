"""zarr-python v3 BytesBytesCodec backed by Rust SIMD bitpacking (BP-128).

Usage::

    from lancell.codecs.bitpacking import BitpackingCodec

    group.create_array(
        "indices",
        data=flat_indices,
        chunks=(4096,),
        shards=(65536,),
        compressors=BitpackingCodec(transform="delta"),
    )
"""

from dataclasses import dataclass
from typing import Self

from zarr.abc.codec import BytesBytesCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.registry import register_codec

from lancell._rust import bitpack_decode, bitpack_encode

CODEC_NAME = "lancell.bitpacking"


@dataclass(frozen=True)
class BitpackingCodec(BytesBytesCodec):
    """BP-128 bitpacking codec for uint32 arrays.

    Parameters
    ----------
    transform
        "none" for raw values (counts), "delta" for sorted indices.
    element_size
        Bytes per element. Only 4 (uint32) is supported.
    """

    is_fixed_size = False

    transform: str = "none"
    element_size: int = 4

    def __init__(self, *, transform: str = "none", element_size: int = 4) -> None:
        object.__setattr__(self, "transform", transform)
        object.__setattr__(self, "element_size", element_size)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration = parse_named_configuration(data, CODEC_NAME)
        return cls(**configuration)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": CODEC_NAME,
            "configuration": {
                "transform": self.transform,
                "element_size": self.element_size,
            },
        }

    async def _decode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer:
        encoded = bytes(chunk_bytes.as_numpy_array())
        decoded_array = bitpack_decode(encoded)
        return chunk_spec.prototype.buffer.from_bytes(bytes(decoded_array))

    async def _encode_single(
        self,
        chunk_bytes: Buffer,
        chunk_spec: ArraySpec,
    ) -> Buffer | None:
        raw = bytes(chunk_bytes.as_numpy_array())
        encoded_array = bitpack_encode(raw, self.transform)
        return chunk_spec.prototype.buffer.from_bytes(bytes(encoded_array))

    def compute_encoded_size(self, input_byte_length: int, _chunk_spec: ArraySpec) -> int:
        # Variable output size — cannot compute statically
        raise NotImplementedError


register_codec(CODEC_NAME, BitpackingCodec)
