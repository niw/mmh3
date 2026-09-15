"""Safetensors files and ConvRot helpers shared by the model tools."""

import json
import os
import struct

import numpy as np

CONVROT_GROUP = 256
DTYPES = {
    "BF16": np.uint16,
    "F16": np.float16,
    "F32": np.float32,
    "I8": np.int8,
    "U8": np.uint8,
}


class SafeTensors:
    """A safetensors file whose tensors are read one at a time.

    NOTE: The tensors are read into memory rather than mapped, and their pages are dropped
    from the page cache after each read, because a pass over checkpoints of tens of GB would
    otherwise fill the memory with mapped or cached pages."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            self.header = json.loads(file.read(length))
        self.header.pop("__metadata__", None)
        self.base = 8 + length

    def __contains__(self, name):
        return name in self.header

    def dtype(self, name):
        return self.header[name]["dtype"]

    def shape(self, name):
        return self.header[name]["shape"]

    def read(self, name):
        """The tensor as stored, BF16 as its uint16 bits."""
        info = self.header[name]
        start, end = info["data_offsets"]
        dtype = np.dtype(DTYPES[info["dtype"]])
        values = np.empty((end - start) // dtype.itemsize, dtype=dtype)
        with open(self.path, "rb") as file:
            file.seek(self.base + start)
            file.readinto(memoryview(values).cast("B"))
            os.posix_fadvise(
                file.fileno(), self.base + start, end - start, os.POSIX_FADV_DONTNEED
            )
        return values.reshape(info["shape"])

    def load(self, name):
        """The tensor as float32, or as stored for integer dtypes."""
        info = self.header[name]
        values = self.read(name)
        if info["dtype"] == "BF16":
            return (values.astype(np.uint32) << 16).view(np.float32)
        if info["dtype"] in ("F16", "F32"):
            return values.astype(np.float32)
        return values


def write_safetensors(path, tensors, metadata):
    """Writes {name: (dtype, shape, bytes)} with string metadata."""
    header = {"__metadata__": metadata}
    offset = 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as output:
        output.write(struct.pack("<Q", len(encoded)))
        output.write(encoded)
        for _, _, data in tensors.values():
            output.write(data)


def hadamard():
    """The normalized regular Hadamard matrix of ConvRot's 256-column groups."""
    kernel = np.array(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=np.float64
    )
    matrix = kernel / 2
    for _ in range(3):
        matrix = np.kron(matrix, kernel / 2)
    return matrix
