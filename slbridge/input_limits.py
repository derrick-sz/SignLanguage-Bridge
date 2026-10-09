"""Check NumPy container headers before allocating uploaded landmark arrays."""
from __future__ import annotations

import math
from pathlib import Path
import zipfile
import zlib


def _header(stream):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError("Unsupported NumPy file version")
    shape, _, dtype = readers[version](stream, max_header_size=10000)
    if dtype.hasobject or any(type(size) is not int or size < 0 for size in shape):
        raise ValueError("Unsafe NumPy array dtype or dimensions")
    byte_count = math.prod(shape) * dtype.itemsize
    return shape, dtype, byte_count, stream.tell()


def check_landmark_file(path: str | Path, representation_id: str) -> None:
    path = Path(path)
    width = 63 if representation_id == "hand63_v1" else 225
    if path.suffix.lower() == ".npy":
        with path.open("rb") as stream:
            shape, dtype, byte_count, offset = _header(stream)
        if (len(shape) != 2 or not 1 <= shape[0] <= 10000 or shape[1] != width
                or dtype.kind != "f" or dtype.itemsize not in {4, 8}):
            raise ValueError("Landmark array header violates the input shape/dtype limit")
        if path.stat().st_size != offset + byte_count:
            raise ValueError("Landmark array size does not match its header")
        return
    if path.suffix.lower() != ".npz":
        raise ValueError("Expected a NumPy landmark file")
    expected = {"values.npy", "timestamps.npy", "frame_indices.npy", "metadata.npy"}
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (len(entries) != 4 or {entry.filename for entry in entries} != expected
                    or any(entry.flag_bits & 1 or entry.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
                           for entry in entries)
                    or sum(entry.file_size for entry in entries) > 64 * 1024 * 1024):
                raise ValueError("Landmark archive violates entry or expanded size limits")
            headers = {}
            for entry in entries:
                with archive.open(entry) as stream:
                    shape, dtype, size, offset = _header(stream)
                if entry.file_size != offset + size:
                    raise ValueError("Landmark archive entry size disagrees with its header")
                headers[entry.filename] = shape, dtype
            shape, dtype = headers["values.npy"]
            if (len(shape) != 2 or not 2 <= shape[0] <= 10000 or shape[1] != width
                    or dtype.kind != "f" or dtype.itemsize != 4):
                raise ValueError("RGB landmark header violates the input shape/dtype limit")
            count = shape[0]
            for name, kind in (("timestamps.npy", "f"), ("frame_indices.npy", "i")):
                shape, dtype = headers[name]
                if shape != (count,) or dtype.kind != kind or dtype.itemsize != 8:
                    raise ValueError("Invalid timestamp/index archive header")
            shape, dtype = headers["metadata.npy"]
            if shape != () or dtype.kind != "U" or dtype.itemsize > 1024 * 1024:
                raise ValueError("Invalid or oversized landmark metadata header")
    except (zipfile.BadZipFile, zlib.error, EOFError) as error:
        raise ValueError("Invalid landmark ZIP archive") from error
