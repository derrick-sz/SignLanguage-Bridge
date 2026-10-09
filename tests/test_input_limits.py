from io import BytesIO
import zipfile

import numpy as np
import pytest

from slbridge.features import load_sequence, load_landmark_archive


def test_oversized_npy_header_rejected_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "bomb.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(stream, {"descr": "<f4", "fortran_order": False,
                                                     "shape": (2**40, 63)})
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("np.load must not allocate"))
    with pytest.raises(ValueError, match="header"):
        load_sequence(path, "hand63_v1")


def test_valid_legacy_numpy_container_unchanged(tmp_path):
    path = tmp_path / "valid.npy"
    values = np.ones((3, 63), dtype=np.float32)
    np.save(path, values)
    assert np.array_equal(load_sequence(path, "hand63_v1"), values)


def test_truncated_numpy_container_rejected_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "truncated.npy"
    np.save(path, np.ones((3, 63), dtype=np.float32))
    path.write_bytes(path.read_bytes()[:-1])
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("np.load must not allocate"))
    with pytest.raises(ValueError, match="size"):
        load_sequence(path, "hand63_v1")


def test_duplicate_zip_entries_rejected_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "duplicate.npz"
    data = BytesIO()
    np.save(data, np.ones((3, 225), dtype=np.float32))
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(path, "w") as archive:
            for _ in range(4):
                archive.writestr("values.npy", data.getvalue())
    monkeypatch.setattr(np, "load", lambda *a, **kw: pytest.fail("np.load must not allocate"))
    with pytest.raises(ValueError, match="entry"):
        load_landmark_archive(path)


def test_corrupted_zip_deflate_is_an_input_error(tmp_path):
    path = tmp_path / "corrupt.npz"
    np.savez_compressed(path, values=np.ones((3, 225), dtype=np.float32),
                        timestamps=np.arange(3, dtype=np.float64),
                        frame_indices=np.arange(3, dtype=np.int64), metadata=np.array('{}'))
    with zipfile.ZipFile(path) as archive:
        offset = archive.getinfo('values.npy').header_offset
    content = bytearray(path.read_bytes())
    filename_size = int.from_bytes(content[offset + 26:offset + 28], 'little')
    extra_size = int.from_bytes(content[offset + 28:offset + 30], 'little')
    payload_start = offset + 30 + filename_size + extra_size
    content[payload_start:payload_start + 4] = b'\xff' * 4
    path.write_bytes(content)
    with pytest.raises(ValueError, match='Invalid landmark ZIP'):
        load_landmark_archive(path)


def test_corrupted_zip_payload_after_valid_header_is_an_input_error(tmp_path):
    from slbridge.input_limits import check_landmark_file
    path = tmp_path / 'corrupt-body.npz'
    values = np.random.default_rng(4).normal(size=(500, 225)).astype(np.float32)
    np.savez_compressed(path, values=values, timestamps=np.arange(500, dtype=np.float64),
                        frame_indices=np.arange(500, dtype=np.int64), metadata=np.array('{}'))
    with zipfile.ZipFile(path) as archive:
        entry = archive.getinfo('values.npy')
        offset, size = entry.header_offset, entry.compress_size
    content = bytearray(path.read_bytes())
    filename_size = int.from_bytes(content[offset + 26:offset + 28], 'little')
    extra_size = int.from_bytes(content[offset + 28:offset + 30], 'little')
    midpoint = offset + 30 + filename_size + extra_size + size // 2
    content[midpoint:midpoint + 16] = b'\0' * 16
    path.write_bytes(content)
    check_landmark_file(path, 'mediapipe225_rgb_v1')  # Header remains valid.
    with pytest.raises(ValueError, match='Invalid landmark ZIP archive payload'):
        load_landmark_archive(path)
