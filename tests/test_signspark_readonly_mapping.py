"""Small real checkpoints guard the pinned, immutable, low-commit load path."""
import gc
import hashlib
import pickle

import numpy as np
import pytest


def checkpoint(tmp_path, torch, state):
    path = tmp_path / 'weights.pt'
    torch.save(state, path)
    return path, {'expected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'expected_bytes': path.stat().st_size}


def test_readonly_mapping_materializes_meta_without_copying_or_changing_fp32(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    state = {'weight': torch.tensor([[2., 3.]], dtype=torch.float32),
             'bias': torch.tensor([4.], dtype=torch.float32),
             'counter': torch.tensor([7, 11], dtype=torch.int64)}
    path, identity = checkpoint(tmp_path, torch, state)
    with torch.device('meta'):
        model = torch.nn.Linear(2, 1)
        model.register_buffer('counter', torch.empty(2, dtype=torch.int64))
    position_ids = torch.arange(3, device='cpu')
    model.register_buffer('position_ids', position_ids, persistent=False)
    model.requires_grad_(False)
    mappings = []
    original_memmap = np.memmap

    def observe_mapping(*args, **kwargs):
        mapping = original_memmap(*args, **kwargs)
        mappings.append(mapping)
        return mapping

    def forbidden_native_load(*args, **kwargs):
        pytest.fail('readonly path fell back to native private/COW torch.load')

    monkeypatch.setattr(np, 'memmap', observe_mapping)
    monkeypatch.setattr(torch, 'load', forbidden_native_load)
    report = load_strict_checkpoint(model, path, **identity, assign=True, readonly_mmap=True)
    assert report['readonly_mmap'] is True and report['mmap'] is True
    assert report['missing_keys'] == report['unexpected_keys'] == []
    assert len(mappings) == 1 and mappings[0].mode == 'r' and not mappings[0].flags.writeable
    assert model.weight.data_ptr() >= mappings[0].ctypes.data
    assert model.weight.data_ptr() + model.weight.numel() * 4 <= mappings[0].ctypes.data + len(mappings[0])
    mappings.clear()
    gc.collect()  # the assigned tensor storages must retain the mapping owner
    assert all(not value.is_meta for value in [*model.parameters(), *model.buffers()])
    assert all(value.device.type == 'cpu' for value in [*model.parameters(), *model.buffers()])
    assert model.position_ids is position_ids
    assert not model.weight.requires_grad
    for name, expected in state.items():
        actual = model.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with torch.inference_mode():
        assert model(torch.tensor([[1., 1.]])).item() == 9.
    assert hashlib.sha256(path.read_bytes()).hexdigest() == identity['expected_sha256']
    assert not torch.cuda.is_initialized()


def test_readonly_mapping_still_rejects_bad_sha_before_deserialization(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path, identity = checkpoint(tmp_path, torch, {'weight': torch.tensor([[2., 3.]])})
    identity['expected_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_strict_checkpoint(None, path, **identity, assign=True, readonly_mmap=True)


@pytest.mark.parametrize('change', ['version', 'interface'])
def test_readonly_mapping_fails_closed_for_unreviewed_torch(tmp_path, monkeypatch, change):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path, identity = checkpoint(tmp_path, torch, torch.nn.Linear(2, 1).state_dict())
    if change == 'version':
        monkeypatch.setattr(torch, '__version__', '2.10.0')
    else:
        monkeypatch.setattr(torch.serialization, '_load', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='PyTorch 2.9.1|interface'):
        load_strict_checkpoint(torch.nn.Linear(2, 1), path, **identity, readonly_mmap=True)


def test_readonly_mapping_keeps_strict_keys(tmp_path):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path, identity = checkpoint(tmp_path, torch, {'weight': torch.tensor([[2., 3.]])})
    with pytest.raises(RuntimeError, match='Missing key'):
        load_strict_checkpoint(torch.nn.Linear(2, 1), path, **identity, assign=True, readonly_mmap=True)


def test_readonly_mapping_rejects_legacy_archive_instead_of_allocating(tmp_path):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path = tmp_path / 'legacy.pt'
    torch.save(torch.nn.Linear(2, 1).state_dict(), path, _use_new_zipfile_serialization=False)
    identity = {'expected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'expected_bytes': path.stat().st_size}
    with pytest.raises(ValueError, match='ZIP'):
        load_strict_checkpoint(torch.nn.Linear(2, 1), path, **identity, readonly_mmap=True)


def test_readonly_mapping_preserves_weights_only_unpickling(tmp_path):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    path, identity = checkpoint(tmp_path, torch, {'weight': torch.tensor([[2., 3.]]),
                                               'unsafe': np.arange(2)})
    with pytest.raises(pickle.UnpicklingError, match='Unsupported global'):
        load_strict_checkpoint(torch.nn.Linear(2, 1), path, **identity, readonly_mmap=True)


def test_readonly_mapping_rejects_foreign_byteorder_before_inplace_byteswap(tmp_path):
    import sys
    import zipfile
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import load_strict_checkpoint
    original, _ = checkpoint(tmp_path, torch, torch.nn.Linear(2, 1).state_dict())
    path = tmp_path / 'foreign.pt'
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(path, 'w') as target:
        for info in source.infolist():
            raw = source.read(info.filename)
            if info.filename.endswith('/byteorder'):
                raw = b'big' if sys.byteorder == 'little' else b'little'
            target.writestr(info, raw)
    identity = {'expected_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'expected_bytes': path.stat().st_size}
    with pytest.raises(ValueError, match='byteorder'):
        load_strict_checkpoint(torch.nn.Linear(2, 1), path, **identity, readonly_mmap=True)
