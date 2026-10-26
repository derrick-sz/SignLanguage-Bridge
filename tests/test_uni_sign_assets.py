from pathlib import Path
import hashlib

import pytest

from slbridge.uni_sign_assets import (
    MT5_FILES,
    UNISIGN_CHECKPOINT_BYTES,
    UNISIGN_CHECKPOINT_SHA256,
    build_bundle_manifest,
    require_disk_space,
    sha256_file,
    verify_checkpoint,
    verify_file,
    verify_mt5_dir,
    write_json_atomic,
)


def test_verify_file_accepts_exact_size_and_hash(tmp_path):
    target = tmp_path / 'weight.pth'
    payload = b'pinned-weight'
    target.write_bytes(payload)
    result = verify_file(target, expected_bytes=len(payload), expected_sha256=hashlib.sha256(payload).hexdigest())
    assert result.ok and result.bytes == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()


def test_verify_file_rejects_missing_size_and_hash(tmp_path):
    missing = verify_file(tmp_path / 'missing.pth', expected_bytes=1, expected_sha256='0' * 64)
    assert not missing.ok and missing.reason == 'missing'
    target = tmp_path / 'weight.pth'
    target.write_bytes(b'abc')
    assert verify_file(target, expected_bytes=2).reason.startswith('size_mismatch')
    assert verify_file(target, expected_bytes=3, expected_sha256='0' * 64).reason == 'sha256_mismatch'


def test_pinned_checkpoint_constants_are_consistent():
    assert len(UNISIGN_CHECKPOINT_SHA256) == 64
    assert UNISIGN_CHECKPOINT_BYTES == 1_186_925_112
    assert MT5_FILES['pytorch_model.bin'][0] == 2_329_735_129


def test_mt5_verification_reports_each_required_file(tmp_path):
    checks = verify_mt5_dir(tmp_path)
    assert set(checks) == set(MT5_FILES)
    assert not any(check.ok for check in checks.values())


def test_bundle_manifest_uses_relative_assets(tmp_path):
    root = tmp_path / 'uni_sign'
    manifest = build_bundle_manifest(root)
    assert manifest['backend'] == 'uni_sign_pose_slt'
    assert manifest['task_type'] == 'text'
    assert manifest['checkpoint'] == '../csl_daily_pose_only_slt.pth'
    assert manifest['mt5_dir'] == '../mt5-base'
    assert manifest['source_dir'] == '../../uni_sign_source'
    assert manifest['license']['model'] == 'CC-BY-NC-4.0'
    assert manifest['runtime']['pose_normalization'] == 'isotropic_short_side'


def test_atomic_json_roundtrip_and_disk_guard(tmp_path):
    path = tmp_path / 'nested' / 'manifest.json'
    write_json_atomic(path, {'中文': '值'})
    assert '"中文": "值"' in path.read_text(encoding='utf-8')
    info = require_disk_space(tmp_path, 0, reserve_bytes=0)
    assert info['free_bytes'] >= 0
    with pytest.raises(OSError):
        require_disk_space(tmp_path, 10**30, reserve_bytes=0)
