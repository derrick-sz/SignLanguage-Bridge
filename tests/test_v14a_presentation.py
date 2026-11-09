"""Builder contract tests using a synthetic catalog, not MotionRegistry end-to-end.

The optional frozen-GLB check requires ANIMATION_V14A_AVATAR_GLB. Unit tests
use fixture bytes with a test-only expected digest; production identity is unchanged.
"""
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import struct

import pytest
from slbridge.animation import godette_v14a_presentation as builder


def sources():
    base = json.loads(files('slbridge.animation').joinpath('godette_v13_manifest.json').read_text(encoding='utf8'))
    return base, {'archive_sha256': 'a' * 64, 'assets': [
        {'id': p['asset_id'], 'sha256': p['source_motion_sha256']} for p in base['pilots']
    ]}


@pytest.fixture
def fixture_avatar(tmp_path, monkeypatch):
    document = json.dumps({'asset': {'version': '2.0'}}).encode()
    document += b' ' * (-len(document) % 4)
    data = struct.pack('<4sIIII', b'glTF', 2, 20 + len(document), len(document), 0x4e4f534a) + document
    avatar = tmp_path / 'fixture.glb'
    avatar.write_bytes(data)
    monkeypatch.setattr(builder, 'AVATAR_SHA256', hashlib.sha256(data).hexdigest())
    return avatar


def assert_bundle(output, avatar, base, catalog):
    manifest = builder.build_godette_v14a_presentation(output, avatar=avatar, catalog=catalog)
    assert manifest['files'] == {'godette-reference-v14a.glb': hashlib.sha256(avatar.read_bytes()).hexdigest()}
    assert manifest['rig_profile'] == base['rig_profile']
    assert manifest['pilots'] == [{**p, 'language_review': 'unverified', 'motion_review': 'unverified', 'expression_review': 'unverified'} for p in base['pilots']]
    assert manifest['automatic_nonmanual_supported'] is False
    assert manifest['export_supported'] is False
    assert manifest['status'] == 'research_candidate'
    assert (output / 'ATTRIBUTION.md').is_file()
    assert (output / builder.AVATAR_NAME).read_bytes() == avatar.read_bytes()
    return manifest


def test_synthetic_catalog_bundle_preserves_all_six_source_bindings(tmp_path, fixture_avatar):
    base, catalog = sources()
    assert len(base['pilots']) == 6
    assert_bundle(tmp_path / 'bundle', fixture_avatar, base, catalog)


@pytest.mark.parametrize('mutation, message', [
    ('missing', 'pilot source mismatch'), ('duplicate', 'duplicate identities'),
    ('wrong_source', 'pilot source mismatch'), ('bad_archive', 'Invalid source archive catalog'),
    ('bad_source', 'Invalid presentation source catalog'),
])
def test_invalid_catalog_rejected_without_output(tmp_path, fixture_avatar, mutation, message):
    _, catalog = sources()
    if mutation == 'missing': catalog['assets'].pop()
    elif mutation == 'duplicate': catalog['assets'].append(dict(catalog['assets'][0]))
    elif mutation == 'wrong_source': catalog['assets'][0]['sha256'] = 'f' * 64
    elif mutation == 'bad_archive': catalog['archive_sha256'] = 'invalid'
    elif mutation == 'bad_source': catalog['assets'][0]['sha256'] = 'invalid'
    output = tmp_path / 'bundle'
    with pytest.raises(ValueError, match=message):
        builder.build_godette_v14a_presentation(output, avatar=fixture_avatar, catalog=catalog)
    assert not output.exists()


def test_wrong_avatar_identity_rejected(tmp_path, fixture_avatar):
    _, catalog = sources()
    fixture_avatar.write_bytes(b'wrong identity')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        builder.build_godette_v14a_presentation(tmp_path / 'bundle', avatar=fixture_avatar, catalog=catalog)
    assert not (tmp_path / 'bundle').exists()


@pytest.mark.parametrize('kind', ['directory', 'symlink', 'dangling_symlink'])
def test_existing_output_refused(tmp_path, fixture_avatar, kind):
    _, catalog = sources()
    output = tmp_path / 'bundle'
    if kind == 'directory': output.mkdir()
    else:
        target = tmp_path / 'target'
        if kind == 'symlink': target.mkdir()
        try: output.symlink_to(target, target_is_directory=True)
        except (OSError, NotImplementedError): pytest.skip('symlink creation unavailable on this platform')
    with pytest.raises(ValueError, match='new directory'):
        builder.build_godette_v14a_presentation(output, avatar=fixture_avatar, catalog=catalog)
    if kind == 'directory': assert list(output.iterdir()) == []
    else: assert output.is_symlink()


def test_concurrently_created_empty_output_is_not_replaced(tmp_path, fixture_avatar, monkeypatch):
    _, catalog = sources()
    output = tmp_path / 'bundle'
    original_mkdir = Path.mkdir
    raced = []
    def racing_mkdir(self, *args, **kwargs):
        if self == output and not raced:
            original_mkdir(self)
            raced.append(True)
        return original_mkdir(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', racing_mkdir)
    with pytest.raises(ValueError, match='new directory'):
        builder.build_godette_v14a_presentation(output, avatar=fixture_avatar, catalog=catalog)
    assert raced == [True]
    assert output.is_dir() and list(output.iterdir()) == []


def test_optional_real_frozen_glb_with_synthetic_catalog(tmp_path):
    configured = os.environ.get('ANIMATION_V14A_AVATAR_GLB')
    if not configured: pytest.skip('set ANIMATION_V14A_AVATAR_GLB for frozen-byte integration check')
    avatar = Path(configured)
    assert avatar.is_file(), 'configured ANIMATION_V14A_AVATAR_GLB must exist'
    assert builder.AVATAR_SHA256 == '535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e'
    base, catalog = sources()
    assert_bundle(tmp_path / 'bundle', avatar, base, catalog)
