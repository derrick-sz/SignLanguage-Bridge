"""Authenticated receipts bind bytes, not the original workstation's paths."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def worker():
    script=Path(__file__).resolve().parents[1]/'scripts/animation/generate_custom_signspark.py'
    spec=importlib.util.spec_from_file_location('signspark_portability',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def write_receipt(path,value):
    raw=json.dumps(value).encode();path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def text_fixture(worker,tmp_path,monkeypatch):
    root=tmp_path/'moved-assets';(root/'mclip').mkdir(parents=True)
    payload=b'verified text config';(root/'mclip/config.json').write_bytes(payload)
    entry={'repo':'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus','revision':'15c3aee9b9e96fd7de1c1d488810a604bab98e78',
           'path':'config.json','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),
           'local_path':'Z:/unavailable-old-machine/config.json'}
    manifest={'schema_version':1,'files':[entry]}
    digest=write_receipt(root/'manifest.json',manifest)
    monkeypatch.setattr(worker,'TEXT_ASSETS_MANIFEST_SHA256',digest)
    return root,manifest


def test_legacy_text_receipt_verifies_relocated_bytes_not_local_path(worker,text_fixture):
    root,_=text_fixture
    assert worker.verify_text_assets(root)==worker.TEXT_ASSETS_MANIFEST_SHA256
    (root/'mclip/config.json').write_bytes(b'wrong text bytes!!!!')
    with pytest.raises(ValueError,match='SHA|length'):worker.verify_text_assets(root)


@pytest.mark.parametrize('patch',[{'repo':'unknown/repository'},{'revision':'not-pinned'},
    {'path':'../outside'},{'path':'/absolute'},{'path':'x\\config.json'},{'path':'C:/absolute'}])
def test_text_mapping_rejects_unknown_repository_revision_and_paths(worker,text_fixture,monkeypatch,patch):
    root,manifest=text_fixture;manifest['files'][0].update(patch)
    digest=write_receipt(root/'manifest.json',manifest)
    monkeypatch.setattr(worker,'TEXT_ASSETS_MANIFEST_SHA256',digest)
    with pytest.raises(ValueError):worker.verify_text_assets(root)


def test_repo_compact_text_receipt_is_used_only_when_local_receipt_absent(worker,text_fixture,tmp_path,monkeypatch):
    root,manifest=text_fixture
    manifest['files'][0].pop('local_path')
    compact=tmp_path/'compact.json';digest=write_receipt(compact,manifest)
    assert hasattr(worker,'DEFAULT_TEXT_ASSETS_MANIFEST')
    monkeypatch.setattr(worker,'DEFAULT_TEXT_ASSETS_MANIFEST',compact)
    monkeypatch.setattr(worker,'COMPACT_TEXT_ASSETS_MANIFEST_SHA256',digest)
    (root/'manifest.json').unlink()
    assert worker.verify_text_assets(root)==digest
    (root/'manifest.json').write_text('{"files":[]}',encoding='utf8')
    with pytest.raises(ValueError,match='SHA256'):worker.verify_text_assets(root)


def test_committed_compact_receipts_are_complete_small_and_pinned(worker):
    assert hasattr(worker,'DEFAULT_SOURCE_MANIFEST') and hasattr(worker,'DEFAULT_TEXT_ASSETS_MANIFEST')
    for path,digest,count in [(worker.DEFAULT_SOURCE_MANIFEST,worker.COMPACT_SOURCE_MANIFEST_SHA256,40),
                             (worker.DEFAULT_TEXT_ASSETS_MANIFEST,worker.COMPACT_TEXT_ASSETS_MANIFEST_SHA256,9)]:
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest()==digest
        assert len(raw)<16*1024 and b'content_base64' not in raw and b'local_path' not in raw
        assert len(json.loads(raw)['files'])==count


def test_compact_source_receipt_checks_blob_and_size(worker,tmp_path,monkeypatch):
    raw=b'# authenticated source\n';(tmp_path/'fixture.py').write_bytes(raw)
    manifest={'repository':'JianHe0628/SignSparK','commit':worker.SOURCE_COMMIT,'files':[{
        'path':'fixture.py','size':len(raw),'blob_sha':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()}]}
    receipt=tmp_path/'source.json';digest=write_receipt(receipt,manifest)
    assert hasattr(worker,'COMPACT_SOURCE_MANIFEST_SHA256')
    monkeypatch.setattr(worker,'COMPACT_SOURCE_MANIFEST_SHA256',digest)
    assert worker.verify_source(tmp_path,receipt)==digest
    (tmp_path/'fixture.py').write_bytes(b'# replaced source\n')
    with pytest.raises(ValueError,match='changed'):worker.verify_source(tmp_path,receipt)


@pytest.mark.parametrize('receipt_mode',['legacy','compact','fallback'])
def test_real_nine_assets_move_with_same_bytes_and_corruption_is_rejected(worker,tmp_path,receipt_mode):
    if os.environ.get('SIGNBRIDGE_REAL_SIGNSPARK_ASSETS_TEST')!='1':pytest.skip('explicit real asset opt-in')
    old=Path('G:/Archive/Zenodo/SignSparK/text-assets')
    manifest=json.loads(worker.DEFAULT_TEXT_ASSETS_MANIFEST.read_bytes())
    moved=tmp_path/'different-drive-root';moved.mkdir()
    repos={'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus':'mclip','FacebookAI/xlm-roberta-large':'xlm-roberta-large'}
    for item in manifest['files']:
        relative=Path(repos[item['repo']])/item['path'];destination=moved/relative
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old/relative,destination)
    if receipt_mode=='legacy':shutil.copy2(old/'manifest.json',moved/'manifest.json')
    elif receipt_mode=='compact':shutil.copy2(worker.DEFAULT_TEXT_ASSETS_MANIFEST,moved/'manifest.json')
    expected=worker.TEXT_ASSETS_MANIFEST_SHA256 if receipt_mode=='legacy' else worker.COMPACT_TEXT_ASSETS_MANIFEST_SHA256
    assert worker.verify_text_assets(moved)==expected
    config=moved/'mclip/config.json';data=config.read_bytes();config.write_bytes(bytes([data[0]^1])+data[1:])
    with pytest.raises(ValueError,match='SHA256'):worker.verify_text_assets(moved)


def test_real_source_all_forty_blobs_accept_both_pinned_receipts(worker):
    if os.environ.get('SIGNBRIDGE_REAL_SIGNSPARK_ASSETS_TEST')!='1':pytest.skip('explicit real asset opt-in')
    source=Path('G:/Archive/Zenodo/SignSparK/source')
    assert worker.verify_source(source,worker.DEFAULT_SOURCE_MANIFEST)==worker.COMPACT_SOURCE_MANIFEST_SHA256
    assert worker.verify_source(source,source.parent/'source-bundle.json')==worker.SOURCE_BUNDLE_SHA256
