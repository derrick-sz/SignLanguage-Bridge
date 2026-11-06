import importlib
import json
import hashlib

import pytest

from slbridge.animation.registry import _validate_clip
from test_animation_presentation import make_bundle


def test_refined_track_validated_without_inheriting_approval():
    clip={'id':'gesture','text':'你好','fps':30,'frames':2,'poses':[[[0,0,0] for _ in range(55)] for _ in range(2)],
          'refined_poses':[[[0,0,0] for _ in range(55)] for _ in range(2)],'gesture_refinement':{'profile':'soke-gesture-refinement-v1'}}
    _validate_clip(clip)
    clip['refined_poses'].pop()
    with pytest.raises(ValueError,match='refined'):_validate_clip(clip)


def test_refined_track_cannot_use_unknown_profile():
    clip={'id':'gesture','text':'你好','fps':30,'frames':1,'poses':[[[0,0,0] for _ in range(55)]],
          'refined_poses':[[[0,0,0] for _ in range(55)]],'gesture_refinement':{'profile':'unknown'}}
    with pytest.raises(ValueError,match='refined'):_validate_clip(clip)


def test_hundred_asset_presentation_preserves_count_and_unverified_reviews(tmp_path):
    from slbridge.animation.presentation import PresentationBundle
    path=make_bundle(tmp_path);manifest=json.loads((path/'manifest.json').read_text())
    assets=[];pilots=[]
    for i in range(100):
        name=f'asset-{i}.glb';data=f'fixture-{i}'.encode();(path/name).write_bytes(data)
        manifest['files'][name]=hashlib.sha256(data).hexdigest()
        assets.append({'id':f'asset-{i}','sha256':'a'*64})
        pilots.append({'asset_id':f'asset-{i}','source_motion_sha256':'a'*64,'temporal_profile':'soke-gesture-refinement-v1'})
    manifest['pilots']=pilots;(path/'manifest.json').write_text(json.dumps(manifest))
    result=PresentationBundle(path,assets).catalog()
    assert len(result['pilots'])==100
    assert all(p['language_review']=='unverified' for p in result['pilots'])


def test_expanded_presentation_enforces_total_memory_budget(tmp_path,monkeypatch):
    module=importlib.import_module('slbridge.animation.presentation')
    monkeypatch.setattr(module,'MAX_PRESENTATION_BYTES',10)
    with pytest.raises(ValueError,match='budget'):
        module.PresentationBundle(make_bundle(tmp_path),[{'id':'hello','sha256':'a'*64}])
