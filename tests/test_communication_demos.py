import hashlib
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from slbridge.communication_demos import DemoCatalog, mount_demo_routes


def fixture(tmp_path):
    media=tmp_path/'media';media.mkdir()
    payload=b'\x00\x00\x00\x18ftypisom'+b'video fixture'*20
    (media/'intro.mp4').write_bytes(payload)
    item=dict(id='intro',title='日常介绍',annotation='你是哪里人？',kind='training_replay',
              source_split='train',sample_id='S000011_P0000_T00',media='media/intro.mp4',
              sha256=hashlib.sha256(payload).hexdigest(),fps=30.,frames=37,width=512,height=512,
              note='已测训练样本，模仿效果待验证。')
    config=tmp_path/'catalog.json'; config.write_text(json.dumps(dict(schema_version=1,items=[item])),encoding='utf-8')
    return config,item,payload


def app(catalog=None):
    result=FastAPI();mount_demo_routes(result,catalog);return result


def test_catalog_is_opt_in_and_only_exposes_explicit_media(tmp_path):
    with TestClient(app()) as client:
        assert client.get('/v1/communication/demos').json()['enabled'] is False
        assert client.get('/v1/communication/demos/intro/video').status_code==404
    config,item,payload=fixture(tmp_path);catalog=DemoCatalog(config)
    with TestClient(app(catalog)) as client:
        response=client.get('/v1/communication/demos');assert response.status_code==200
        data=response.json();public=data['items'][0]
        assert data['catalog_sha256']==hashlib.sha256(config.read_bytes()).hexdigest()
        assert public['annotation']=='你是哪里人？' and public['source_split']=='train'
        assert 'media' not in public and str(tmp_path) not in response.text
        media=client.get(public['media_url']);assert media.content==payload
        assert media.headers['content-type']=='video/mp4'
        assert media.headers['x-content-type-options']=='nosniff'
        partial=client.get(public['media_url'],headers={'Range':'bytes=0-15'})
        assert partial.status_code==206 and partial.content==payload[:16]
        assert client.get('/v1/communication/demos/unknown/video').status_code==404
        assert client.get('/v1/communication/demos/intro/video?sha256='+'0'*64).status_code==409


def test_video_and_catalog_changes_are_not_silently_served_under_old_identity(tmp_path):
    config,item,_=fixture(tmp_path);catalog=DemoCatalog(config)
    with TestClient(app(catalog)) as client:
        (tmp_path/item['media']).write_bytes(b'changed')
        assert client.get('/v1/communication/demos/intro/video').status_code==409
    config.write_text('{}',encoding='utf-8')
    with TestClient(app(catalog)) as client:
        assert client.get('/v1/communication/demos').status_code==409


@pytest.mark.parametrize('field,value',[
    ('media','../private.mp4'),('media','C:/private.mp4'),('source_split','test'),
    ('kind','camera'),('fps',float('nan')),('frames',0),('width',4096),('sha256','bad'),
])
def test_malformed_or_out_of_scope_catalog_fails_closed(tmp_path,field,value):
    config,item,_=fixture(tmp_path);item[field]=value
    config.write_text(json.dumps(dict(schema_version=1,items=[item])),encoding='utf-8')
    with pytest.raises(ValueError):DemoCatalog(config)


def test_duplicate_ids_and_unknown_manifest_fields_fail_closed(tmp_path):
    config,item,_=fixture(tmp_path)
    for manifest in [dict(schema_version=1,items=[item,item]),dict(schema_version=1,items=[item],file_root='C:/')]:
        config.write_text(json.dumps(manifest),encoding='utf-8')
        with pytest.raises(ValueError):DemoCatalog(config)
