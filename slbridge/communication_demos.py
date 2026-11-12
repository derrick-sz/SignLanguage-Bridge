"""Opt-in, fixed training media for local demos; no recognition dependencies."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from fastapi import HTTPException, Query
from fastapi.responses import FileResponse

_ID=re.compile(r'[a-z0-9][a-z0-9_-]{0,63}')
_SHA=re.compile(r'[0-9a-f]{64}')
_MEDIA=re.compile(r'media/[A-Za-z0-9_-]{1,64}\.mp4')
_FIELDS={'id','title','annotation','kind','source_split','sample_id','media','sha256',
         'fps','frames','width','height','note'}
_MAX_MANIFEST=64*1024
_MAX_VIDEO=8*1024*1024
DATASET_URL='https://ustc-slr.github.io/datasets/2021_csl_daily/'


def _manifest_bytes(path):
    with path.open('rb') as stream:
        raw=stream.read(_MAX_MANIFEST+1)
    if len(raw)>_MAX_MANIFEST: raise ValueError('Demo catalog is too large')
    return raw


def _digest(path):
    size=path.stat().st_size
    if not 12<=size<=_MAX_VIDEO: raise ValueError('Invalid demo video size')
    digest=hashlib.sha256();count=0
    with path.open('rb') as stream:
        header=stream.read(12)
        if header[4:8]!=b'ftyp': raise ValueError('Demo media must be MP4')
        digest.update(header);count+=len(header)
        while chunk:=stream.read(128*1024):
            count+=len(chunk)
            if count>_MAX_VIDEO: raise ValueError('Demo video exceeds byte limit')
            digest.update(chunk)
    if count!=size: raise ValueError('Demo video changed while reading')
    return digest.hexdigest()


class DemoCatalog:
    def __init__(self, path):
        self.path=Path(path).resolve();self.root=self.path.parent
        raw=_manifest_bytes(self.path);data=json.loads(raw)
        if (not isinstance(data,dict) or set(data)!={'schema_version','items'} or
                type(data['schema_version']) is not int or data['schema_version']!=1 or
                not isinstance(data['items'],list) or not 1<=len(data['items'])<=16):
            raise ValueError('Invalid demo catalog')
        self.sha256=hashlib.sha256(raw).hexdigest();self._items={}
        for row in data['items']:
            if not isinstance(row,dict) or set(row)!=_FIELDS: raise ValueError('Invalid demo item fields')
            if any(not isinstance(row[k],str) or not row[k].strip() or len(row[k])>limit for k,limit in
                   [('id',64),('title',80),('annotation',200),('sample_id',64),('note',300),('media',80),('sha256',64)]):
                raise ValueError('Invalid demo item text')
            if (not _ID.fullmatch(row['id']) or row['id'] in self._items or
                    not _SHA.fullmatch(row['sha256']) or not _MEDIA.fullmatch(row['media']) or
                    not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',row['sample_id']) or
                    not isinstance(row['kind'],str) or row['kind'] not in {'training_replay','diagnostic_replay'} or row['source_split']!='train'):
                raise ValueError('Invalid demo identity or scope')
            if (type(row['fps']) not in {int,float} or not math.isfinite(row['fps']) or not 10<=row['fps']<=30 or
                    type(row['frames']) is not int or not 1<=row['frames']<=450 or row['frames']/row['fps']>15 or
                    any(type(row[k]) is not int or not 16<=row[k]<=512 for k in ('width','height'))):
                raise ValueError('Invalid demo video metadata')
            self._items[row['id']]=dict(row)
            self.video(row['id'])

    def _verify_manifest(self):
        if hashlib.sha256(_manifest_bytes(self.path)).hexdigest()!=self.sha256:
            raise ValueError('Demo catalog changed; restart the configured product')

    def video(self, identifier, expected_sha256=None):
        self._verify_manifest()
        row=self._items[identifier]
        path=(self.root/row['media']).resolve()
        if not path.is_relative_to(self.root) or not path.is_file(): raise ValueError('Demo media is unavailable')
        if expected_sha256 is not None and expected_sha256!=row['sha256']: raise ValueError('Stale demo URL')
        if _digest(path)!=row['sha256']: raise ValueError('Demo media changed')
        return path

    def public(self):
        self._verify_manifest()
        items=[]
        for row in self._items.values():
            item={k:v for k,v in row.items() if k!='media'}
            item['media_url']=f"/v1/communication/demos/{row['id']}/video?sha256={row['sha256']}"
            item['duration_seconds']=row['frames']/row['fps'];items.append(item)
        return dict(enabled=True,schema_version=1,catalog_sha256=self.sha256,
                    dataset='CSL-Daily',dataset_url=DATASET_URL,sign_language='csl',
                    scope='training_replay_not_camera_or_imitation_validation',items=items)


def mount_demo_routes(app, catalog=None):
    @app.get('/v1/communication/demos')
    def demos():
        if catalog is None: return dict(enabled=False,schema_version=1,items=[])
        try: return catalog.public()
        except (OSError,ValueError) as error:
            raise HTTPException(409,'示范目录已变化，请重新加载对应服务。') from error

    @app.get('/v1/communication/demos/{identifier}/video')
    def video(identifier: str, sha256: str | None=Query(None,pattern=r'^[0-9a-f]{64}$')):
        if catalog is None: raise HTTPException(404,'Unknown demo')
        try: path=catalog.video(identifier,sha256)
        except KeyError as error: raise HTTPException(404,'Unknown demo') from error
        except (OSError,ValueError) as error:
            raise HTTPException(409,'这条示范已变化或暂不可用，请重新选择。') from error
        return FileResponse(path,media_type='video/mp4',headers={
            'Cache-Control':'no-cache','X-Content-Type-Options':'nosniff'})
