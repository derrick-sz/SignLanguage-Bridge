"""Read-only snapshot of explicitly configured, source-bound presentation files."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import struct


class PresentationBundle:
    def __init__(self, path: str | Path, assets: list[dict]):
        root = Path(path).resolve()
        raw = (root / 'manifest.json').read_bytes()
        if len(raw) > 128 * 1024:
            raise ValueError('Presentation manifest too large')
        manifest = json.loads(raw)
        files = manifest.get('files')
        if (manifest.get('schema_version') != 1 or not isinstance(files, dict)
                or not 1 <= len(files) <= 32 or manifest.get('avatar') not in files):
            raise ValueError('Invalid presentation manifest')
        self._resources = {}
        for name, digest in files.items():
            if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', name)
                    or '..' in name or not name.endswith(('.glb', '.json'))
                    or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest)):
                raise ValueError('Invalid presentation resource name or checksum')
            target = (root / name).resolve()
            if target.parent != root or target.stat().st_size > 32 * 1024 * 1024:
                raise ValueError('Invalid presentation resource path or size')
            data = target.read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError('Presentation resource checksum mismatch')
            self._resources[name] = (data, 'model/gltf-binary' if name.endswith('.glb') else 'application/json')
        sources = {item['id']: item['sha256'] for item in assets}
        pilots = manifest.get('pilots', [])
        if not isinstance(pilots, list) or len(pilots) > 20:
            raise ValueError('Invalid presentation pilots')
        ids = set()
        public_pilots=[]
        for pilot in pilots:
            if not isinstance(pilot,dict):
                raise ValueError('Invalid presentation pilot source')
            identity = pilot.get('asset_id')
            digest=pilot.get('source_motion_sha256')
            if (not isinstance(identity,str) or identity not in sources or identity in ids
                    or not isinstance(digest,str) or not re.fullmatch(r'[0-9a-f]{64}',digest)
                    or sources[identity] != digest):
                raise ValueError('Presentation pilot source mismatch')
            ids.add(identity)
            temporal_profile=pilot.get('temporal_profile')
            if temporal_profile is not None and temporal_profile!='source-quaternion-temporal-v1':
                raise ValueError('Unknown presentation temporal profile')
            if pilot.get('export_file') is not None and pilot['export_file'] not in files:
                raise ValueError('Unknown presentation pilot export')
            if pilot.get('export_file') is not None:
                data=self._resources[pilot['export_file']][0]
                try:
                    magic,version,total=struct.unpack_from('<4sII',data)
                    length,kind=struct.unpack_from('<II',data,12)
                    if magic!=b'glTF' or version!=2 or total!=len(data) or kind!=0x4e4f534a or 20+length>len(data):
                        raise ValueError('Invalid GLB header')
                    document=json.loads(data[20:20+length])
                    binding=document['asset']['extras']['slbridge_motion']
                    expected={'asset_id':identity,'source_motion_sha256':digest,
                              'avatar_sha256':files[manifest['avatar']],
                              'processing_version':manifest.get('processing_version')}
                    if temporal_profile is not None:
                        expected['temporal_profile']=temporal_profile
                    if binding!=expected or document['animations'][0]['name']!=identity:
                        raise ValueError('Wrong source or target')
                except (ValueError,KeyError,IndexError,TypeError,struct.error) as exc:
                    raise ValueError('Presentation export binding mismatch') from exc
            public={k:deepcopy(pilot[k]) for k in ('asset_id','source_motion_sha256','export_file','text','temporal_profile') if k in pilot}
            public.update(language_review='unverified',motion_review='unverified',expression_review='unverified')
            public_pilots.append(public)
        profile=manifest.get('rig_profile')
        if profile is not None:
            if (not isinstance(profile,dict) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}',str(profile.get('id','')))
                    or not isinstance(profile.get('bones'),dict) or not profile['bones']):
                raise ValueError('Invalid rig profile')
            for key,bindings in profile['bones'].items():
                if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9]{0,79}',key) or not isinstance(bindings,list)
                        or not 1 <= len(bindings) <= 8 or any(not isinstance(b,str) or not 1 <= len(b) <= 128 for b in bindings)):
                    raise ValueError('Invalid rig profile binding')
            for key,values in profile.get('corrections',{}).items():
                if (key not in profile['bones'] or not isinstance(values,list) or len(values)!=3
                        or any(type(v) not in (int,float) or not -0.7853981633974483 <= v <= 0.7853981633974483 for v in values)):
                    raise ValueError('Invalid rig profile correction')
        self._catalog = {key: deepcopy(manifest.get(key)) for key in
                         ('avatar', 'attribution', 'processing_version', 'pilots', 'rig_profile')}
        self._catalog.update(schema_version=1, language_review='unverified',
                             pilots=public_pilots,
                             avatar_sha256=files[manifest['avatar']],
                             manifest_sha256=hashlib.sha256(raw).hexdigest())

    def catalog(self) -> dict:
        return deepcopy(self._catalog)

    def resource(self, name: str) -> tuple[bytes, str]:
        return self._resources[name]
