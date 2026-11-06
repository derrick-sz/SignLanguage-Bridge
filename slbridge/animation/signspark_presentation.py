"""Publish an avatar-only presentation for independently generated candidates.

No pretrained weights, baked motions, service state, or source files are changed.
The pinned source manifest authenticates the rig profile and attribution as well
as the already calibrated avatar; its old pilots are never copied or loaded.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .corpus_store import JAW_CANDIDATE
from .presentation import PresentationBundle
from .registry import _validate_clip
from .signspark_codec import CODEC_VERSION

DEFAULT_SOURCE_PRESENTATION = Path('G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/presentation')
SOURCE_MANIFEST_SHA256 = 'a9fa82f145861c731d6aadc63957df549e7008fb701cb282a5f8dcd7143d80c8'
PRESENTATION_VERSION = 'signspark-avatar-jaw-candidate-v1'
AVATAR_NAME = 'avatar-clean.glb'


def _bounded_bytes(path, limit):
    with path.open('rb') as stream:
        data=stream.read(limit+1)
    if not data or len(data)>limit:
        raise ValueError('Presentation input exceeds size limit')
    return data


def _source(source_presentation):
    root=Path(source_presentation).resolve()
    manifest_path=(root/'manifest.json').resolve()
    avatar_path=(root/AVATAR_NAME).resolve()
    if manifest_path.parent!=root or avatar_path.parent!=root:
        raise ValueError('Presentation source path leaves its directory')
    raw=_bounded_bytes(manifest_path,128*1024)
    if hashlib.sha256(raw).hexdigest()!=SOURCE_MANIFEST_SHA256:
        raise ValueError('Presentation source manifest SHA256 mismatch')
    manifest=json.loads(raw)
    expected=JAW_CANDIDATE['target_avatar_sha256']
    if (manifest.get('schema_version')!=1 or manifest.get('avatar')!=AVATAR_NAME
            or manifest.get('files',{}).get(AVATAR_NAME)!=expected
            or not isinstance(manifest.get('rig_profile'),dict)
            or not isinstance(manifest.get('attribution'),str) or not manifest['attribution']):
        raise ValueError('Invalid calibrated presentation source')
    avatar=_bounded_bytes(avatar_path,32*1024*1024)
    if hashlib.sha256(avatar).hexdigest()!=expected:
        raise ValueError('Calibrated avatar SHA256 mismatch')
    return manifest,avatar


def verify_presentation_inputs(*, smplx_model, source_presentation=DEFAULT_SOURCE_PRESENTATION):
    """Authenticate the target and source template before expensive inference."""
    _source(source_presentation)
    expected=JAW_CANDIDATE['source_template_sha256']
    digest=hashlib.sha256();size=0
    with Path(smplx_model).open('rb') as stream:
        while block:=stream.read(1024*1024):
            size+=len(block)
            if size>128*1024*1024:
                raise ValueError('SMPL-X template exceeds size limit')
            digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('SMPL-X template SHA256 mismatch')
    return {'source_template_sha256':expected,
            'target_avatar_sha256':JAW_CANDIDATE['target_avatar_sha256'],
            'source_presentation_manifest_sha256':SOURCE_MANIFEST_SHA256}


def opt_in_generated_jaw(clip, *, smplx_model, source_presentation=DEFAULT_SOURCE_PRESENTATION):
    """Return a detached, explicit jaw candidate after checking actual file bytes.

    Call BEFORE writing the final clip/archive and computing their digests. This
    does not decode, modify or smooth any pose/expression value and does not add
    SOKE hand means. SignSparK's codec must already have put jaw in joint 22.
    """
    _validate_clip(clip)
    unmapped=clip.get('unmapped_target_channels')
    if (clip.get('source_kind')!='pretrained_text_conditioned_generation'
            or clip.get('generation_codec')!=CODEC_VERSION
            or type(clip.get('expression_channels_retained')) is not int
            or clip['expression_channels_retained']!=50
            or not isinstance(unmapped,list) or any(not isinstance(v,str) for v in unmapped)
            or 'expression' not in unmapped):
        raise ValueError('Unsupported generated jaw/expression protocol')
    expected=JAW_CANDIDATE['source_template_sha256']
    if clip.get('template_sha256',expected)!=expected:
        raise ValueError('Generated clip template SHA256 mismatch')
    if clip.get('nonmanual_candidate',JAW_CANDIDATE)!=JAW_CANDIDATE:
        raise ValueError('Generated clip already has another jaw protocol')
    verify_presentation_inputs(smplx_model=smplx_model,source_presentation=source_presentation)
    result=deepcopy(clip)
    result.update(template_sha256=expected,nonmanual_candidate=deepcopy(JAW_CANDIDATE),
                  unmapped_target_channels=list(dict.fromkeys('jaw_swing' if v=='jaw' else v for v in unmapped)),
                  presentation_status='research_candidate',export_supported=False,
                  experimental=True,human_approved=False,training_ready=False,
                  language_review='unverified',motion_review='unverified',expression_review='unverified')
    return result


def build_avatar_presentation(output, *, assets=(), source_presentation=DEFAULT_SOURCE_PRESENTATION):
    """Atomically create a NEW directory with only manifest.json and the avatar.

    ``assets`` are the NEW MotionRegistry.catalog()['assets'], after the final
    candidate archive is written. Only id/sha256/text are used for optional
    pilot provenance; old source pilots and any export_file field are ignored.
    Returns the new manifest. Existing outputs are never replaced.
    """
    source,avatar=_source(source_presentation)
    output=Path(output).resolve()
    if output.exists():
        raise ValueError('Presentation output must be a new directory')
    assets=deepcopy(list(assets))
    pilots=[]
    for item in assets:
        pilot={'asset_id':item['id'],'source_motion_sha256':item['sha256']}
        if 'text' in item:pilot['text']=item['text']
        pilot.update(language_review='unverified',motion_review='unverified',expression_review='unverified')
        pilots.append(pilot)
    manifest={'schema_version':1,'avatar':AVATAR_NAME,
              'files':{AVATAR_NAME:JAW_CANDIDATE['target_avatar_sha256']},
              'attribution':deepcopy(source['attribution']),'rig_profile':deepcopy(source['rig_profile']),
              'processing_version':PRESENTATION_VERSION,'source_presentation_manifest_sha256':SOURCE_MANIFEST_SHA256,
              'pilots':pilots,'status':'research_candidate','export_supported':False,
              'language_review':'unverified','motion_review':'unverified','expression_review':'unverified'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.'+output.name+'-',dir=output.parent) as name:
        staging=Path(name)
        (staging/AVATAR_NAME).write_bytes(avatar)
        (staging/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
        PresentationBundle(staging,assets)
        if output.exists():
            raise ValueError('Presentation output appeared during validation')
        os.rename(staging,output)
    return deepcopy(manifest)
