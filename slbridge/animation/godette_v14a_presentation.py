"""Publish the frozen Godette v14a avatar without replacing any motion archive.

This is independent of the legacy, source/target-bound jaw calibration and of
SignSparK generation. It never imports that calibration, copies baked exports,
or changes the source catalog, corpus, motion data or review status.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import re
import tempfile

from .presentation import PresentationBundle

AVATAR_SHA256 = '535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e'
AVATAR_NAME = 'godette-reference-v14a.glb'
_MAX_AVATAR_BYTES = 32 * 1024 * 1024
_DIGEST = re.compile(r'[0-9a-f]{64}\Z')


def build_godette_v14a_presentation(output, *, avatar, catalog):
    """Create a new, validated avatar-only bundle from a real registry catalog.

    ``catalog`` is MotionRegistry.catalog(), not a synthesized six-pilot list.
    All six frozen source bindings must still match. Unrelated library and
    generated motions remain in their original registries; this function
    neither changes nor exports them. Existing output directories are refused.
    """
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise ValueError('Presentation output must be a new directory')
    archive_sha = catalog.get('archive_sha256') if isinstance(catalog, dict) else None
    assets = catalog.get('assets') if isinstance(catalog, dict) else None
    if (not isinstance(archive_sha, str) or not _DIGEST.fullmatch(archive_sha)
            or not isinstance(assets, list)):
        raise ValueError('Invalid source archive catalog')
    sources = {}
    for item in assets:
        if (not isinstance(item, dict) or not isinstance(item.get('id'), str)
                or not isinstance(item.get('sha256'), str) or not _DIGEST.fullmatch(item['sha256'])):
            raise ValueError('Invalid presentation source catalog')
        if item['id'] in sources:
            raise ValueError('Presentation source catalog has duplicate identities')
        sources[item['id']] = item['sha256']
    resources = files(__package__)
    manifest = json.loads(resources.joinpath('godette_v14a_manifest.json').read_text(encoding='utf8'))
    for pilot in manifest['pilots']:
        if sources.get(pilot['asset_id']) != pilot['source_motion_sha256']:
            raise ValueError('Godette v14a presentation pilot source mismatch: ' + pilot['asset_id'])
        pilot.update(language_review='unverified', motion_review='unverified', expression_review='unverified')
    with Path(avatar).open('rb') as stream:
        data = stream.read(_MAX_AVATAR_BYTES + 1)
    if not data or len(data) > _MAX_AVATAR_BYTES:
        raise ValueError('Godette v14a avatar exceeds size limit')
    if hashlib.sha256(data).hexdigest() != AVATAR_SHA256:
        raise ValueError('Frozen Godette v14a avatar SHA256 mismatch')
    manifest.update(avatar=AVATAR_NAME, files={AVATAR_NAME: AVATAR_SHA256},
                    source_archive_sha256=archive_sha, status='research_candidate',
                    export_supported=False, automatic_nonmanual_supported=False,
                    language_review='unverified', motion_review='unverified', expression_review='unverified')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.' + output.name + '-', dir=output.parent) as name:
        staging = Path(name)
        (staging / AVATAR_NAME).write_bytes(data)
        (staging / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        (staging / 'ATTRIBUTION.md').write_text(
            resources.joinpath('godette_v14a_attribution.md').read_text(encoding='utf8'), encoding='utf8')
        PresentationBundle(staging, assets)
        # mkdir is an exclusive cross-platform claim. Check-then-rename would
        # replace a concurrently created empty directory on POSIX. Validate
        # first, claim the new directory, then publish manifest.json LAST.
        # Readers should open the bundle only after this builder returns.
        try:
            output.mkdir()
        except FileExistsError as exc:
            raise ValueError('Presentation output must be a new directory') from exc
        published = []
        try:
            for filename in (AVATAR_NAME, 'ATTRIBUTION.md', 'manifest.json'):
                os.rename(staging / filename, output / filename)
                published.append(output / filename)
        except OSError:
            for path in reversed(published):
                path.unlink(missing_ok=True)
            output.rmdir()
            raise
    return deepcopy(manifest)
