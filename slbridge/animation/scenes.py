"""Read-only full-expression indexing and bounded, source-preserving composition.

Keyword membership is a browsing aid, never evidence of exact target coverage.
Only explicit sentence boundaries can introduce a composition boundary.
"""
from contextlib import closing
from copy import deepcopy
import hashlib
import hmac
import json
from pathlib import Path
import re
import secrets
import sqlite3
from uuid import uuid4

import numpy as np

from .corpus_store import CorpusIntegrityError
from .gesture import _quaternions, _rotvec, _slerp


PROCESSING_VERSION = 'basic-scene-source-preserving-slerp-v1'
MAX_SEGMENTS = 6
MAX_FRAMES = 1800
TRANSITION_FRAMES = 9
FPS = 30.0


class SceneIntegrityError(CorpusIntegrityError):
    """A planned source or its identity changed; a fresh plan is required."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _text(value):
    # text is natively stored Chinese; some imported indexes JSON-encode strings.
    if isinstance(value, str) and value.startswith('"'):
        try:
            decoded = json.loads(value)
            if isinstance(decoded, str):
                return decoded
        except (ValueError, TypeError):
            pass
    return value


def normalize_expression(text):
    """Preserve semantic punctuation, decimals, names and internal whitespace."""
    text = text.translate(str.maketrans({'？': '?', '！': '!', '；': ';', '，': ',', '．': '.', '。': '.'}))
    text = text.strip()
    # Only terminal sentence periods are optional, never ?/! or decimal points.
    return text.rstrip('.').rstrip()


def _segments(text):
    # A period between digits belongs to a number, including a final fractional digit.
    return [part.strip().rstrip(';；').rstrip() for part in re.findall(r'.*?(?:[?？!！;；\n]+|(?<!\d)[.．]|[.．](?!\d)|。|$)', text)
            if normalize_expression(part)]


class SceneLibrary:
    def __init__(self, store, definitions_path):
        self.store = store
        self.definitions_path = Path(definitions_path)
        definitions = json.loads(self.definitions_path.read_text(encoding='utf-8'))
        if definitions.get('schema_version') != 1 or not isinstance(definitions.get('scenes'), list):
            raise ValueError('Invalid scene definitions')
        self.scenes = deepcopy(definitions['scenes'])
        ids = set()
        for scene in self.scenes:
            if (not isinstance(scene, dict) or not isinstance(scene.get('id'), str) or not scene['id']
                    or scene['id'] in ids or not isinstance(scene.get('title'), str)
                    or not isinstance(scene.get('keywords'), list) or not all(isinstance(k, str) and k for k in scene['keywords'])
                    or not isinstance(scene.get('targets'), list)):
                raise ValueError('Invalid scene definition')
            ids.add(scene['id'])
            target_ids = set()
            for target in scene['targets']:
                if (not isinstance(target, dict) or not isinstance(target.get('id'), str) or not target['id']
                        or target['id'] in target_ids or not isinstance(target.get('text'), str) or not target['text'].strip()):
                    raise ValueError('Invalid scene target')
                target_ids.add(target['id'])
            if not isinstance(scene.get('featured_sample_ids', []), list) or not all(isinstance(x, str) for x in scene.get('featured_sample_ids', [])):
                raise ValueError('Invalid featured sample identities')
        self._secret = secrets.token_bytes(32)
        self._source_identity = self._identity()
        self._rows = {}
        self._by_text = {}
        self._instances = 0
        glosses = set()
        with closing(self._connect()) as db:
            rows = db.execute("SELECT * FROM samples WHERE split='train' AND state='converted_candidate' "
                              "AND actual_frames=expected_frames AND actual_frames BETWEEN 1 AND 3000 ORDER BY id").fetchall()
        for source in rows:
            row = dict(source)
            text = _text(row['text'])
            if not isinstance(text, str) or not normalize_expression(text):
                continue
            self._rows[row['id']] = row
            self._instances += 1
            gloss = json.loads(row['gloss'])
            if isinstance(gloss, str):
                gloss = gloss.split()
            if not isinstance(gloss, list) or not all(isinstance(g, str) for g in gloss):
                raise ValueError('Invalid gloss tokens')
            glosses.update(gloss)
            key = normalize_expression(text)
            # Stable first sample gives one browse entry per distinct expression.
            self._by_text.setdefault(key, self._asset(row, text, gloss))
        self._assets = list(self._by_text.values())
        self._gloss_count = len(glosses)
        self._coverage = self._build_coverage()

    def _connect(self):
        db = sqlite3.connect(Path(self.store.database).resolve().as_uri()+'?mode=ro', uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        return db

    def _identity(self):
        with closing(self._connect()) as db:
            metadata = {row['key']: json.loads(row['value']) for row in db.execute(
                "SELECT key,value FROM metadata WHERE key IN ('build','skeleton')")}
        return _digest({'database': metadata, 'build': self.store.build, 'skeleton': self.store.skeleton})

    def _asset(self, row, text, gloss):
        identity = _digest([row, self._source_identity, PROCESSING_VERSION])[:20]
        return {'id': 'scene-source-'+row['id']+'-'+identity, 'asset_id': 'scene-source-'+row['id']+'-'+identity,
                'sample_id': row['id'], 'text': text, 'gloss': gloss, 'split': row['split'], 'signer': row['signer'],
                'frames': row['actual_frames'], 'fps': FPS, 'duration': row['actual_frames']/FPS,
                'parameter_sha256': row['parameter_sha256'], 'state': row['state'],
                'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN'}

    def _candidates(self, scene):
        featured = scene.get('featured_sample_ids', [])
        candidates = [a for a in self._assets if a['sample_id'] in featured or any(k in a['text'] for k in scene['keywords'])]
        return sorted(candidates, key=lambda a: (featured.index(a['sample_id']) if a['sample_id'] in featured else len(featured), a['sample_id']))

    def catalog(self, scene=None, q='', limit=30, offset=0):
        if (type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or not 0 <= offset <= 2**63-1
                or not isinstance(q, str) or len(q) > 500 or scene is not None and scene not in {s['id'] for s in self.scenes}):
            raise ValueError('Invalid scene catalog parameters')
        assets = self._assets if scene is None else self._candidates(next(s for s in self.scenes if s['id'] == scene))
        if q.strip():
            query = q.strip()
            assets = [a for a in assets if query in a['text'] or query in a['sample_id'] or any(query in g for g in a['gloss'])]
        page = assets[offset:offset+limit]
        return {'schema_version': 1, 'scenes': [{k: row[k] for k in ('id', 'title', 'candidate_texts', 'target_total', 'target_exact')}
                                                for row in self._coverage['scenes']],
                'assets': deepcopy(page), 'total': len(assets), 'limit': limit, 'offset': offset,
                'has_more': offset+len(page) < len(assets), 'language_review': 'NOT_RUN'}

    def _match(self, text):
        key = normalize_expression(text)
        if key in self._by_text:
            return [self._by_text[key]], [], 'existing_expression'
        segments = _segments(text)
        if len(segments) > MAX_SEGMENTS:
            return [], segments, 'expression_sequence'
        matched, unmatched = [], []
        for segment in segments:
            asset = self._by_text.get(normalize_expression(segment))
            if asset is None:
                unmatched.append(segment)
            else:
                matched.append(asset)
        if not segments:
            unmatched = [text]
        return matched, unmatched, 'expression_sequence'

    def _build_coverage(self):
        scenes = []
        for scene in self.scenes:
            targets = []
            for target in scene['targets']:
                assets, unmatched, mode = self._match(target['text'])
                frames = sum(a['frames'] for a in assets)+TRANSITION_FRAMES*max(0,len(assets)-1)
                status = 'blocked' if unmatched or frames > MAX_FRAMES else 'exact' if mode == 'existing_expression' else 'composed'
                targets.append({**target, 'status': status, 'mode': mode, 'unmatched': unmatched,
                                'sample_ids': [a['sample_id'] for a in assets]})
            scenes.append({'id': scene['id'], 'title': scene['title'], 'candidate_texts': len(self._candidates(scene)),
                           'target_total': len(targets), 'target_exact': sum(t['status']=='exact' for t in targets),
                           'target_composed': sum(t['status']=='composed' for t in targets),
                           'target_blocked': sum(t['status']=='blocked' for t in targets), 'targets': targets})
        return {'schema_version': 1, 'scenes': scenes, 'target_total': sum(s['target_total'] for s in scenes),
                'target_exact': sum(s['target_exact'] for s in scenes), 'target_composed': sum(s['target_composed'] for s in scenes),
                'target_blocked': sum(s['target_blocked'] for s in scenes), 'instance_count': self._instances,
                'distinct_texts': len(self._by_text), 'gloss_entries': self._gloss_count,
                'language_review': 'NOT_RUN', 'composition_review': 'NOT_RUN',
                'coverage_basis': 'fixed targets; exact expression matching, not keyword membership'}

    def coverage(self):
        return deepcopy(self._coverage)

    def _load(self, asset):
        try:
            if self._identity() != self._source_identity:
                raise SceneIntegrityError('Corpus template or processing identity changed; create a new library')
            row = self.store._row(asset['sample_id'])
            if row != self._rows[asset['sample_id']]:
                raise SceneIntegrityError('Source metadata changed; select and confirm again')
            poses = self.store._parameters(row)
            if poses.shape != (asset['frames'],55,3) or not np.isfinite(poses).all():
                raise SceneIntegrityError('Invalid source motion')
            return poses
        except (OSError, KeyError, ValueError, sqlite3.Error) as exc:
            raise SceneIntegrityError('Scene source unavailable or identity changed; select and confirm again') from exc

    def _seal(self, plan):
        return hmac.new(self._secret, _json({k:v for k,v in plan.items() if k != '_signature'}).encode(), hashlib.sha256).hexdigest()

    def plan(self, text, confirmed=False, research_preview=False):
        if not isinstance(text, str) or len(text) > 500 or type(confirmed) is not bool or type(research_preview) is not bool:
            raise ValueError('Invalid scene expression input')
        assets, unmatched, mode = self._match(text)
        frames = sum(a['frames'] for a in assets)+TRANSITION_FRAMES*max(0,len(assets)-1)
        plan = {'schema_version': 1, 'plan_id': uuid4().hex, 'input_text': text, 'mode': mode,
                'status': 'blocked', 'can_play': False, 'items': [], 'unmatched': unmatched,
                'reason': '', 'language_review': 'NOT_RUN', 'composition_review': 'NOT_RUN',
                'processing_version': PROCESSING_VERSION, 'fps': FPS, 'frames': frames, 'transitions': [],
                'source_identity': self._source_identity, 'source_archive_sha256': self.store.build['source_archive_sha256'],
                'confirmed': confirmed, 'research_preview': research_preview}
        if unmatched:
            plan['reason'] = '存在未覆盖的完整表达；全部动作已阻断。'
        elif not assets or len(assets) > MAX_SEGMENTS or frames > MAX_FRAMES:
            plan['reason'] = '表达超过6段或1800帧预算。'
        else:
            position = 0
            for asset in assets:
                if position:
                    plan['transitions'].append({'output_frame_range': [position, position+TRANSITION_FRAMES],
                                                'frames': TRANSITION_FRAMES, 'method': 'quaternion-shortest-arc-slerp'})
                    position += TRANSITION_FRAMES
                # Validate parameters even before confirmation: a plan never binds
                # a checksum merely asserted in an index row.
                poses = self._load(asset)
                row = self._rows[asset['sample_id']]
                plan['items'].append({'sample_id': asset['sample_id'], 'asset_id': asset['asset_id'], 'text': asset['text'],
                                      'sha256': _digest([row, self._source_identity]),
                                      'parameter_sha256': row['parameter_sha256'],
                                      'source_parameters_sha256': row['source_parameters_sha256'],
                                      'template_sha256': self.store.build['template_sha256'],
                                      'processing_version': PROCESSING_VERSION, 'fps': FPS, 'frames': len(poses),
                                      'source_frame_range': [0,len(poses)], 'output_frame_range': [position,position+len(poses)]})
                position += len(poses)
            if not confirmed:
                plan.update(status='needs_confirmation', reason='请确认完整输入后再预览。')
            elif not research_preview:
                plan.update(status='needs_language_review', reason='语言尚未认证，请明确选择研究预览。')
            else:
                plan.update(status='research_preview', can_play=True, reason='完整原句研究预览；多句过渡的自然度与接触未经审核。')
        plan['_signature'] = self._seal(plan)
        return plan

    def motion(self, plan):
        try:
            if not isinstance(plan, dict) or not hmac.compare_digest(plan.get('_signature',''), self._seal(plan)):
                raise SceneIntegrityError('Unrecognized or modified scene plan')
            if not plan['can_play'] or not plan['confirmed'] or not plan['research_preview']:
                raise SceneIntegrityError('Scene plan is blocked or unconfirmed')
            pieces = []
            for item in plan['items']:
                asset = self._by_text[normalize_expression(item['text'])]
                poses = self._load(asset)
                if pieces:
                    left = _quaternions(pieces[-1][-1])
                    right = _quaternions(poses[0])
                    pieces.append(np.stack([_rotvec(_slerp(left,right,i/(TRANSITION_FRAMES+1))) for i in range(1,TRANSITION_FRAMES+1)]))
                pieces.append(poses)
            poses = np.concatenate(pieces)
            if len(poses) != plan['frames'] or len(poses) > MAX_FRAMES:
                raise SceneIntegrityError('Scene motion frame budget mismatch')
            composition = {'items': deepcopy(plan['items']), 'transitions': deepcopy(plan['transitions']),
                           'source_frames_preserved': True, 'frame_range_convention': 'zero-based half-open',
                           'processing_version': PROCESSING_VERSION, 'composition_review': 'NOT_RUN'}
            clip = {'schema': 'basic-scene-expression-clip-v1', 'id': 'scene-'+plan['plan_id'],
                    'text': '。'.join(item['text'] for item in plan['items']), 'aliases': [],
                    'fps': FPS, 'frames': len(poses), 'poses': poses.tolist(),
                    'processing_version': PROCESSING_VERSION, 'composition': composition,
                    'template_sha256': self.store.build['template_sha256'],
                    'source_archive_sha256': self.store.build['source_archive_sha256'],
                    'source_dataset': 'SOKE/CSL-Daily train', 'experimental': True, 'human_approved': False,
                    'training_ready': False, 'language_review': 'NOT_RUN', 'motion_review': 'NOT_RUN',
                    'composition_review': 'NOT_RUN', 'unmapped_target_channels': ['expression','jaw','shape','camera_translation']}
            # Repeat metadata identity validation after loading all segments.
            if self._identity() != self._source_identity:
                raise SceneIntegrityError('Corpus identity changed during motion construction')
            return {'clip': clip, 'parents': deepcopy(self.store.skeleton['parents']),
                    'restJoints': deepcopy(self.store.skeleton['restJoints']), 'sha256': _digest(clip),
                    'processing_version': PROCESSING_VERSION, 'catalog_version': self.store.version}
        except (KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise SceneIntegrityError('Scene plan unavailable or source identity changed') from exc
