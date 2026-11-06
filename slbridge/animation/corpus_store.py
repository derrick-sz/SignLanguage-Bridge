"""Read-only, bounded access to a live SQLite SOKE library and one NPZ at a time."""
from contextlib import closing
from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
from uuid import uuid4
import zipfile

import numpy as np

from .gesture import GESTURE_PROFILE, refine_gestures
from .planner import normalize_text
from .registry import _skeleton, _validate_clip

MAX_PARAMETER_BYTES = 32 * 1024 * 1024
MAX_FRAMES = 3000
PLAYBACK_VERSION = 'soke-full-lazy-gesture-jaw-v2'
JAW_TEMPLATE_SHA256 = '376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992'
JAW_CANDIDATE = {
    'profile': 'smplx-jaw-godette-calibrated-v1', 'source_joint': 22,
    'source_template_sha256': JAW_TEMPLATE_SHA256,
    'target_avatar_sha256': 'b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c',
    'status': 'research_candidate', 'export_supported': False,
}
_SAMPLE = re.compile(r'S\d{6}_P\d{4}_T\d{2}\Z')
_SHA = re.compile(r'[0-9a-f]{64}\Z')


class CorpusIntegrityError(ValueError):
    """Unavailable, changed or invalid single-sequence data; never expose paths."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


class CorpusStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.database = self.root/'corpus.sqlite3'
        with closing(self._connect()) as db:
            metadata = {r['key']: json.loads(r['value']) for r in db.execute(
                "SELECT key,value FROM metadata WHERE key IN ('build','skeleton')")}
        self.build = metadata.get('build', {})
        self.skeleton = metadata.get('skeleton', {})
        if self.build.get('schema') != 'soke-full-library-v1':
            raise ValueError('Unsupported corpus schema')
        for key in ('source_archive_sha256', 'template_sha256'):
            if not isinstance(self.build.get(key), str) or not _SHA.fullmatch(self.build[key]):
                raise ValueError('Invalid corpus source identity')
        if not isinstance(self.build.get('processing_version'), str):
            raise ValueError('Missing corpus processing version')
        _skeleton(self.skeleton)
        if self.skeleton.get('smplx_rest_model_sha256') != self.build['template_sha256']:
            raise ValueError('Corpus skeleton/template mismatch')
        self.version = hashlib.sha256(_json(metadata).encode()).hexdigest()

    def _connect(self):
        db = sqlite3.connect(self.database.as_uri()+'?mode=ro', uri=True, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA query_only=ON')
        return db

    def status(self):
        with closing(self._connect()) as db:
            counts = {row['state']: row['n'] for row in db.execute('SELECT state,COUNT(*) AS n FROM samples GROUP BY state')}
        return {'enabled': True, 'schema_version': 1, 'catalog_version': self.version,
                'counts': counts, 'total': sum(counts.values()), 'build': deepcopy(self.build),
                'language_review': 'unverified', 'motion_review': 'unverified'}

    def _row(self, sample_id):
        if not isinstance(sample_id, str) or not _SAMPLE.fullmatch(sample_id):
            raise KeyError(sample_id)
        with closing(self._connect()) as db:
            row = db.execute('SELECT * FROM samples WHERE id=?', (sample_id,)).fetchone()
        if row is None:
            raise KeyError(sample_id)
        return dict(row)

    def _asset(self, row):
        # Include source metadata and playback protocol so a changed file cannot hit
        # MotionStage's ID-only cache with different poses or sentence attribution.
        identity = hashlib.sha256(_json([row, self.version, PLAYBACK_VERSION, GESTURE_PROFILE, JAW_CANDIDATE]).encode()).hexdigest()[:16]
        frames = row['actual_frames'] or row['expected_frames']
        metrics = json.loads(row['metrics_json']) if row['metrics_json'] else {}
        return {'id': 'soke-full-'+row['id']+'-'+identity, 'sample_id': row['id'], 'text': row['text'],
                'gloss': json.loads(row['gloss']), 'split': row['split'], 'signer': row['signer'],
                'aliases': [], 'group': 'corpus', 'fps': 30.0, 'frames': frames,
                'duration': frames/30.0, 'state': row['state'],
                'playable': row['state'] == 'converted_candidate' and 1 <= frames <= MAX_FRAMES,
                'quality_warning': metrics.get('mapped_max_step_degrees', 0) > 30,
                'metrics': metrics, 'parameter_sha256': row['parameter_sha256'],
                'language_review': 'unverified', 'motion_review': 'unverified',
                'expression_review': 'unverified', 'temporal_alignment': 'unverified'}

    def detail(self, sample_id):
        return self._asset(self._row(sample_id))

    def search(self, *, q='', split=None, quality='all', state='converted_candidate', limit=30, offset=0):
        if (not isinstance(q, str) or len(q) > 500 or type(limit) is not int or not 1 <= limit <= 100
                or type(offset) is not int or not 0 <= offset <= 2**63-1 or split not in (None, 'train', 'val', 'test')
                or quality not in ('all', 'warning', 'quiet') or state not in ('all', 'converted_candidate')):
            raise ValueError('Invalid corpus search parameters')
        clauses = []; values = []
        if state != 'all':
            clauses.append('state=?'); values.append(state)
        if split is not None:
            clauses.append('split=?'); values.append(split)
        if q.strip():
            escaped = q.strip().replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
            clauses.append("(text LIKE ? ESCAPE '\\' OR gloss LIKE ? ESCAPE '\\' OR id LIKE ? ESCAPE '\\')")
            values.extend(['%'+escaped+'%']*3)
        if quality != 'all':
            comparison = '>' if quality == 'warning' else '<='
            clauses.append("COALESCE(json_extract(metrics_json,'$.mapped_max_step_degrees'),0) "+comparison+' 30')
        where = ' WHERE '+' AND '.join(clauses) if clauses else ''
        with closing(self._connect()) as db:
            # Keep total and page within one reader snapshot while the writer commits.
            db.execute('BEGIN')
            total = db.execute('SELECT COUNT(*) FROM samples'+where, values).fetchone()[0]
            rows = db.execute('SELECT * FROM samples'+where+' ORDER BY id LIMIT ? OFFSET ?', [*values,limit,offset]).fetchall()
        return {'schema_version': 1, 'catalog_version': self.version,
                'assets': [self._asset(dict(row)) for row in rows], 'total': total,
                'limit': limit, 'offset': offset, 'has_more': offset+len(rows) < total}

    def _parameters(self, row):
        try:
            frames = row['actual_frames']
            if row['state'] != 'converted_candidate' or type(frames) is not int or not 1 <= frames <= MAX_FRAMES or frames != row['expected_frames']:
                raise ValueError('Sequence is not converted and complete')
            if not isinstance(row['source_parameters_sha256'], str) or not _SHA.fullmatch(row['source_parameters_sha256']):
                raise ValueError('Invalid source parameter identity')
            relative = row['parameter_file']
            if not isinstance(relative, str) or '\\' in relative or ':' in relative or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts:
                raise ValueError('Invalid parameter path')
            path = (self.root/relative).resolve()
            if not path.is_relative_to(self.root) or path.suffix != '.npz':
                raise ValueError('Invalid parameter path')
            with path.open('rb') as stream:
                raw = stream.read(MAX_PARAMETER_BYTES+1)
            if not raw or len(raw) > MAX_PARAMETER_BYTES:
                raise ValueError('Parameter size exceeds bound')
            if not isinstance(row['parameter_sha256'], str) or not _SHA.fullmatch(row['parameter_sha256']) or hashlib.sha256(raw).hexdigest() != row['parameter_sha256']:
                raise ValueError('Parameter checksum mismatch')
            with zipfile.ZipFile(BytesIO(raw)) as archive:
                entries = archive.infolist()
                if len(entries) > 32 or len({e.filename for e in entries}) != len(entries) or sum(e.file_size for e in entries) > MAX_PARAMETER_BYTES:
                    raise ValueError('Expanded parameter budget exceeded')
                # Check NPY headers before numpy allocates from their declared shapes.
                for entry in entries:
                    with archive.open(entry) as stream:
                        version = np.lib.format.read_magic(stream)
                        if version == (1, 0): shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                        elif version == (2, 0): shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                        else: raise ValueError('Unsupported numeric array version')
                        size = math.prod(shape)*dtype.itemsize
                        if dtype.hasobject or dtype.kind not in 'fibu' or size > MAX_PARAMETER_BYTES or stream.tell()+size != entry.file_size:
                            raise ValueError('Invalid numeric array payload')
            with np.load(BytesIO(raw), allow_pickle=False) as arrays:
                poses = arrays['effective_poses']; numbers = arrays['frame_numbers']
                if poses.shape != (frames,55,3) or poses.dtype.kind != 'f' or not np.isfinite(poses).all():
                    raise ValueError('Invalid effective poses')
                if numbers.dtype.kind not in 'iu' or not np.array_equal(numbers, np.arange(frames)):
                    raise ValueError('Non-contiguous parameter frames')
                return poses.copy()
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, OverflowError, EOFError) as exc:
            raise CorpusIntegrityError('Sequence unavailable or failed identity/parameter validation') from exc

    def motion(self, sample_id, expected_sha256=None):
        row = self._row(sample_id); asset = self._asset(row)
        poses = self._parameters(row)
        refined, audit, _ = refine_gestures(poses, 30.0)
        clip = {'schema': 'soke-full-lazy-clip-v1', 'id': asset['id'], 'text': asset['text'], 'aliases': [],
                'fps': 30.0, 'frames': len(poses), 'poses': poses.tolist(), 'refined_poses': refined.tolist(),
                'gesture_refinement': audit, 'refinement_origin': 'on_demand_from_effective_poses',
                'processing_version': PLAYBACK_VERSION, 'parameter_sha256': row['parameter_sha256'],
                'source_id': sample_id, 'source_dataset': 'SOKE/CSL-Daily',
                'source_archive_sha256': self.build['source_archive_sha256'],
                'fitted_sha256': row['source_parameters_sha256'], 'template_sha256': self.build['template_sha256'],
                'source_video_binding': {'temporal_alignment': 'unverified', 'pose_frames': len(poses), 'pose_fps': 30.0,
                    'pose_fps_basis': 'CSL-Daily native 30 FPS; release timestamps not independently verified'},
                'experimental': True, 'human_approved': False, 'training_ready': False,
                'language_review': 'unverified', 'motion_review': 'unverified', 'expression_review': 'unverified',
                'unmapped_target_channels': ['expression','jaw','shape','camera_translation']}
        if self.build['template_sha256'] == JAW_TEMPLATE_SHA256:
            clip['nonmanual_candidate'] = deepcopy(JAW_CANDIDATE)
            clip['unmapped_target_channels'] = ['expression','jaw_swing','eyes','shape','camera_translation']
        _validate_clip(clip)
        digest = hashlib.sha256(_json(clip).encode()).hexdigest()
        if expected_sha256 is not None and expected_sha256 != digest:
            raise CorpusIntegrityError('Motion changed since planning; select and confirm again')
        return {'clip': clip, 'parents': deepcopy(self.skeleton['parents']),
                'restJoints': deepcopy(self.skeleton['restJoints']), 'sha256': digest,
                'catalog_version': self.version, 'processing_version': PLAYBACK_VERSION}

    def plan(self, sample_id, text, *, confirmed=False, research_preview=False, origin=None):
        asset = self.detail(sample_id)
        normalized = normalize_text(text).replace(',', '')
        plan = {'schema_version': 1, 'plan_id': uuid4().hex, 'input_text': text, 'origin': deepcopy(origin or {}),
                'sign_system': 'CSL', 'source_archive_sha256': self.build['source_archive_sha256'],
                'catalog_version': self.version, 'language_review': 'unverified', 'can_play': False, 'items': []}
        if not asset['playable']:
            plan.update(status='blocked', reason='该序列尚未完成参数转换。')
        elif not normalized or normalized != normalize_text(asset['text']).replace(',', ''):
            plan.update(status='blocked', reason='输入与选中完整句不一致；请重新选择，未省略或猜测内容。')
        elif not confirmed:
            plan.update(status='needs_confirmation', reason='请确认完整输入后再预览。')
        elif not research_preview:
            plan.update(status='needs_language_review', reason='语言尚未认证，只可明确选择研究预览。')
        else:
            motion = self.motion(sample_id)
            plan.update(status='research_preview', can_play=True, reason='研究候选：动作、表情及原视频时间对齐未审核。',
                        items=[{'sample_id': sample_id, 'asset_id': motion['clip']['id'], 'sha256': motion['sha256'],
                                'duration': asset['duration'], 'asset': asset}])
        return plan
