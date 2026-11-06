"""Bounded, source-bound SOKE ingestion. No language approval or gap filling."""
from contextlib import contextmanager
from dataclasses import astuple
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time

import numpy as np

from .gesture import motion_metrics
from .soke import Member, effective_poses, load_sequence, scan_archive

PROCESSING_VERSION = 'soke-full-raw-upper133-v1'
ID = re.compile(r'S\d{6}_P\d{4}_T\d{2}\Z')


def serialized(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(',', ':'))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(8*1024*1024):
            h.update(data)
    return h.hexdigest()


@contextmanager
def writer_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'writer.lock').open('a+b') as f:
        f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another library writer is active') from exc
        try:
            yield
        finally:
            f.seek(0)
            if os.name == 'nt':
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def _validate_rows(rows):
    ids = set()
    for row in rows:
        name = row['name']
        if (not isinstance(name, str) or not ID.fullmatch(name) or name in ids
                or type(row['num_frames']) is not int or not 1 <= row['num_frames'] <= 3000
                or not isinstance(row['text'], str) or type(row['signer']) is not int):
            raise ValueError('Invalid annotation identity or frame bounds')
        ids.add(name)


def _initialize(path, archive, rows, metadata, progress):
    staging = path.with_suffix('.indexing.sqlite3')
    # Only this writer's single staging file; an interrupted index is never published.
    staging.unlink(missing_ok=True)
    db = sqlite3.connect(staging)
    try:
        db.executescript('''
            CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE TABLE samples (
                id TEXT PRIMARY KEY,text TEXT NOT NULL,gloss TEXT NOT NULL,split TEXT NOT NULL,
                signer INTEGER NOT NULL,expected_frames INTEGER NOT NULL,actual_frames INTEGER NOT NULL,
                state TEXT NOT NULL,parameter_file TEXT,parameter_sha256 TEXT,
                source_parameters_sha256 TEXT,metrics_json TEXT,error TEXT);
            CREATE INDEX samples_state_id ON samples(state,id);
            CREATE TABLE members (
                sample_id TEXT NOT NULL,filename TEXT NOT NULL,frame_number INTEGER NOT NULL,
                flags INTEGER,method INTEGER,crc INTEGER,compressed_size INTEGER,size INTEGER,offset INTEGER,
                PRIMARY KEY(sample_id,frame_number));
        ''')
        db.executemany('INSERT INTO metadata VALUES (?,?)', [(k, serialized(v)) for k, v in metadata.items()])
        db.executemany('INSERT INTO samples(id,text,gloss,split,signer,expected_frames,actual_frames,state) VALUES (?,?,?,?,?,?,0,?)',
                       [(r['name'], r['text'], serialized(r['gloss']), r['split'], r['signer'], r['num_frames'], 'pending') for r in rows])
        known = {r['name'] for r in rows}
        batch = []

        def receive(identity, member):
            if identity in known:
                batch.append((identity, *astuple(member)))
                if len(batch) >= 5000:
                    db.executemany('INSERT INTO members VALUES (?,?,?,?,?,?,?,?,?)', batch)
                    batch.clear()

        inventory, _ = scan_archive(archive, set(), progress=progress, member_callback=receive)
        db.executemany('INSERT INTO members VALUES (?,?,?,?,?,?,?,?,?)', batch)
        for row in rows:
            item = inventory['sequences'].get(row['name'])
            state = ('missing_3d_sequence' if item is None else
                     'pending' if item['frames'] == row['num_frames'] and item['first_frame'] == 0
                     and not item['missing_frame_numbers'] else 'incomplete_frames')
            db.execute('UPDATE samples SET state=?,actual_frames=? WHERE id=?',
                       (state, item['frames'] if item else 0, row['name']))
        info = {'archive_entries': inventory['archive_entries'],
                'archive_sequences': len(inventory['sequences']),
                'actual_frame_files': sum(r['frames'] for r in inventory['sequences'].values()),
                'unknown_files': inventory['unknown_files']}
        db.execute('INSERT INTO metadata VALUES (?,?)', ('inventory', serialized(info)))
        db.commit()
    finally:
        db.close()
    os.replace(staging, path)


def _atomic_json(path, value):
    temp = path.with_suffix('.tmp.json')
    temp.write_text(serialized(value), encoding='utf-8')
    os.replace(temp, path)


def library_progress(db, attempted=0, started=None):
    states = dict(db.execute('SELECT state,count(*) FROM samples GROUP BY state'))
    frames = db.execute('SELECT coalesce(sum(expected_frames),0) FROM samples WHERE state="converted_candidate"').fetchone()[0]
    return {'schema': 'soke-full-library-progress-v1', 'states': states,
            'indexed_samples': sum(states.values()), 'converted_frames': frames,
            'attempted_this_run': attempted, 'elapsed_seconds': round(time.monotonic()-started, 3) if started else None,
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'motion_review': 'NOT_RUN', 'language_review': 'NOT_RUN',
            'original_video_temporal_alignment': 'unverified'}


def build_library(archive, rows, hand_mean, skeleton, output, *, template_sha256,
                  index_sha256, limit=None, retry_failed=False, progress=None):
    """Index all identities, then convert every complete sequence, safely resumable."""
    archive, output = Path(archive), Path(output)
    rows = list(rows)
    _validate_rows(rows)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    mean = np.asarray(hand_mean, dtype=np.float32)
    if mean.shape != (2, 45) or not np.isfinite(mean).all():
        raise ValueError('Invalid verified hand mean')
    skeleton = dict(skeleton)
    if skeleton.get('smplx_rest_model_sha256', template_sha256) != template_sha256:
        raise ValueError('Skeleton/template identity differs')
    skeleton['smplx_rest_model_sha256'] = template_sha256
    with writer_lock(output):
        build = {'schema': 'soke-full-library-v1', 'source_archive_sha256': sha256_file(archive),
                 'template_sha256': template_sha256, 'index_sha256': index_sha256,
                 'annotations_sha256': hashlib.sha256(serialized(rows).encode()).hexdigest(),
                 'processing_version': PROCESSING_VERSION, 'refined': False,
                 'hand_mean_sha256': hashlib.sha256(mean.tobytes()).hexdigest()}
        metadata = {'build': build, 'skeleton': skeleton, 'hand_mean': mean.tolist()}
        path = output/'corpus.sqlite3'
        started = time.monotonic()
        if not path.exists():
            _initialize(path, archive, rows, metadata, progress)
        db = sqlite3.connect(path, timeout=30)
        try:
            for key, value in metadata.items():
                saved = db.execute('SELECT value FROM metadata WHERE key=?', (key,)).fetchone()
                # One additive provenance repair for the first full-library run.
                # Build identity is already checked; graph/rest coordinates must
                # remain exactly equal to the caller's verified template skeleton.
                if key == 'skeleton' and saved is not None:
                    old = json.loads(saved[0])
                    if 'smplx_rest_model_sha256' not in old and old | {'smplx_rest_model_sha256': template_sha256} == value:
                        db.execute('UPDATE metadata SET value=? WHERE key=?', (serialized(value), key))
                        saved = (serialized(value),)
                if saved is None or json.loads(saved[0]) != value:
                    raise ValueError('Library inputs/protocol/identity differ; use a new output directory')
            db.commit()
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('PRAGMA synchronous=NORMAL')
            # SHA ties the already-validated arrays to their committed files.
            for identity, relative, expected in db.execute('SELECT id,parameter_file,parameter_sha256 FROM samples WHERE state="converted_candidate"').fetchall():
                target = (output/relative).resolve()
                if not target.is_relative_to(output.resolve()):
                    raise ValueError('Output parameter path escapes library')
                if not target.is_file() or sha256_file(target) != expected:
                    db.execute('UPDATE samples SET state="pending",error="Resume file missing or corrupt" WHERE id=?', (identity,))
            if retry_failed:
                db.execute('UPDATE samples SET state="pending" WHERE state="failed"')
            db.commit()
            attempted = 0
            pending = db.execute('SELECT id,expected_frames FROM samples WHERE state="pending" ORDER BY id').fetchall()
            with archive.open('rb') as stream:
                for identity, count in pending:
                    if limit is not None and attempted >= limit:
                        break
                    attempted += 1
                    try:
                        members = [Member(*m) for m in db.execute('SELECT filename,frame_number,flags,method,crc,compressed_size,size,offset FROM members WHERE sample_id=? ORDER BY frame_number', (identity,))]
                        parameters, digest = load_sequence(archive, members, count, stream=stream)
                        poses = effective_poses(parameters, mean)
                        metrics = motion_metrics(poses, 30.0)
                        features = np.concatenate([parameters['smplx_body_pose'][:, 33:], parameters['smplx_lhand_pose'],
                                                   parameters['smplx_rhand_pose'], parameters['smplx_jaw_pose'], parameters['smplx_expr']], axis=-1)
                        relative = f'sequences/{identity[1:4]}/{identity}.npz'
                        target = output/relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        temporary = target.with_suffix('.tmp.npz')
                        with temporary.open('wb') as f:
                            np.savez_compressed(f, **parameters, effective_poses=poses, features133_raw=features,
                                                frame_numbers=np.arange(count, dtype=np.int32))
                            f.flush()
                            os.fsync(f.fileno())
                        file_digest = sha256_file(temporary)
                        os.replace(temporary, target)
                        db.execute('UPDATE samples SET state="converted_candidate",parameter_file=?,parameter_sha256=?,source_parameters_sha256=?,metrics_json=?,error=NULL WHERE id=?',
                                   (relative, file_digest, digest, serialized(metrics), identity))
                    except (ValueError, OSError) as exc:
                        db.execute('UPDATE samples SET state="failed",error=? WHERE id=?', (str(exc)[:2000], identity))
                    db.commit()
                    if attempted % 25 == 0:
                        status = library_progress(db, attempted, started)
                        _atomic_json(output/'progress.json', status)
                        if progress:
                            progress(status, None)
            status = library_progress(db, attempted, started)
            _atomic_json(output/'progress.json', status)
            return status
        finally:
            db.close()
