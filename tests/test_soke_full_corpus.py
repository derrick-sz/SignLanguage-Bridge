"""Breaks caught: silent gap filling, lost raw fields, false resume and stale inputs."""
import hashlib
import json
import pickle
import sqlite3
import zipfile

import numpy as np
import pytest

from slbridge.animation.soke import FIELDS


def source_fixture(tmp_path, bad=False):
    archive = tmp_path/'source.zip'
    rows = []
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, numbers in enumerate(([0, 1, 2], [0, 2], [])):
            identity = f'S{i:06}_P0000_T00'
            rows.append(dict(name=identity, text=['你好', '开会', '缺失'][i], gloss='你好',
                             num_frames=3, signer=0, split='train'))
            for n in numbers:
                frame = {k: np.full(size, n/100, dtype=np.float32) for k, size in FIELDS.items()}
                if bad and i == 0 and n == 1:
                    frame['smplx_body_pose'][0] = np.nan
                z.writestr(f'csl-daily_pose/{identity}/{n:06}.pkl', pickle.dumps(frame, protocol=4))
    return archive, rows


def run(tmp_path, archive, rows, **kwargs):
    from slbridge.animation.full_corpus import build_library
    return build_library(archive, rows, np.zeros((2, 45), dtype=np.float32),
                         {'parents': [-1]+list(range(54)), 'restJoints': [[0, 0, 0]]*55},
                         tmp_path/'library', template_sha256='a'*64,
                         index_sha256={'train': 'b'*64}, **kwargs)


def test_full_conversion_retains_raw_and_records_gaps_and_missing(tmp_path):
    archive, rows = source_fixture(tmp_path)
    result = run(tmp_path, archive, rows)
    assert result['states'] == {'converted_candidate': 1, 'incomplete_frames': 1, 'missing_3d_sequence': 1}
    assert result['converted_frames'] == 3
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    path, digest = db.execute('select parameter_file,parameter_sha256 from samples where state=?',
                             ('converted_candidate',)).fetchone()
    output = tmp_path/'library'/path
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    with np.load(output, allow_pickle=False) as value:
        assert set(FIELDS) <= set(value.files)
        np.testing.assert_array_equal(value['smplx_body_pose'][1], np.full(63, .01, dtype=np.float32))
        np.testing.assert_array_equal(value['effective_poses'][:, :12], 0)
        np.testing.assert_array_equal(value['frame_numbers'], [0, 1, 2])
        assert value['features133_raw'].shape == (3, 133)
    assert json.loads(db.execute('select value from metadata where key="build"').fetchone()[0])['source_archive_sha256'] == hashlib.sha256(archive.read_bytes()).hexdigest()


def test_resume_skips_verified_output_but_rebuilds_missing_or_corrupt(tmp_path):
    archive, rows = source_fixture(tmp_path)
    run(tmp_path, archive, rows)
    output = next((tmp_path/'library/sequences').rglob('*.npz'))
    original = output.read_bytes()
    assert run(tmp_path, archive, rows)['attempted_this_run'] == 0
    output.write_bytes(b'corrupt')
    assert run(tmp_path, archive, rows)['attempted_this_run'] == 1
    assert output.read_bytes() == original
    output.unlink()
    assert run(tmp_path, archive, rows)['attempted_this_run'] == 1


def test_resume_rejects_changed_source_and_annotations(tmp_path):
    archive, rows = source_fixture(tmp_path)
    run(tmp_path, archive, rows)
    changed = [dict(row) for row in rows]
    changed[0]['text'] = '错误身份'
    with pytest.raises(ValueError, match='identity|protocol|inputs'):
        run(tmp_path, archive, changed)
    with zipfile.ZipFile(archive, 'a') as z:
        z.writestr('new.txt', b'drift')
    with pytest.raises(ValueError, match='identity|protocol|inputs'):
        run(tmp_path, archive, rows)


def test_nonfinite_sequence_is_failed_without_publishing_playable_output(tmp_path):
    archive, rows = source_fixture(tmp_path, bad=True)
    result = run(tmp_path, archive, rows)
    assert result['states']['failed'] == 1
    assert not list((tmp_path/'library/sequences').rglob('*.npz'))
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    assert 'smplx_body_pose' in db.execute('select error from samples where state="failed"').fetchone()[0]


def test_malformed_dtype_fails_one_sequence_and_continues_conversion(tmp_path):
    from contextlib import closing

    malformed = (b'cnumpy.core.multiarray\n_reconstruct\n)R(K\x01K\x01\x85'
                 b'cnumpy\ndtype\n)R\x89C\x04\x00\x00\x00\x00tb.')
    valid = pickle.dumps({key: np.zeros(size, dtype=np.float32)
                          for key, size in FIELDS.items()}, protocol=4)
    archive = tmp_path/'source.zip'
    rows = [dict(name=f'S{i:06}_P0000_T00', text='fixture', gloss='fixture',
                 num_frames=1, signer=0, split='train') for i in range(2)]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for row, payload in zip(rows, [malformed, valid]):
            z.writestr(f"csl-daily_pose/{row['name']}/000000.pkl", payload)

    result = run(tmp_path, archive, rows)

    assert result['states'] == {'failed': 1, 'converted_candidate': 1}
    with closing(sqlite3.connect(tmp_path/'library/corpus.sqlite3')) as db:
        stored = db.execute('SELECT state,error,parameter_file FROM samples ORDER BY id').fetchall()
    assert stored[0][0] == 'failed' and 'dtype' in stored[0][1]
    assert stored[0][2] is None
    assert stored[1][0] == 'converted_candidate'
    assert (tmp_path/'library'/stored[1][2]).is_file()


def test_writer_lock_rejects_second_writer(tmp_path):
    from slbridge.animation.full_corpus import writer_lock
    with writer_lock(tmp_path):
        with pytest.raises(ValueError, match='writer'):
            with writer_lock(tmp_path):
                pass


def test_damaged_deflate_is_recorded_as_failed_instead_of_stopping_the_job(tmp_path):
    import struct
    archive, rows = source_fixture(tmp_path)
    with archive.open('r+b') as stream:
        header = struct.unpack('<4s5H3L2H', stream.read(30))
        stream.seek(header[-2]+header[-1], 1)
        stream.write(b'\x07\xff\xff')  # invalid DEFLATE block type
    result = run(tmp_path, archive, rows)
    assert result['states']['failed'] == 1
    assert result['states']['missing_3d_sequence'] == 1


def test_artifact_audit_detects_hand_mean_or_raw_feature_drift(tmp_path):
    from slbridge.animation.full_corpus_audit import audit_library
    archive, rows = source_fixture(tmp_path)
    run(tmp_path, archive, rows)
    result = audit_library(tmp_path/'library')
    assert result['passed'] is True
    assert result['verified_sequences'] == 1 and result['verified_frames'] == 3
    assert result['unchecked_archive_frame_files'] == 2
    output = next((tmp_path/'library/sequences').rglob('*.npz'))
    with np.load(output, allow_pickle=False) as f:
        values = {key: f[key] for key in f.files}
    values['effective_poses'][1, 25, 0] = 1
    np.savez_compressed(output, **values)
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    db.execute('UPDATE samples SET parameter_sha256=? WHERE state="converted_candidate"',
               (hashlib.sha256(output.read_bytes()).hexdigest(),))
    db.commit()
    result = audit_library(tmp_path/'library')
    assert result['passed'] is False
    assert result['errors'][0]['id'] == 'S000000_P0000_T00'


def test_audit_does_not_call_a_partial_run_full_conversion(tmp_path):
    from slbridge.animation.full_corpus_audit import audit_library
    archive, rows = source_fixture(tmp_path)
    # A structurally complete second identity remains pending after one attempt.
    with zipfile.ZipFile(archive, 'a', zipfile.ZIP_DEFLATED) as z:
        with zipfile.ZipFile(archive) as reader:
            raw = reader.read('csl-daily_pose/S000000_P0000_T00/000000.pkl')
        for number in range(3):
            z.writestr(f'csl-daily_pose/S000003_P0000_T00/{number:06}.pkl', raw)
    rows.append(dict(rows[0], name='S000003_P0000_T00'))
    run(tmp_path, archive, rows, limit=1)
    result = audit_library(tmp_path/'library')
    assert result['passed'] is True
    assert result['full_numeric_conversion_complete'] is False
    assert result['states']['pending'] == 1


def test_verified_template_binding_survives_registry_motion_metadata_loss(tmp_path):
    archive, rows = source_fixture(tmp_path)
    run(tmp_path, archive, rows)
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    skeleton = json.loads(db.execute('select value from metadata where key="skeleton"').fetchone()[0])
    assert skeleton['smplx_rest_model_sha256'] == 'a'*64
    # Initial interrupted rollout omitted only this redundant binding field.
    skeleton.pop('smplx_rest_model_sha256')
    db.execute('update metadata set value=? where key="skeleton"', (json.dumps(skeleton),))
    db.commit()
    assert run(tmp_path, archive, rows)['attempted_this_run'] == 0
    restored = json.loads(db.execute('select value from metadata where key="skeleton"').fetchone()[0])
    assert restored['smplx_rest_model_sha256'] == 'a'*64
    restored['restJoints'][1][0] = 1
    db.execute('update metadata set value=? where key="skeleton"', (json.dumps(restored),))
    db.commit()
    with pytest.raises(ValueError, match='identity|protocol|inputs'):
        run(tmp_path, archive, rows)


def test_malformed_pickle_isolated_and_later_healthy_sequence_converts(tmp_path):
    archive, rows = source_fixture(tmp_path)
    with zipfile.ZipFile(archive) as z:
        entries = [(i.filename, z.read(i)) for i in z.infolist()]
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for i, (name, raw) in enumerate(entries):
            z.writestr(name, b'}K\x01a.' if i == 0 else raw)
        for n in range(3):
            z.writestr(f'csl-daily_pose/S000003_P0000_T00/{n:06}.pkl', entries[0][1])
    rows.append(dict(rows[0], name='S000003_P0000_T00'))
    result = run(tmp_path, archive, rows)
    assert result['states']['failed'] == 1
    assert result['states']['converted_candidate'] == 1
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    assert 'Malformed pickle' in db.execute('select error from samples where state="failed"').fetchone()[0]


def test_audit_rejects_huge_npy_shape_before_numpy_allocation(tmp_path, monkeypatch):
    import io
    from slbridge.animation.full_corpus_audit import audit_library
    archive, rows = source_fixture(tmp_path)
    run(tmp_path, archive, rows)
    output = next((tmp_path/'library/sequences').rglob('*.npz'))
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<f8', 'fortran_order': False, 'shape': (10**10, 3)})
    with zipfile.ZipFile(output, 'w') as z:
        z.writestr('effective_poses.npy', header.getvalue())
    db = sqlite3.connect(tmp_path/'library/corpus.sqlite3')
    db.execute('update samples set parameter_sha256=? where state="converted_candidate"',
               (hashlib.sha256(output.read_bytes()).hexdigest(),))
    db.commit()

    def unsafe_allocation(*args, **kwargs):
        raise AssertionError('Unbounded NPY reached numpy allocation')

    # Protect the machine at the dangerous allocation boundary; the ZIP/NPY
    # headers and DB are real, and must be rejected before this boundary.
    monkeypatch.setattr(np, 'load', unsafe_allocation)
    result = audit_library(tmp_path/'library')
    assert result['passed'] is False
    assert 'budget' in result['errors'][0]['error'].lower()
