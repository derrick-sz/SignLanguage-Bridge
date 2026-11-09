"""GPU-free task lifecycle and publication contracts using real small archives."""
import hashlib
import json
import threading
import time

import pytest

from slbridge.animation.text_generation_jobs import (
    GenerationService, GenerationBusyError, GenerationUnavailableError,
)
from slbridge.animation.signspark_generation import MODEL_FILES, MODEL_REVISION, SOURCE_COMMIT
from slbridge.animation.soke import write_candidate_archive
from test_animation_registry import make_archive


def wait(service, identity, state):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        job = service.get(identity)
        if job['state'] == state:
            return job
        time.sleep(.01)
    pytest.fail(str(service.get(identity)))


def result(run, text, frames, *, change=None):
    run.mkdir()
    source = make_archive(run)
    request = dict(text=text, frames=frames, steps=50, seed=123)
    digest = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    clip = dict(id='generated-csl-'+digest[:16], text=text, aliases=[], frames=frames, fps=30.,
                poses=[[[0., 0., 0.] for _ in range(55)] for _ in range(frames)],
                source_kind='pretrained_text_conditioned_generation', source_sha256=digest,
                source_model_sha256={s: v[1] for s, v in MODEL_FILES.items()},
                model_revision=MODEL_REVISION, generation_source_commit=SOURCE_COMMIT,
                target_motion=None, observed_keyframes=0, explicit_output_frames=frames,
                quality_warning=True)
    if change == 'clip': clip['text'] = '错误文字'
    archive = run/'candidate-motion.zip'
    write_candidate_archive(archive, source, [clip], 'Test only')
    manifest = dict(state='complete', requests=[request], target_motion=None, observed_keyframes=0,
                    candidate_archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    if change == 'text': request['text'] = '错误文字'
    if change == 'frames': request['frames'] += 1
    if change == 'float_frames': request['frames'] = float(frames)
    if change == 'sha': manifest['candidate_archive_sha256'] = '0'*64
    if change == 'target': manifest['target_motion'] = 'old-motion'
    if change == 'observed': manifest['observed_keyframes'] = 1
    if change == 'missing_target': del manifest['target_motion']
    if change == 'incomplete': manifest['state'] = 'running'
    (run/'generation-manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def test_free_text_and_verified_snapshot(tmp_path):
    def runner(config, text, frames, run, stop, progress):
        progress('hand')
        result(run, text, frames)
        return 0
    service = GenerationService({'output_root': tmp_path}, runner=runner)
    try:
        submitted = service.submit('不要去台北\n明天12点\t再见', 4)
        job = wait(service, submitted['job_id'], 'complete')
        assert job['generation_text'] == '不要去台北 明天12点 再见'
        assert job['text'] == '不要去台北\n明天12点\t再见'
        assert job['language_review'] == 'NOT_RUN'
        motion = service.motion(job['job_id'])
        assert motion['clip']['text'] == job['generation_text']
        assert motion['clip']['frames'] == 120
        from slbridge.animation.registry import MotionRegistry
        archive = next(tmp_path.rglob('candidate-motion.zip'))
        verified = MotionRegistry(archive)
        asset = next(a for a in verified.catalog()['assets'] if a['id'] == motion['clip']['id'])
        assert motion['sha256'] == asset['sha256']
        assert motion['archive_sha256'] == verified.sha256
        for path in tmp_path.rglob('candidate-motion.zip'): path.write_bytes(b'changed')
        assert service.motion(job['job_id']) == motion
        assert str(tmp_path) not in json.dumps(job)
    finally: service.close()


@pytest.mark.parametrize('change', ['text','frames','float_frames','sha','clip','target','observed','missing_target','incomplete','exit'])
def test_invalid_receipt_never_published(tmp_path, change):
    def runner(config, text, frames, run, stop, progress):
        result(run, text, frames, change=change)
        return 1 if change == 'exit' else 0
    service = GenerationService({'output_root': tmp_path}, runner=runner)
    try:
        job = service.submit('库外新句')
        failed = wait(service, job['job_id'], 'failed')
        assert str(tmp_path) not in failed['error']
        with pytest.raises(ValueError): service.motion(job['job_id'])
    finally: service.close()


def test_serial_queue_bound_and_cancel_late_result(tmp_path):
    entered, release = threading.Event(), threading.Event()
    calls = []
    def runner(config, text, frames, run, stop, progress):
        calls.append(text); entered.set(); release.wait(4)
        result(run, text, frames)
        return 0
    service = GenerationService({'output_root': tmp_path}, runner=runner)
    try:
        first = service.submit('第一条'); assert entered.wait(2)
        jobs = [service.submit('排队'+str(i)) for i in range(7)]
        with pytest.raises(GenerationBusyError): service.submit('满了')
        assert len(calls) == 1
        assert service.cancel(first['job_id'])['state'] == 'cancelled'
        for job in jobs: service.cancel(job['job_id'])
        release.set(); time.sleep(.1)
        assert service.get(first['job_id'])['state'] == 'cancelled'
        assert len(calls) == 1
        with pytest.raises(ValueError): service.motion(first['job_id'])
    finally: release.set(); service.close()


@pytest.mark.parametrize('text,seconds', [('',6), (' '*4,6), ('x'*201,6), ('x\x00',6), ('x\x7f',6), ('x',5), ('x',True), ('x',6.)])
def test_invalid_input(tmp_path, text, seconds):
    service = GenerationService({'output_root':tmp_path}, runner=lambda *a: 1)
    try:
        with pytest.raises(ValueError): service.submit(text, seconds)
    finally: service.close()


def test_unavailable_and_unknown(tmp_path):
    service = GenerationService({'output_root':tmp_path})
    try:
        with pytest.raises(GenerationUnavailableError): service.submit('新句')
        for method in (service.get, service.cancel, service.motion):
            with pytest.raises(KeyError): method('unknown')
    finally: service.close()


def test_real_phase_is_exposed_and_status_retention_is_bounded(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def runner(config, text, frames, run, stop, progress):
        if text == '第一条':
            progress('hand'); entered.set(); release.wait(2)
        return 1
    service = GenerationService({'output_root':tmp_path}, runner=runner)
    try:
        first = service.submit('第一条')
        assert entered.wait(2)
        assert service.get(first['job_id'])['phase'] == 'hand'
        release.set(); wait(service, first['job_id'], 'failed')
        for i in range(33):
            current = service.submit('新句'+str(i))
            wait(service, current['job_id'], 'failed')
        with pytest.raises(KeyError): service.get(first['job_id'])
        assert len(service._jobs) == 32
        service.close()
        with pytest.raises(GenerationUnavailableError): service.submit('关闭后')
    finally: release.set(); service.close()


def test_cli_uses_fixed_argv_and_kills_only_its_process_tree(tmp_path, monkeypatch):
    import os
    from slbridge.animation import text_generation_jobs as jobs
    captured = {}
    class Process:
        pid = 87654
        code = None
        def poll(self): return self.code
        def wait(self): return self.code
        def kill(self): self.code = -9
    process = Process()
    def popen(argv, **options):
        captured.update(argv=argv, options=options)
        return process
    def taskkill(argv, **options):
        captured['kill'] = argv
        process.code = -9
    monkeypatch.setattr(jobs.subprocess, 'Popen', popen)
    monkeypatch.setattr(jobs.subprocess, 'run', taskkill)
    if os.name != 'nt':
        monkeypatch.setattr(jobs.os, 'killpg', lambda pid, sig: (captured.update(kill=[pid,sig]), process.kill()))
    config = {k: '/server/'+k for k in jobs._PATH_KEYS}
    run = tmp_path/'run'
    stop = threading.Event(); stop.set()
    assert jobs.cli_runner(config, '--output=malicious 中文', 120, run, stop, lambda p: None) == -9
    assert not run.exists()
    assert '--text=--output=malicious 中文' in captured['argv']
    assert captured['argv'].count('--output') == 1
    assert captured['options']['shell'] is False
    assert captured['kill'][0:2] == (['taskkill','/PID'] if os.name == 'nt' else [process.pid, jobs.signal.SIGKILL])
    if os.name == 'nt':
        assert captured['kill'] == ['taskkill','/PID',str(process.pid),'/T','/F']
        assert captured['options']['creationflags'] == jobs.subprocess.CREATE_NO_WINDOW
def test_memory_failure_has_readable_action_without_server_paths(tmp_path):
    def runner(config,text,frames,run,stop,progress):
        run.mkdir()
        (run/'generation-manifest.json').write_text(json.dumps({'state':'failed','error':'MemoryError: C:/private/weights'}),encoding='utf-8')
        return 1
    service=GenerationService({'output_root':str(tmp_path/'jobs')},runner=runner)
    try:
        submitted=service.submit('新的输入')
        state=wait(service,submitted['job_id'],'failed')
        assert '内存' in state['error']
        assert 'private' not in json.dumps(state)
    finally:
        service.close()
