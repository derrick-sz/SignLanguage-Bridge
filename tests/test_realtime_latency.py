"""Timing must survive cancellation, and warmup must use the native owner."""
from contextlib import contextmanager
import threading
from types import SimpleNamespace

import numpy as np

from slbridge import realtime_service as service_module
from slbridge.runtime import ModelManager
from slbridge.uni_sign import RtmposeExtractor
from tests.test_realtime_engine import setup, feed
from tests.test_realtime_service import config, model, Runtime
from tests.test_realtime_worker import packet, session, wait


def test_cancel_keeps_bounded_timing_report_before_native_close():
    gate = threading.Event()
    active, engines = session(gate)
    engine = engines[0]
    engine.performance = lambda: {'native_closed': engine.closed, 'frames_processed': len(engine.seen)}
    active.poll()
    active.submit(packet(0))
    wait(lambda: engine.seen)
    active.cancel('source_changed')
    gate.set()
    assert active.join(2)
    report = active.performance()
    assert report['reason'] == 'source_changed' and report['frames_completed'] == 1
    assert report['native'] == {'native_closed': False, 'frames_processed': 1}
    assert report['processing_ms']['count'] == 1
    assert report['processing_ms']['p95'] >= 0
    assert report['sample_limit'] == 512
    assert not any(e['type'] in {'word', 'ack', 'sentence'} for e in active.poll())


def test_ack_contains_frame_processing_time_and_last_report_is_retained():
    active, _ = session()
    active.submit(packet(0)); active.finish()
    assert active.join(2)
    ack = next(e for e in active.poll() if e['type'] == 'ack')
    assert ack['processing_ms'] >= 0
    assert active.performance()['frames_completed'] == 1


def test_engine_timing_snapshots_do_not_double_count_across_resets():
    engine, _, events, _ = setup()
    for i in range(25): feed(engine, i)
    first = engine.performance()
    assert first == engine.performance()
    engine.finish()
    summary = next(e for e in events if e['type'] == 'summary')
    assert summary['timings'] == engine.performance()['timings']
    assert first['timings']['pose_ms']['count'] == 25
    assert summary['timings']['pose_ms']['count'] == 25
    engine.close()


def test_live_warmup_exercises_pose_even_when_detector_finds_no_person():
    calls = []
    extractor = RtmposeExtractor.__new__(RtmposeExtractor)
    extractor._last_boxes = ['stale']; extractor._frames_seen = 42
    def detector(image):
        calls.append(('det', image.shape))
        return np.empty((0, 4), np.float32)
    def pose(image, bboxes):
        calls.append(('pose', bboxes.copy()))
        return np.zeros((1, 133, 2)), np.zeros((1, 133))
    extractor._wholebody = SimpleNamespace(det_model=detector, pose_model=pose)
    report = extractor.warmup_live()
    assert len(calls) == 6
    assert all(row[1] == (320, 320, 3) for row in calls if row[0] == 'det')
    for _, box in (row for row in calls if row[0] == 'pose'):
        np.testing.assert_array_equal(box, [[0, 0, 319, 319]])
    assert report['iterations'] == 3 and report['pose_ms'] >= 0
    assert extractor._last_boxes is None and extractor._frames_seen == 0


def test_service_warms_on_worker_before_ready_then_releases_and_retains_report(tmp_path, monkeypatch):
    warmed = []; caches = []
    native = model()
    extractor = SimpleNamespace(runtime_info={'pose_device':'cuda'}, reset_detection=lambda: None,
                               warmup_live=lambda: warmed.append(threading.get_ident()) or {'pose_ms': 1.})
    @contextmanager
    def borrow(**kwargs):
        yield extractor
    native.live_pose_extractor = borrow
    class WarmRuntime(Runtime):
        vocabulary = ('<blank>', '我')
        def clear_frame_cache(self): caches.append(threading.get_ident())
    manager = ModelManager(initial_recognizer=native)
    service = service_module.RealtimeService(manager, config(tmp_path), runtime_factory=WarmRuntime)
    service.warmup()
    active = service.create_session(dict(source_epoch=1, fps=30))
    original_status = service.status
    def status_without_native_lock_inversion():
        # General model-status readers may already own the manager lock while
        # waiting for native runtime_info. Never request that lock from inside
        # the live extractor's native lease merely to assemble capabilities.
        assert threading.get_ident() not in warmed
        return original_status()
    monkeypatch.setattr(service, 'status', status_without_native_lock_inversion)
    active.start()
    seen = []
    wait(lambda: bool(seen.extend(active.poll()) or seen))
    assert next(e for e in seen if e['type'] == 'ready')['capabilities']['preparation']['pose']['pose_ms'] == 1.
    assert len(warmed) == 1 and warmed[0] != threading.get_ident()
    assert warmed[0] in caches
    active.cancel('test_done'); assert active.join(2)
    assert not service.status()['busy']
    assert service.status()['performance']['last']['reason'] == 'test_done'
    assert service.status()['performance']['last']['prepare_ms'] >= 0
    service.close(); manager.close()
