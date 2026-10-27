from pathlib import Path
import json

import numpy as np
import pytest

from slbridge.model_adapters import model_adapter_ids
from slbridge.runtime_contracts import InferenceRequest
from slbridge.runtime_options import RuntimeOptions
from slbridge.uni_sign import (
    BACKEND_ID,
    REPRESENTATION_ID,
    SCORE_TYPE,
    TextStabilizer,
    UniSignAdapter,
    UniSignBackend,
    UniSignConfig,
)
from slbridge.uni_sign_assets import (
    MT5_FILES,
    MT5_REVISION,
    UNISIGN_CHECKPOINT_BYTES,
    UNISIGN_CHECKPOINT_SHA256,
    UNISIGN_REVISION,
    UNISIGN_SOURCE_COMMIT,
    FileCheck,
)
import slbridge.uni_sign as uni_sign_module
from slbridge.uni_sign_preprocess import prepare_pose_window


ZERO_SHA = '0' * 64


def config(tmp_path):
    root = tmp_path / 'uni_sign'
    bundle = root / 'model_bundle'
    bundle.mkdir(parents=True)
    return UniSignConfig(
        bundle_dir=bundle,
        checkpoint=root / 'csl_daily_pose_only_slt.pth',
        checkpoint_bytes=UNISIGN_CHECKPOINT_BYTES,
        checkpoint_sha256=UNISIGN_CHECKPOINT_SHA256,
        mt5_dir=root / 'mt5-base',
        source_dir=tmp_path / 'uni_sign_source',
        source_commit=UNISIGN_SOURCE_COMMIT,
        source_zip_sha256=ZERO_SHA,
        model_revision=UNISIGN_REVISION,
        mt5_revision=MT5_REVISION,
        run_id='uni-sign-csl-daily-pose-only-182913ca',
        display_name='Uni-Sign Pose-only（CSL-Daily，实验）',
        runtime={
            'device': 'cpu', 'dtype': 'float32', 'max_frames': 256,
            'num_beams': 4, 'max_new_tokens': 20, 'pose_mode': 'lightweight',
            'pose_backend': 'onnxruntime', 'pose_device': 'cpu',
            'rtmlib_cache_dir': '../rtmlib-cache', 'stabilize_after': 2,
            'max_video_frames': 600,
        },
        license={'model': 'CC-BY-NC-4.0'},
    )


def raw_pose(frames):
    points = np.zeros((frames, 1, 133, 2), dtype=np.float32)
    scores = np.ones((frames, 1, 133), dtype=np.float32)
    for frame in range(frames):
        points[frame, 0, 0] = [0.3, 0.3]
        points[frame, 0, 3:11] = np.linspace(0.2, 0.8, 16).reshape(8, 2)
        points[frame, 0, 91:112] = np.linspace(0.3, 0.6, 42).reshape(21, 2)
        points[frame, 0, 112:133] = np.linspace(0.4, 0.7, 42).reshape(21, 2)
        points[frame, 0, 23:40:2] = np.linspace(0.5, 0.7, 18).reshape(9, 2)
        points[frame, 0, 83:91] = np.linspace(0.6, 0.8, 16).reshape(8, 2)
        points[frame, 0, 53] = [0.5, 0.5]
    return points, scores


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, frame):
        self.calls += 1
        points, scores = raw_pose(1)
        return points, scores

    @property
    def runtime_info(self):
        return {'pose_device': 'cpu', 'pose_providers': ['CPUExecutionProvider']}

    def close(self):
        pass


class FakeRuntime:
    instances = []

    def __init__(self, config, *, device, dtype):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.loaded = False
        self.closed = False
        FakeRuntime.instances.append(self)

    def load(self):
        self.loaded = True

    def infer(self, pose, *, num_beams, max_new_tokens):
        return '同上。', {'model_forward_ms': 1.0, 'generation_ms': 2.0,
                          'decode_ms': .1, 'model_total_ms': 3.1}

    @property
    def runtime_info(self):
        return {'device': self.device, 'dtype': self.dtype,
                'providers': ['PyTorch CPU'], 'load_ms': 0.0}

    def close(self):
        self.closed = True


def test_uni_sign_adapter_is_registered_and_typed():
    assert BACKEND_ID in model_adapter_ids()
    assert REPRESENTATION_ID == 'uni_sign_rgb_v1'
    assert SCORE_TYPE == 'uncalibrated_text'


def test_pose_cache_reuses_overlapping_frames(tmp_path):
    backend = UniSignBackend(config(tmp_path), RuntimeOptions(device='cpu'))
    extractor = FakeExtractor()
    backend._extractor = extractor
    backend._loaded = True
    frames = [(np.zeros((16, 16, 3), np.uint8), index) for index in range(4)]
    pose, metrics = backend._prepare_frames(frames, source_id='camera:0')
    assert extractor.calls == 4
    overlapping = [(np.zeros((16, 16, 3), np.uint8), index) for index in range(2, 6)]
    pose, metrics = backend._prepare_frames(overlapping, source_id='camera:0')
    assert extractor.calls == 6
    assert metrics['pose_cache_hit_frames'] == 2
    assert metrics['pose_new_frames'] == 2
    assert pose.quality['keypoint_count'] == 133


def test_first_person_pose_uses_isotropic_short_side_and_reports_mode(tmp_path):
    cfg = config(tmp_path)
    backend = UniSignBackend(cfg, RuntimeOptions(device='cpu'))
    points = np.zeros((1, 133, 2), dtype=np.float32)
    points[0, 1] = [1080.0, 0.0]
    points[0, 2] = [0.0, 1080.0]
    normalized, _ = backend._first_person_pose(
        points, np.ones((1, 133), dtype=np.float32), width=1920, height=1080)
    assert np.allclose(normalized[1], [1.0, 0.0])
    assert np.allclose(normalized[2], [0.0, 1.0])
    assert backend.runtime_info['pose_normalization'] == 'isotropic_short_side'


def test_invalid_pose_normalization_mode_is_rejected(tmp_path):
    cfg = config(tmp_path)
    cfg.runtime['pose_normalization'] = 'unsupported'
    with pytest.raises(ValueError, match='pose_normalization'):
        UniSignBackend(cfg, RuntimeOptions(device='cpu'))


def test_pose_orientation_is_explicit_and_rejects_unknown_modes(tmp_path):
    cfg = config(tmp_path)
    assert UniSignBackend(cfg, RuntimeOptions(device='cpu')).runtime_info['pose_orientation'] == 'preserve'
    cfg.runtime['pose_orientation'] = 'mirror'
    assert UniSignBackend(cfg, RuntimeOptions(device='cpu')).runtime_info['pose_orientation'] == 'mirror'
    cfg.runtime['pose_orientation'] = 'auto'
    with pytest.raises(ValueError, match='pose_orientation'):
        UniSignBackend(cfg, RuntimeOptions(device='cpu'))


def test_text_stabilizer_temporary_confirmed_and_withdrawal():
    stabilizer = TextStabilizer(confirm_after=2)
    first = stabilizer.update(' 你好  世界 ')
    second = stabilizer.update('你好 世界')
    third = stabilizer.update('新的句子。')
    assert first['state'] == 'temporary' and first['duplicate'] is False
    assert second['state'] == 'confirmed' and second['duplicate'] is True
    assert third['state'] == 'temporary' and third['changed'] is True
    assert third['previous_text'] == '你好 世界'


def test_make_result_marks_stable_text_without_claiming_human_confirmation(tmp_path):
    backend = UniSignBackend(config(tmp_path), RuntimeOptions(device='cpu'))
    points, scores = raw_pose(3)
    pose = prepare_pose_window(points, scores)
    req1 = InferenceRequest('continuous-demo-1')
    req2 = InferenceRequest('continuous-demo-2')
    first = backend._make_result(request=req1, pose=pose, text='你好。', timings={'adapter_total_ms': 10.0},
                                 metrics={'window_frames': 3}, time_range=(0.0, 1.0))
    second = backend._make_result(request=req2, pose=pose, text='你好。', timings={'adapter_total_ms': 10.0},
                                  metrics={'window_frames': 3}, time_range=(1.0, 2.0))
    assert first.task_type == 'text' and first.status == 'recognized'
    assert first.quality['stability_state'] == 'temporary'
    assert second.quality['stability_state'] == 'confirmed'
    assert second.quality['continuous_candidate'] is True
    json.dumps(second.to_dict(), ensure_ascii=False)


def test_adapter_warmup_loads_torch_before_pose_extractor(tmp_path, monkeypatch):
    FakeRuntime.instances.clear()
    cfg = config(tmp_path)
    order = []
    monkeypatch.setattr(uni_sign_module, 'verify_file', lambda path, **kwargs: FileCheck(
        Path(path), True, UNISIGN_CHECKPOINT_BYTES, UNISIGN_CHECKPOINT_SHA256, True))
    monkeypatch.setattr(uni_sign_module, 'verify_mt5_dir', lambda path: {
        name: FileCheck(Path(path) / name, True, size, digest, True)
        for name, (size, digest) in MT5_FILES.items()
    })

    class Runtime(FakeRuntime):
        def load(self):
            order.append('runtime.load')
            super().load()

    def extractor_factory(**kwargs):
        order.append('extractor')
        return FakeExtractor()

    adapter = UniSignAdapter(cfg, RuntimeOptions(device='cpu'), runtime_factory=Runtime,
                             extractor_factory=extractor_factory)
    adapter.warmup()
    assert order == ['runtime.load', 'extractor']
    assert adapter.recognizer.loaded
    adapter.close()
    assert FakeRuntime.instances[-1].closed is True


def test_runtime_load_failure_releases_runtime(tmp_path, monkeypatch):
    cfg = config(tmp_path)
    monkeypatch.setattr(uni_sign_module, 'verify_file', lambda path, **kwargs: FileCheck(
        Path(path), True, UNISIGN_CHECKPOINT_BYTES, UNISIGN_CHECKPOINT_SHA256, True))
    monkeypatch.setattr(uni_sign_module, 'verify_mt5_dir', lambda path: {
        name: FileCheck(Path(path) / name, True, size, digest, True)
        for name, (size, digest) in MT5_FILES.items()
    })
    events = []

    class BadRuntime(FakeRuntime):
        def load(self):
            raise RuntimeError('load failed')

        def close(self):
            events.append('runtime.close')

    def extractor_factory(**kwargs):
        events.append('extractor.created')
        return FakeExtractor()

    adapter = UniSignAdapter(cfg, RuntimeOptions(device='cpu'), runtime_factory=BadRuntime,
                             extractor_factory=extractor_factory)
    try:
        adapter.warmup()
    except RuntimeError:
        pass
    else:
        raise AssertionError('warmup should fail')
    assert events == ['runtime.close']


@pytest.mark.parametrize('input_kind', ['file', 'captured_clip'])
@pytest.mark.parametrize('orientation', ['preserve', 'mirror'])
def test_public_inference_applies_direction_to_file_and_captured_video(tmp_path, monkeypatch, input_kind, orientation):
    cv2 = pytest.importorskip('cv2')
    from slbridge.sources import CapturedClip, FramePacket

    cfg = config(tmp_path)
    cfg.runtime['pose_orientation'] = orientation
    monkeypatch.setattr(uni_sign_module, 'verify_file', lambda path, **kwargs: FileCheck(
        Path(path), True, UNISIGN_CHECKPOINT_BYTES, UNISIGN_CHECKPOINT_SHA256, True))
    monkeypatch.setattr(uni_sign_module, 'verify_mt5_dir', lambda path: {
        name: FileCheck(Path(path) / name, True, size, digest, True)
        for name, (size, digest) in MT5_FILES.items()
    })

    class RecordingRuntime(FakeRuntime):
        def infer(self, pose, **kwargs):
            self.last_pose = pose
            return super().infer(pose, **kwargs)

    class VideoExtractor(FakeExtractor):
        def extract(self, frame):
            points, scores = super().extract(frame)
            return points[0], scores[0]  # RTMPose returns people,joints,xy, without T

    adapter = UniSignAdapter(cfg, RuntimeOptions(device='cpu'), runtime_factory=RecordingRuntime,
                             extractor_factory=lambda **kwargs: VideoExtractor())
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    if input_kind == 'file':
        source = tmp_path / 'direction.avi'
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 10, (32, 32))
        if not writer.isOpened():
            pytest.skip('MJPG encoder unavailable')
        for _ in range(3):
            writer.write(image)
        writer.release()
    else:
        source_id = 'camera:direction-test'
        source = CapturedClip(tuple(FramePacket(image, i / 10, i, source_id) for i in range(3)), source_id)
    try:
        result, _ = adapter.infer(source, InferenceRequest('direction-public-path'), input_kind=input_kind)
        assert result.status == 'recognized'
        assert result.quality['pose_orientation'] == orientation
        observed = FakeRuntime.instances[-1].last_pose
        # Independent anatomical example: source right wrist lies at (1,1).
        if orientation == 'mirror':
            assert np.allclose(observed.body[0, 7], [-1, 1, 1], atol=1e-6)
        else:
            assert np.allclose(observed.body[0, 8], [1, 1, 1], atol=1e-6)
    finally:
        adapter.close()


def test_windows_cuda_path_entries_are_not_duplicated(tmp_path):
    import os

    if os.name != "nt":
        pytest.skip("Windows CUDA DLL path handling")
    from slbridge.uni_sign import RtmposeExtractor

    dll_dir = tmp_path / "dll"
    dll_dir.mkdir()
    extractor = RtmposeExtractor.__new__(RtmposeExtractor)
    extractor._dll_handles = []
    extractor.preload_failures = []
    segments_before = os.environ.get("PATH", "").split(os.pathsep).count(str(dll_dir))
    extractor._configure_windows_cuda((str(dll_dir),))
    extractor._configure_windows_cuda((str(dll_dir),))
    segments_after = os.environ.get("PATH", "").split(os.pathsep).count(str(dll_dir))
    assert segments_after == segments_before + 1

