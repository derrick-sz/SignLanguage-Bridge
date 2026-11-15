from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.slrt.assets import SLRTConfig
from slbridge.slrt.pose import RTMPoseExperimentalFrontend


def test_existing_protocol_stays_full_precision_and_detects_every_frame():
    c = SLRTConfig()
    assert c.precision == 'float32' and c.pose_detect_stride == 1
    changed = replace(c, precision='float16', pose_detect_stride=5)
    assert changed.to_dict()['pose_detect_stride'] == 5
    assert changed.to_dict()['precision'] == 'float16'


@pytest.mark.parametrize('value', [0, 31, True, 1.5])
def test_invalid_detection_interval_rejected(value):
    with pytest.raises(ValueError, match='pose_detect_stride'):
        SLRTConfig(pose_detect_stride=value)


def test_half_precision_never_silently_runs_as_cpu_or_unknown_precision():
    with pytest.raises(ValueError, match='precision'):
        SLRTConfig(precision='int8')
    with pytest.raises(ValueError, match='CUDA'):
        SLRTConfig(device='cpu', precision='float16')


def test_quality_loss_forgets_tracked_person_box():
    f = RTMPoseExperimentalFrontend.__new__(RTMPoseExperimentalFrontend)
    resets = []
    f.extractor = SimpleNamespace(extract=lambda image: (np.zeros((1, 133, 2)), np.zeros((1, 133))),
                                  reset_detection=lambda: resets.append(True))
    assert f.extract(np.zeros((512, 512, 3), np.uint8), frame_index=0, timestamp=0) is None
    assert resets == [True]


def test_frontend_uses_configured_detector_reuse(monkeypatch, tmp_path):
    import slbridge.slrt.pose as module
    import slbridge.uni_sign as uni
    captured = []

    class Extractor:
        def __init__(self, **kwargs):
            captured.append(kwargs)
            self.runtime_info = {}

    monkeypatch.setattr(module, 'require_file', lambda path, name: path)
    monkeypatch.setattr(module, 'sha256', lambda path: 'fixture-digest')
    monkeypatch.setattr(uni, 'RtmposeExtractor', Extractor)
    f = RTMPoseExperimentalFrontend(SLRTConfig(pose_detect_stride=5, rtmpose_cache_dir=str(tmp_path)))
    assert captured[0].get('detect_stride') == 5
    assert f.info['pose_detect_stride'] == 5
