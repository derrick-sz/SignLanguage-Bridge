"""Scheduling tests use deterministic probabilities, never model-accuracy claims."""
from dataclasses import replace
import numpy as np
import pytest

from slbridge.slrt.assets import SLRTConfig
from slbridge.slrt.stream import SLRTStream
from slbridge.sources import FramePacket


class Runtime:
    vocabulary = ('<blank>','A','B')
    info = {'test_double':True}
    def __init__(self):
        self.windows = []
    def predict(self, rgb, kp):
        self.windows.append(rgb[:,0,0,0].tolist())
        return np.array([.05,.9,.05],np.float32),dict(preprocess_ms=1.,inference_ms=1.,window_total_ms=2.)


class Pose:
    info = {'test_double':True}
    closed = False
    missing = False
    def extract(self,*args,**kwargs):
        return None if self.missing else np.zeros((133,3),np.float32)
    def close(self):
        self.closed = True


def packet(i, fps=30., source='test'):
    return FramePacket(np.full((16,16,3),i % 255,np.uint8),i/fps,i,source)


def test_no_future_access_and_full_eos_coverage():
    r,p = Runtime(),Pose()
    windows = []
    s = SLRTStream(r,p,SLRTConfig(vote_size=1),window_sink=windows.append)
    for i in range(8):
        assert s.feed(packet(i)) == []
    assert r.windows == []
    s.feed(packet(8))
    assert r.windows[0] == [0]*8+list(range(1,9))
    assert windows[0]['available_time'] == 8/30
    for i in range(9,40):
        s.feed(packet(i))
        assert all(max(w['source_frame_indices']) <= i for w in windows)
        assert len(s._frames) <= 16
    s.flush()
    assert len(r.windows) == 40
    assert all(w['right_edge_repeats'] == 0 for w in windows if not w['eos_padding'])
    assert s.result()['gloss_sequence'] == ['A']
    assert s.flush() == []
    with pytest.raises(RuntimeError):
        s.feed(packet(40))
    s.close(); s.close()
    assert p.closed


@pytest.mark.parametrize('fps',[25.,29.97,30.,60.])
def test_frame_count_readiness_at_actual_fps(fps):
    r,p = Runtime(),Pose()
    s = SLRTStream(r,p,SLRTConfig(vote_size=1,stride=4))
    for i in range(48): s.feed(packet(i,fps))
    assert len(r.windows) == 10
    s.flush()
    assert len(r.windows) == 12


def test_drop_pause_reset_and_missing_person_do_not_commit_partial():
    r,p = Runtime(),Pose()
    s = SLRTStream(r,p,SLRTConfig(vote_size=1))
    for i in range(9): s.feed(packet(i))
    assert s.result()['partial']['gloss'] == 'A'
    e = s.feed(packet(11))
    assert e[0]['status'] == 'retracted'
    assert s.result()['gloss_sequence'] == [] and s.result()['dropped_frames'] == 2
    s.pause(); s.resume()
    for i in range(20,29): s.feed(packet(i))
    p.missing = True
    assert s.feed(packet(29))[0]['status'] == 'retracted'
    assert s.flush() == []
    assert s.result()['gloss_sequence'] == []
    s.reset()
    assert s.result()['frames_received'] == 0 and s.result()['partial'] is None


def test_source_or_time_changes_require_explicit_reset():
    s = SLRTStream(Runtime(),Pose(),SLRTConfig())
    s.feed(packet(0))
    with pytest.raises(ValueError): s.feed(packet(1,source='other'))
    with pytest.raises(ValueError): s.feed(packet(0))


def test_zero_signer_produces_no_predictions():
    r,p = Runtime(),Pose(); p.missing = True
    s = SLRTStream(r,p,SLRTConfig())
    for i in range(64): s.feed(packet(i))
    assert s.flush() == [] and r.windows == []
    assert s.result()['quality_rejected_frames'] == 64
