"""Asset, decoder and source failure checks; no physical device access."""
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest

from slbridge.slrt.assets import SLRTConfig,SLRTResourceError,require_file,sha256
from slbridge.slrt.backend import video_packets
from slbridge.slrt.cli import _wer,main
from slbridge.slrt.pose import HRNetNPZFrontend,make_frontend
from slbridge.sources import SourceError


def test_hash_and_unknown_backend_fail_early(tmp_path):
    p=tmp_path/'asset';p.write_bytes(b'abc')
    with pytest.raises(SLRTResourceError,match='SHA256'):
        require_file(p,'test','0'*64)
    c=tmp_path/'cfg.json';c.write_text(json.dumps({'backend':'random_forest_v1'}))
    with pytest.raises(ValueError,match='backend'):SLRTConfig.from_file(c)


def test_config_paths_resolve_relative_to_config_file(tmp_path):
    p=tmp_path/'cfg.json';p.write_text(json.dumps({'resource_dir':'weights','rtmpose_cache_dir':'pose'}))
    c=SLRTConfig.from_file(p)
    assert c.checkpoint_path==tmp_path/'weights/ckpts/cslr_best.ckpt'
    assert Path(c.rtmpose_cache_dir)==tmp_path/'pose'


def test_default_hrnet_never_silently_uses_mediapipe():
    with pytest.raises(SLRTResourceError,match='detector_config'):
        make_frontend(SLRTConfig())


def test_hrnet_cache_binds_video_and_every_frame(tmp_path):
    video=tmp_path/'v.mp4';video.write_bytes(b'video-content')
    meta=dict(video_sha256=sha256(video),frontend='hrnet_w48_dark',coordinate_system='coco_wholebody_133_xyc_512_v1')
    cache=tmp_path/'kp.npz'
    np.savez(cache,keypoints=np.zeros((2,133,3),np.float32),frame_indices=np.arange(2),timestamps=np.arange(2)/30,
             metadata_json=json.dumps(meta))
    f=HRNetNPZFrontend(cache,video)
    assert f.extract(None,frame_index=0,timestamp=0) is None
    with pytest.raises(SLRTResourceError,match='timestamp'):f.extract(None,frame_index=1,timestamp=.5)
    video.write_bytes(b'changed-video')
    with pytest.raises(SLRTResourceError,match='SHA256'):HRNetNPZFrontend(cache,video)


@pytest.mark.parametrize('declared,actual',[(3,2),(2,3)])
def test_incomplete_or_extra_video_frames_raise_and_release(monkeypatch,declared,actual):
    class Capture:
        i=0;closed=False
        def isOpened(self):return True
        def get(self,prop):return {1:30.,2:declared,3:max(0,self.i-1)*1000/30}[prop]
        def read(self):
            self.i+=1
            return (True,np.zeros((8,8,3),np.uint8)) if self.i<=actual else (False,None)
        def release(self):self.closed=True
    c=Capture()
    monkeypatch.setitem(sys.modules,'cv2',SimpleNamespace(VideoCapture=lambda p:c,CAP_PROP_FPS=1,CAP_PROP_FRAME_COUNT=2,CAP_PROP_POS_MSEC=3))
    with pytest.raises(SourceError,match='count mismatch'):list(video_packets('any.mp4'))
    assert c.closed


def test_gloss_wer_is_token_level_and_keeps_repeats():
    assert _wer(['A','A'],['A'])==dict(wer=.5,substitutions=0,insertions=0,deletions=1,reference_tokens=2,scope='single_video_user_supplied_reference')
    assert _wer([],['A'])['wer'] is None


def test_hrnet_npz_shared_adapter_rejected_before_loading_model():
    from slbridge.slrt.adapter import SLRTAdapter
    with pytest.raises(SLRTResourceError,match='standalone'):
        SLRTAdapter(SLRTConfig(pose_backend='hrnet-npz'))


def test_adapter_warmup_checks_pose_before_model_activation(monkeypatch):
    from slbridge.slrt.adapter import SLRTAdapter
    adapter=SLRTAdapter.__new__(SLRTAdapter)
    adapter.config=SLRTConfig()
    monkeypatch.setattr(adapter,'_load',lambda:None)
    with pytest.raises(SLRTResourceError,match='detector_config'):adapter.warmup()


def test_cli_refuses_to_overwrite_any_source_even_on_error(tmp_path):
    p=tmp_path/'original.mp4';p.write_bytes(b'original-video-bytes')
    assert main(['video','--video',str(p),'--output',str(p)])==2
    assert p.read_bytes()==b'original-video-bytes'
    assert main(['video','--video',str(p),'--events-jsonl',str(p)])==2
    assert p.read_bytes()==b'original-video-bytes'
