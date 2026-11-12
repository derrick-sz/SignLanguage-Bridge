from types import SimpleNamespace
import numpy as np
import pytest

from slbridge import realtime_engine
from slbridge.slrt import pose,stream
from slbridge.slrt.preprocess import canonical_frame
from slbridge.slrt.assets import SLRTConfig
from slbridge.sources import FramePacket


def test_live_frontend_reuses_exact_canonical_pixels_for_the_same_frame(monkeypatch):
    calls=[];windows=[];indices=[]
    def counted(image):calls.append(True);return canonical_frame(image)
    monkeypatch.setattr(pose,'canonical_frame',counted)
    monkeypatch.setattr(stream,'canonical_frame',counted)
    extractor=SimpleNamespace(runtime_info={'pose_device':'cpu'},_detect_stride=1,
        extract=lambda image:(np.full((1,133,2),90,np.float32),np.ones((1,133),np.float32)),reset_detection=lambda:None)
    config=SLRTConfig(device='cpu',pose_device='cpu',pose_backend='rtmpose-experimental',stride=4,vote_size=1)
    frontend=realtime_engine._PoseTap(pose.RTMPoseExperimentalFrontend.from_shared(config,extractor))
    class Runtime:
        vocabulary=('<blank>','test');info={}
        def predict(self,images,keypoints):
            windows.append(images.copy());return np.array([.01,.99]),{}
    decoder=stream.SLRTStream(Runtime(),frontend,config,window_sink=lambda e:indices.append(e['source_frame_indices']))
    rng=np.random.default_rng(73)
    images=[rng.integers(0,256,(512,512,3),np.uint8) for _ in range(13)]
    for i,image in enumerate(images):decoder.feed(FramePacket(image,i/30,i,'fixture'))
    assert len(calls)==len(images)
    for window,source_indices in zip(windows,indices):
        np.testing.assert_array_equal(window,np.stack([canonical_frame(images[i]) for i in source_indices]))
    assert frontend.last.shape==(133,3)
    decoder.close()


def test_an_invalid_prepared_image_cannot_silently_change_the_window_protocol():
    class Frontend:
        info={}
        def extract(self,*a,**kw):return np.ones((133,3),np.float32)
        def extract_prepared(self,*a,**kw):return np.zeros((16,16,3),np.uint8),np.ones((133,3),np.float32)
        def close(self):pass
    runtime=SimpleNamespace(vocabulary=('<blank>','test'),info={})
    decoder=stream.SLRTStream(runtime,Frontend(),SLRTConfig())
    with pytest.raises(ValueError,match='prepared'):
        decoder.feed(FramePacket(np.zeros((32,32,3),np.uint8),0,0,'fixture'))
