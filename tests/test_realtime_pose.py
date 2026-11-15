from collections import OrderedDict
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.runtime import InferenceRequest, ModelManager, RuntimeBusy
from slbridge.slrt.assets import SLRTConfig
from slbridge.slrt.pose import RTMPoseExperimentalFrontend
from slbridge.uni_sign import UniSignBackend
from slbridge.uni_sign_preprocess import normalize_rtmpose_pixels, prepare_pose_window
from tests.test_runtime import Model


def bare_backend():
    backend=UniSignBackend.__new__(UniSignBackend)
    backend._lock=threading.RLock();backend._loaded=True;backend._cache=OrderedDict()
    backend.pose_normalization='isotropic_short_side';backend.pose_orientation='preserve'
    backend.max_frames=256;backend.num_beams=2;backend.max_new_tokens=128
    return backend


def test_native_lease_pins_a_model_until_all_native_work_has_returned():
    model=Model('owned');manager=ModelManager(initial_recognizer=model)
    with manager.native_session(expected_model=model) as active:
        assert active is model
        with pytest.raises(RuntimeBusy):manager.run(lambda m:None)
        manager.close()
        assert model.closed==0
    assert model.closed==1


def test_borrowed_pose_extractor_is_restored_even_after_a_failure():
    backend=bare_backend();resets=[]
    extractor=SimpleNamespace(_detect_stride=1,reset_detection=lambda:resets.append(True))
    backend._extractor=extractor;backend._cache['old']=True
    with pytest.raises(RuntimeError):
        with backend.live_pose_extractor(detect_stride=5) as shared:
            assert shared is extractor and extractor._detect_stride==5
            assert not backend._cache
            raise RuntimeError('fixture')
    assert extractor._detect_stride==1 and len(resets)==2


def test_frontend_never_closes_an_extractor_owned_by_unisign():
    closed=[];resets=[]
    extractor=SimpleNamespace(runtime_info={'pose_device':'cuda'},
         close=lambda:closed.append(True),reset_detection=lambda:resets.append(True))
    frontend=RTMPoseExperimentalFrontend.from_shared(SLRTConfig(pose_device='cuda'),extractor)
    frontend.close()
    assert closed==[] and resets


def test_pose_translation_preserves_existing_geometry_and_does_not_run_pose_extraction():
    backend=bare_backend();seen=[]
    backend._runtime=SimpleNamespace(infer=lambda pose,**kwargs:(seen.append(pose) or '测试句子',{'forward_ms':1}))
    points=np.arange(4*133*2,dtype=np.float32).reshape(4,133,2)%300
    scores=np.ones((4,133),np.float32)
    expected=prepare_pose_window(normalize_rtmpose_pixels(points,640,360),scores)
    text,details=backend.translate_pose_arrays(points,scores,width=640,height=360,request=InferenceRequest('pose-test'))
    assert text=='测试句子' and details['shared_pose'] is True
    for part in expected.parts:
        np.testing.assert_array_equal(expected.parts[part],seen[0].parts[part])


def test_pose_translation_does_not_generate_text_from_invalid_body_frames():
    backend=bare_backend()
    backend._runtime=SimpleNamespace(infer=lambda *a,**k:pytest.fail('Invalid pose must not reach generation'))
    text,details=backend.translate_pose_arrays(np.zeros((4,133,2)),np.zeros((4,133)),
                         width=512,height=512,request=InferenceRequest('pose-test'))
    assert not text and details['rejected_reason']=='insufficient_body_frames'
