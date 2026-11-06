import numpy as np
import pytest

from slbridge.animation.gesture import refine_gestures, motion_metrics, GESTURE_PROFILE


def poses(count=15):return np.zeros((count,55,3),dtype=np.float64)


@pytest.mark.parametrize('span',[1,2])
def test_isolated_hand_island_repaired_and_recorded(span):
    source=poses();source[6:6+span,25,0]=np.radians(150)
    snapshot=source.copy();result,audit,mask=refine_gestures(source,30)
    assert np.linalg.norm(result[6:6+span,25],axis=-1).max()<1e-7
    assert int(mask.sum())==span
    assert audit['isolated_repair_events'][0]['frames']==span
    assert audit['profile']==GESTURE_PROFILE
    np.testing.assert_array_equal(source,snapshot)
    np.testing.assert_array_equal(result[[0,-1]],source[[0,-1]])


def test_fast_continuous_rotation_and_sustained_handshape_preserved():
    source=poses();source[:,25,2]=np.arange(len(source))*0.3
    result,audit,mask=refine_gestures(source,30)
    assert not mask.any()
    assert audit['regular_correction_max_degrees']<1e-5
    held=poses(25);held[7:18,25,0]=np.radians(120)
    result,audit,mask=refine_gestures(held,30)
    assert not mask.any()
    np.testing.assert_allclose(result[10:15],held[10:15],atol=1e-7)


def test_wrap_and_undriven_jaw_do_not_trigger_repairs():
    source=poses(7);source[:,25,2]=np.radians([174,176,178,180,-178,-176,-174]);source[3,22]=[3,1,0]
    result,audit,mask=refine_gestures(source,30)
    assert not mask.any()
    np.testing.assert_array_equal(result[:,22],source[:,22])
    assert audit['regular_correction_max_degrees']<1e-5


def test_wrist_outlier_between_distinct_nearby_anchors_preserves_net_motion():
    source=poses();source[7:,20,0]=np.radians(54);source[6,20,0]=np.radians(-150)
    result,audit,mask=refine_gestures(source,30)
    assert mask[6,20]
    assert abs(np.degrees(result[6,20,0])-27)<1e-6
    np.testing.assert_array_equal(result[[0,-1]],source[[0,-1]])


def test_non_isolated_changes_bounded_and_only_mapped_channels_change():
    rng=np.random.default_rng(41);source=rng.normal(0,0.15,(20,55,3))
    result,audit,mask=refine_gestures(source,30)
    assert audit['regular_finger_correction_max_degrees']<=3.000001
    assert audit['regular_body_correction_max_degrees']<=5.000001
    np.testing.assert_array_equal(result[:,:12],source[:,:12])
    np.testing.assert_array_equal(result[:,22:25],source[:,22:25])
    assert motion_metrics(result,30)['mapped_step_events_over60']<=motion_metrics(source,30)['mapped_step_events_over60']


def test_unbracketed_fast_transition_is_preserved_for_review():
    source=poses();source[7:,20,0]=np.radians(110)
    result,audit,mask=refine_gestures(source,30)
    assert not mask.any()
    np.testing.assert_allclose(result[6:8,20],source[6:8,20],atol=1e-8)


@pytest.mark.parametrize('source,fps',[(np.zeros((2,54,3)),30),(np.full((3,55,3),np.nan),30),(poses(),0)])
def test_invalid_refinement_input_rejected(source,fps):
    with pytest.raises(ValueError):refine_gestures(source,fps)
