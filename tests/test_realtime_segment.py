import numpy as np

from slbridge import realtime_segment as m
from slbridge.sources import FramePacket


def pose(shift=0, high=False):
    kp=np.zeros((133,3),np.float32);kp[:,2]=1
    kp[5,:2]=[200,100];kp[6,:2]=[300,100]
    kp[91:,:2]=[200+shift,80 if high else 350]
    return kp


def feed(buffer,index,shift=0,high=False,words=True,blank=False):
    packet=FramePacket(np.zeros((32,32,3),np.uint8),index/10,index,'test')
    return buffer.observe(packet,pose(shift,high),has_words=words,word_changed=index==4,blank=blank)


def test_static_person_without_sign_activity_does_not_form_a_sentence():
    b=m.UtteranceBuffer(fps=10,pause_seconds=.8)
    for i in range(30):assert feed(b,i,words=False) is None
    assert b.take('manual') is None


def test_motion_then_blank_and_rest_produces_a_bounded_pose_segment():
    b=m.UtteranceBuffer(fps=10,pause_seconds=.8);segments=[]
    for i in range(25):
        result=feed(b,i,shift=min(i,6)*10,blank=i>8)
        if result:segments.append(result)
    assert len(segments)==1 and segments[0].reason=='pause'
    assert segments[0].samples[0].index < segments[0].samples[-1].index
    assert all(s.keypoints.shape==(133,3) for s in segments[0].samples)
    assert b.take('natural_eof') is None


def test_a_static_held_sign_is_not_committed_by_low_motion_alone():
    b=m.UtteranceBuffer(fps=10,pause_seconds=.8)
    for i in range(25):assert feed(b,i,shift=min(i,6)*10,high=True) is None
    segment=b.take('manual')
    assert segment is not None and segment.reason=='manual'


def test_overflow_never_translates_a_silently_truncated_utterance():
    b=m.UtteranceBuffer(fps=10,max_frames=16,max_seconds=20)
    for i in range(30):assert feed(b,i,shift=i*10,high=True) is None
    assert b.overflowed and b.take('natural_eof') is None


def test_frame_discontinuity_invalidates_the_unfinished_utterance():
    b=m.UtteranceBuffer(fps=10)
    for i in range(6):feed(b,i,shift=i*10,high=True)
    feed(b,15,shift=60,high=True)
    assert b.take('manual') is None
