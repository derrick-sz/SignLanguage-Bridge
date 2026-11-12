from pathlib import Path
import numpy as np

from scripts.validate_training_application import choose,write_clip,live_result


def test_selection_is_deterministic_balanced_and_obeys_the_duration_boundary():
    rows=[dict(sample_id=f'{i:02}-{s}',signer=s,start=5,end=205,source_present=True) for i in range(4) for s in ['a','b']]
    rows+=[dict(sample_id='00-x',signer='x',start=0,end=201,source_present=True),
           dict(sample_id='00-y',signer='y',start=0,end=100,source_present=False)]
    expected=choose(rows,4)
    assert expected==choose(list(reversed(rows)),4)
    assert [r['signer'] for r in expected]==['a','b','a','b']


def test_real_video_crop_uses_exclusive_end_and_preserves_clock(tmp_path):
    import cv2
    source=tmp_path/'source.mp4';target=tmp_path/'crop.mp4'
    writer=cv2.VideoWriter(str(source),cv2.VideoWriter_fourcc(*'mp4v'),25,(32,32));assert writer.isOpened()
    for i in range(8):writer.write(np.full((32,32,3),i*25,np.uint8))
    writer.release()
    info,_=write_clip(source,target,2,6)
    cap=cv2.VideoCapture(str(target));images=[]
    while True:
        ok,image=cap.read()
        if not ok:break
        images.append(image)
    cap.release()
    assert info['fps']==25 and info['frames']==len(images)==4
    assert images[0].shape==(512,512,3)
    assert abs(float(images[0].mean())-50)<12 and abs(float(images[-1].mean())-125)<12


def test_live_evaluation_removes_invalidated_tokens_and_keeps_completed_previous_sentence():
    data=dict(request_status='ok',source_capture_end_ms=1000,events=[
        dict(type='word',utterance_id=0,event_id='a',status='final',gloss='先',received_ms=500),
        dict(type='sentence',utterance_id=0,status='committed',text='上一句',received_ms=1100),
        dict(type='word',utterance_id=1,event_id='b',status='final',gloss='无效',received_ms=1200),
        dict(type='utterance_reset',utterance_id=1,reason='input_discontinuity'),
        dict(type='word',utterance_id=2,event_id='c',status='partial',gloss='未确认',received_ms=1300),
    ])
    result=live_result(data)
    assert result['predicted_gloss']==['先'] and result['text']=='上一句'
    assert result['extra_after_source_end_ms']==100
