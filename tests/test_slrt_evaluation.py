"""Public evaluation path checks; failed inputs cannot become model scores."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from scripts.evaluate_slrt_datasets import _MirroredFrontend, oriented_packet, summarize, validate_records
from slbridge._io import canonical_hash
from slbridge.slrt.cli import _wer
from slbridge.slrt.preprocess import mirror_keypoints_133
from slbridge.sources import FramePacket


def test_mirror_path_keeps_rgb_and_pose_in_the_same_orientation():
    frame=np.zeros((16,24,3),np.uint8);frame[:,2]=255
    kp=np.zeros((133,3),np.float32);kp[:,0]=np.arange(133);kp[:,2]=.7
    seen=[]
    class Inner:
        def extract(self,bgr,**kwargs):seen.append(bgr.copy());return kp.copy()
    packet=FramePacket(frame,0.,0,'test')
    mirrored=oriented_packet(packet,mirror=True)
    actual=_MirroredFrontend(Inner()).extract(mirrored.image,frame_index=0,timestamp=0.)
    assert actual.shape==(133,3)
    assert np.array_equal(seen[0],frame)
    assert np.array_equal(mirrored.image,frame[:,::-1])
    assert np.array_equal(actual,mirror_keypoints_133(kp[None])[0])


def test_failed_run_does_not_produce_corpus_wer_or_direction_conclusions(tmp_path):
    p=tmp_path/'detail-dev.jsonl'
    p.write_text(json.dumps(dict(sample_id='broken',reference=['A'],hypothesis=[],error='shape mismatch'))+'\n',encoding='utf-8')
    args=SimpleNamespace(split='dev',pose_backend='rtmpose-experimental',device='cpu',mirror_all=True,direction_table=None)
    summarize(p,tmp_path,args)
    x=json.loads((tmp_path/'summary-dev.json').read_text(encoding='utf-8'))
    assert x['failed_videos']==1 and x['successful_videos']==0
    assert x['corpus_wer'] is None and x['empty_output_videos']==0
    assert x['direction_table']=='all mirror (diagnostic)'


def protocol_and_record():
    inputs=[dict(sample_id=name,video_sha256=name+'-video',reference=['A'],direction='preserve')
            for name in ('one','two')]
    protocol=dict(static=dict(split='dev',inputs=inputs))
    protocol['protocol_sha256']=canonical_hash(protocol)
    record=dict(sample_id='one',split='dev',run_protocol_sha256=protocol['protocol_sha256'],
                video_sha='one-video',reference=['A'],hypothesis=['A'],direction='preserve',
                wer=_wer(['A'],['A']),reference_oov_count=0,translator='A',
                wall_seconds=1.,frames_received=20,error=None,status='final')
    return protocol,record


def test_partial_success_is_not_a_complete_evaluation(tmp_path):
    protocol,record=protocol_and_record()
    (tmp_path/'run-protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    detail=tmp_path/'detail-dev.jsonl'
    detail.write_text(json.dumps(record)+'\n',encoding='utf-8')
    args=SimpleNamespace(split='dev',pose_backend='rtmpose-experimental',device='cpu',mirror_all=False,direction_table=None)
    summarize(detail,tmp_path,args)
    summary=json.loads((tmp_path/'summary-dev.json').read_text(encoding='utf-8'))
    assert summary['evaluation_complete'] is False
    assert summary['corpus_wer'] is None
    assert summary['successful_subset_corpus_wer']==0.
    assert summary['missing_sample_ids']==['two'] and summary['expected_videos']==2


def test_complete_verified_manifest_can_publish_corpus_wer(tmp_path):
    protocol,record=protocol_and_record()
    second=dict(record,sample_id='two',video_sha='two-video')
    (tmp_path/'run-protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    detail=tmp_path/'detail-dev.jsonl'
    detail.write_text('\n'.join(json.dumps(r) for r in [record,second])+'\n',encoding='utf-8')
    args=SimpleNamespace(split='dev',pose_backend='rtmpose-experimental',device='cpu',mirror_all=False,direction_table=None)
    summarize(detail,tmp_path,args)
    summary=json.loads((tmp_path/'summary-dev.json').read_text(encoding='utf-8'))
    assert summary['evaluation_complete'] is True and summary['corpus_wer']==0.
    assert summary['missing_sample_ids']==[] and summary['coverage_verified'] is True


@pytest.mark.parametrize('field,value',[
    ('sample_id','orphan'),('split','test'),('run_protocol_sha256','wrong'),
    ('video_sha','wrong-video'),('reference',['B']),('direction','mirror'),
    ('wer',{'wer':0.5}),
])
def test_resume_rejects_records_from_a_different_protocol(field,value):
    protocol,record=protocol_and_record()
    record[field]=value
    with pytest.raises(ValueError):validate_records([record],protocol,split='dev')


def test_resume_rejects_duplicate_sample_ids_and_tampered_protocol():
    protocol,record=protocol_and_record()
    with pytest.raises(ValueError):validate_records([record,record],protocol,split='dev')
    protocol['static']['inputs'][0]['reference']=['B']
    with pytest.raises(ValueError):validate_records([],protocol,split='dev')


def test_bound_error_records_have_no_wer():
    protocol,record=protocol_and_record()
    record['error']='preprocessing failed'
    with pytest.raises(ValueError):validate_records([record],protocol,split='dev')
    record['wer']=None
    assert validate_records([record],protocol,split='dev')=={'one'}


def test_partial_or_unmarked_prediction_cannot_complete_bound_evaluation():
    protocol,record=protocol_and_record()
    for status in ('partial',None):
        record['status']=status
        with pytest.raises(ValueError):validate_records([record],protocol,split='dev')
    record.update(status='final',frames_received=1,windows_processed=0,
                  hypothesis=[],wer=_wer(['A'],[]))
    assert validate_records([record],protocol,split='dev')=={'one'}
