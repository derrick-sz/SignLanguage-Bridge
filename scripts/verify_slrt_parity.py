"""Compare the extracted inference path with unmodified upstream AST classes.

This is a source-equivalence check on real video/pose inputs, not a WER run.
Only training-irrelevant imports and K400 initialization are omitted in this
isolated verification namespace. No global module mocks are installed.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import logging
import math
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision

from slbridge.slrt.assets import SLRTConfig, sha256
from slbridge.slrt.backend import video_packets
from slbridge.slrt.runtime import SLRTRuntime
from slbridge.slrt.pose import make_frontend
from slbridge.slrt.preprocess import canonical_frame, prepare_inputs, select_keypoints


def upstream_classes(path,names,namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    exec(compile(tree,str(path),'exec'),namespace)


def main():
    p=argparse.ArgumentParser();p.add_argument('--video',type=Path,required=True)
    p.add_argument('--source',type=Path,default=Path('Paper-algorithm/SLRT-main/Online/CSLR'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--start-frame',type=int,default=0)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    c=SLRTConfig(pose_backend='rtmpose-experimental')
    r=SLRTRuntime(c);pose=make_frontend(c)
    packets=video_packets(args.video)
    frames=[];points=[]
    try:
        for packet in packets:
            if packet.frame_index < args.start_frame:continue
            kp=pose.extract(packet.image,frame_index=packet.frame_index,timestamp=packet.timestamp)
            if kp is None:
                continue
            frames.append(canonical_frame(packet.image));points.append(kp)
            if len(frames)==16:break
    finally:
        packets.close();pose.close()
    if len(frames)!=16:raise RuntimeError('Need 16 real signer frames')
    frames=np.stack(frames);points=np.stack(points)
    np.savez_compressed(args.output.with_suffix('.inputs.npz'),bgr=frames,keypoints=points,
                        metadata_json=json.dumps(dict(video_sha256=sha256(args.video),start_frame=args.start_frame,
                                                      frontend='rtmpose-experimental',coordinate_system='coco_wholebody_133_xyc_512_v1')))
    ns=dict(torch=torch,nn=nn,F=F,math=math,np=np,torchvision=torchvision,
            get_logger=lambda:logging.getLogger('slrt-upstream'),BLOCK2SIZE={1:64,2:192,3:480,4:832,5:1024})
    base=args.source/'modelling'
    source_names=['S3D_base.py','S3D.py','two_stream.py','fusion.py','Visualhead.py','recognition.py']
    upstream_classes(base/'S3D_base.py',{'S3D_base','BasicConv3d','SepConv3d','Mixed_3b','Mixed_3c','Mixed_4b','Mixed_4c','Mixed_4d','Mixed_4e','Mixed_4f','Mixed_5b','Mixed_5c'},ns)
    backbone_namespace=ns.copy()
    upstream_classes(base/'S3D.py',{'S3Ds','S3D_backbone'},backbone_namespace)
    original_backbone=backbone_namespace['S3D_backbone']
    def scratch_backbone(*a,**kw):
        kw['pretrained_ckpt']='scratch'
        return original_backbone(*a,**kw)
    ns['S3D_backbone']=scratch_backbone
    upstream_classes(base/'fusion.py',{'Lateral_Conn'},ns)
    upstream_classes(base/'two_stream.py',{'S3D_two_stream_v2'},ns)
    upstream_classes(base/'Visualhead.py',{'SepConvVisualHead'},ns)
    upstream_classes(args.source/'utils/gen_gaussian.py',{'gen_gaussian_hmap_op'},ns)
    upstream_classes(base/'recognition.py',{'RecognitionNetwork'},ns)
    original_pre=ns['RecognitionNetwork'].__new__(ns['RecognitionNetwork']);nn.Module.__init__(original_pre)
    original_pre.cfg={}
    original_pre.transform_cfg={'img_size':224,'center_crop':True,'center_crop_size':270}
    original_pre.heatmap_cfg={'input_size':112}
    original_pre.preprocess_chunksize=16
    bgr,kp=prepare_inputs(frames,points,device=r.torch_device)
    rgb_input=torch.from_numpy(frames[...,::-1].copy()).permute(0,3,1,2).unsqueeze(0).to(r.torch_device).float()/255
    coords=torch.from_numpy(select_keypoints(points)).to(r.torch_device)
    hmap=ns['gen_gaussian_hmap_op'](coords,raw_size=(512,512),map_size=(170,170),sigma=8).unsqueeze(0)
    orig_rgb,orig_hm,_,_=original_pre.augment_preprocess_inputs(False,rgb_input,hmap)
    original=nn.Module()
    original.recognition_network=nn.Module()
    rec=original.recognition_network
    rec.visual_backbone_twostream=ns['S3D_two_stream_v2'](use_block=5,freeze_block=(0,0),pose_inchannels=63,flag_lateral=(True,True),
        cfg_pyramid={'version':'v1','rgb':None,'pose':None},fusion_features=['c1','c2','c3','c4'],lateral_ksize=(1,3,3),
        lateral_ratio=(1,2,2),word_emb_tab=None,word_emb_dim=300,temp=.1,contras_setting=None)
    rec.visual_head=ns['SepConvVisualHead'](2001,input_size=1024,contras_setting=None,temp=.1)
    rec.visual_head_keypoint=ns['SepConvVisualHead'](2001,input_size=1024,contras_setting=None,temp=.1)
    rec.visual_head_fuse=ns['SepConvVisualHead'](2001,input_size=2048,contras_setting=None,temp=.1)
    original.load_state_dict(r.model.state_dict(),strict=True)
    original.to(r.torch_device).eval()
    with torch.inference_mode():
        f=rec.visual_backbone_twostream(orig_rgb,orig_hm)
        rgb_f,kp_f=f['rgb_fea_lst'][-1],f['pose_fea_lst'][-1]
        a=rec.visual_head(rgb_f)['gloss_logits']
        b=rec.visual_head_keypoint(kp_f)['gloss_logits']
        d=rec.visual_head_fuse(torch.cat([rgb_f,kp_f],dim=-1))['gloss_logits']
        expected=(a.softmax(-1)+b.softmax(-1)+d.softmax(-1))/3
        actual=r.model(bgr,kp)
    result=dict(status='passed',input='16 decoded real video frames and RTMPose experimental points',
        video_sha256=sha256(args.video),start_frame=args.start_frame,checkpoint_sha256=c.checkpoint_sha256,state_keys=r.state_keys,
        rgb_max_abs=float((bgr-orig_rgb).abs().max()),heatmap_max_abs=float((kp-orig_hm).abs().max()),
        probability_max_abs=float((actual-expected).abs().max()),top_id_actual=int(actual.argmax()),top_id_original=int(expected.argmax()),
        source_files={name:sha256(base/name) for name in source_names},
        excludes='Training/loss and K400 initialization, official HRNet extraction and historical dependency-version parity')
    ok=result['rgb_max_abs']<1e-6 and result['heatmap_max_abs']<1e-6 and result['probability_max_abs']<1e-6
    result['status']='passed' if ok else 'failed'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
    r.close()
    assert ok, 'Upstream parity failed; see recorded numerical differences'


if __name__=='__main__':main()
