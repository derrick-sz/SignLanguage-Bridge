"""Generate new CSL text candidates without target motions or dataset loading."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slbridge.animation.signspark_generation import (
    GenerationRequest,MODEL_FILES,MODEL_REVISION,SOURCE_COMMIT,
    STREAM_WIDTHS,source_free_condition,verified_file,load_strict_checkpoint,
)
from slbridge.animation.signspark_presentation import (
    DEFAULT_SOURCE_PRESENTATION,verify_presentation_inputs,opt_in_generated_jaw,build_avatar_presentation,
)

# Same verified original 100-clip avatar archive used by the SOKE builders.
BASE_ARCHIVE_SHA256='4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5'
# Exact GitHub-connector receipt for the reviewed fixed source commit. A
# self-declared commit plus replaceable per-file hashes is not authentication.
SOURCE_BUNDLE_SHA256='880b46ee3e46aba550e4975eaa746ad0d18ac82a2a2bd7d05180fa5e6cffbad6'
# Nine official config/tokenizer files, checked against HF tree blob/LFS hashes.
TEXT_ASSETS_MANIFEST_SHA256='3f7c359cbf06363e8054bcc1a901d09fbe6b4b2c8bb7e01309e1f8a383267f2b'
# Portable projections of those authenticated receipts. Legacy identities remain
# accepted for evidence replay; neither format trusts a self-declared digest.
DEFAULT_SOURCE_MANIFEST=ROOT/'configs/signspark/source-manifest.json'
DEFAULT_TEXT_ASSETS_MANIFEST=ROOT/'configs/signspark/text-assets-manifest.json'
COMPACT_SOURCE_MANIFEST_SHA256='0883d912770c8ebc3d59cb66c82c6a228b95b31a250752d90205a963f93b252d'
COMPACT_TEXT_ASSETS_MANIFEST_SHA256='e27b5393be1d295c36ba805b118712c7c85b4b9a8d951b933bcba09801e058d7'
TEXT_REPOSITORIES={
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus':('mclip','15c3aee9b9e96fd7de1c1d488810a604bab98e78'),
    'FacebookAI/xlm-roberta-large':('xlm-roberta-large','c23d21b0620b635a76227c604d44e43a9f0ee389'),
}


def json_write(path,value):
    temp=path.with_suffix('.tmp.json')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def memory_snapshot(torch_module):
    """Observe bytes without initializing CUDA or changing allocator state.

    Windows counters use the documented PROCESS_MEMORY_COUNTERS_EX and
    PERFORMANCE_INFORMATION layouts (SIZE_T fields retain pointer width).
    Observation errors are recorded, never substituted with a zero reading.
    """
    result={'pid':os.getpid(),'time_unix':time.time()}
    if os.name=='nt':
        try:
            import ctypes
            from ctypes import wintypes
            size=ctypes.c_size_t
            class ProcessMemory(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[
                    (name,size) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                    'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                    'PagefileUsage','PeakPagefileUsage','PrivateUsage')]
            class PerformanceInfo(ctypes.Structure):
                _fields_=[('cb',wintypes.DWORD)]+[(name,size) for name in (
                    'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
                    'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[
                    (name,wintypes.DWORD) for name in ('HandleCount','ProcessCount','ThreadCount')]
            kernel=ctypes.WinDLL('kernel32',use_last_error=True)
            psapi=ctypes.WinDLL('psapi',use_last_error=True)
            kernel.GetCurrentProcess.argtypes=[];kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemory),wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
            psapi.GetPerformanceInfo.argtypes=[ctypes.POINTER(PerformanceInfo),wintypes.DWORD]
            psapi.GetPerformanceInfo.restype=wintypes.BOOL
            process=ProcessMemory();process.cb=ctypes.sizeof(process)
            if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(process),process.cb):
                raise ctypes.WinError(ctypes.get_last_error())
            result['process']={'rss_bytes':process.WorkingSetSize,'peak_rss_bytes':process.PeakWorkingSetSize,
                'pagefile_bytes':process.PagefileUsage,'peak_pagefile_bytes':process.PeakPagefileUsage,
                'private_bytes':process.PrivateUsage}
            host=PerformanceInfo();host.cb=ctypes.sizeof(host)
            if not psapi.GetPerformanceInfo(ctypes.byref(host),host.cb):raise ctypes.WinError(ctypes.get_last_error())
            result['host']={'committed_bytes':host.CommitTotal*host.PageSize,
                'commit_limit_bytes':host.CommitLimit*host.PageSize,'peak_committed_bytes':host.CommitPeak*host.PageSize,
                'available_bytes':host.PhysicalAvailable*host.PageSize}
        except Exception as exc:
            result['host_process_error']=type(exc).__name__+': '+str(exc)
    else:result['host_process_status']='Windows counters unavailable on this platform'
    try:
        cuda=torch_module.cuda;result['cuda']={'initialized':bool(cuda.is_initialized())}
        if result['cuda']['initialized']:
            result['cuda'].update(allocated_bytes=cuda.memory_allocated(),reserved_bytes=cuda.memory_reserved(),
                                  peak_allocated_bytes=cuda.max_memory_allocated())
            free,total=cuda.mem_get_info()
            result['cuda'].update(free_bytes=free,total_bytes=total)
    except Exception as exc:
        result.setdefault('cuda',{})['error']=type(exc).__name__+': '+str(exc)
    return result


def record_memory(manifest,report,torch_module,stream,stage,**details):
    snapshot=memory_snapshot(torch_module)
    snapshot.update(stream=stream,stage=stage,**details)
    manifest.setdefault('memory_snapshots',[]).append(snapshot)
    manifest['active_step']={'stream':stream,'stage':stage}
    json_write(report,manifest)


def save_raw_stream(output,request_index,role,values):
    """Publish each completed native stream before later inference/codec work."""
    if type(request_index) is not int or request_index<0 or role not in ('body','left','right','face'):
        raise ValueError('Invalid raw stream identity')
    target=output/f'raw-stream-{request_index}-{role}.npz'
    if target.exists():raise ValueError('Raw stream evidence already exists')
    temporary=target.with_suffix('.tmp.npz')
    with temporary.open('xb') as stream:
        np.savez_compressed(stream,values=values)
    os.replace(temporary,target)
    return {'path':target.name,'request_index':request_index,'role':role,
            'shape':list(values.shape),'dtype':str(values.dtype),
            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}


def receipt_path(root,relative):
    if (not isinstance(relative,str) or not relative or '\\' in relative or ':' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts):
        raise ValueError('Invalid authenticated asset path')
    root=Path(root).resolve();path=(root/relative).resolve()
    if path==root or not path.is_relative_to(root):raise ValueError('Invalid authenticated asset path')
    return path


def verify_source(source,manifest_path):
    try:
        receipt=manifest_path.read_bytes()
        manifest=json.loads(receipt)
    except (OSError,ValueError) as exc:raise ValueError('Pinned source manifest unavailable') from exc
    digest=hashlib.sha256(receipt).hexdigest()
    if digest not in (SOURCE_BUNDLE_SHA256,COMPACT_SOURCE_MANIFEST_SHA256):
        raise ValueError('Pinned source manifest SHA256 mismatch')
    if manifest.get('commit')!=SOURCE_COMMIT:raise ValueError('Pinned source manifest revision differs')
    if manifest.get('repository')!='JianHe0628/SignSparK':raise ValueError('Pinned source repository differs')
    if not isinstance(manifest.get('files'),list) or not manifest['files']:raise ValueError('Empty source receipt')
    for item in manifest['files']:
        path=receipt_path(source,item['path'])
        if type(item.get('size')) is not int or path.stat().st_size!=item['size']:
            raise ValueError('Pinned source file changed: '+item['path'])
        raw=path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()!=item['blob_sha']:
            raise ValueError('Pinned source file changed: '+item['path'])
    return digest


def verify_base_archive(path):
    from slbridge.animation.registry import MotionRegistry
    return MotionRegistry(path,BASE_ARCHIVE_SHA256).sha256


def verify_text_assets(root):
    root=Path(root).resolve();local_receipt=root/'manifest.json'
    receipt=(local_receipt if local_receipt.exists() else DEFAULT_TEXT_ASSETS_MANIFEST).read_bytes()
    digest=hashlib.sha256(receipt).hexdigest()
    if digest not in (TEXT_ASSETS_MANIFEST_SHA256,COMPACT_TEXT_ASSETS_MANIFEST_SHA256):
        raise ValueError('Pinned text asset manifest SHA256 mismatch')
    manifest=json.loads(receipt)
    if not isinstance(manifest.get('files'),list) or not manifest['files']:raise ValueError('Empty text asset receipt')
    for item in manifest['files']:
        repository=TEXT_REPOSITORIES.get(item.get('repo'))
        if repository is None or item.get('revision')!=repository[1]:
            raise ValueError('Unknown text asset repository or revision')
        # Authenticate the receipt as before, but never use its old local_path.
        path=receipt_path(root/repository[0],item['path'])
        if (not path.is_relative_to(root) or type(item.get('bytes')) is not int
                or not 0<item['bytes']<=32*1024*1024 or path.stat().st_size>32*1024*1024):
            raise ValueError('Invalid local text asset path/budget')
        verified_file(path,item['sha256'],item['bytes'])
    return digest


def cold_model(stream,text_assets):
    """Construct identical registered modules; all weights must come from EMA."""
    import torch
    from transformers import AutoConfig,AutoModel,AutoTokenizer
    from signspark.unet_model import Wilor_UNET
    from signspark.rflow import RFlow
    config=json.loads((text_assets/'mclip/config.json').read_text(encoding='utf-8'))
    if (config.get('modelBase')!='xlm-roberta-large' or config.get('transformerDimensions')!=1024
            or config.get('numDims')!=640):raise ValueError('Unexpected pinned MCLIP architecture')
    base=AutoConfig.from_pretrained(str(text_assets/'xlm-roberta-large'),local_files_only=True)

    class ColdWilor(Wilor_UNET):
        def load_and_freeze_clip(self,version):
            wrapper=torch.nn.Module()
            # The author EMA includes these frozen tensors. Preserve default
            # pooler and exact module names, then require strict complete load.
            wrapper.transformer=AutoModel.from_config(base,add_pooling_layer=True)
            wrapper.LinearTransformation=torch.nn.Linear(1024,640)
            tokenizer=AutoTokenizer.from_pretrained(str(text_assets/'mclip'),local_files_only=True)
            return self.freeze_params(wrapper),tokenizer

    model=ColdWilor(njoints=STREAM_WIDTHS[stream],nfeats=1,latent_dim=512,
                    dim_mults=(2,2,2,2),attention=False,dataset=stream,arch='unet_large',
                    unet_out_mult=8,keyframe_conditioned=True,length_mask_conditioned=True,
                    text_conditioned=True,text_mask_prob=.1,text_enc_name='CLIP',text_enc_dim=640,
                    text_enc_ver='M-CLIP/XLM-Roberta-Large-Vit-B-16Plus')
    return model,RFlow()


def empty_model(stream,text_assets):
    """Keep the author architecture and CPU buffers; omit initial CPU weights."""
    from accelerate import init_empty_weights
    with init_empty_weights(include_buffers=False):
        model,flow=cold_model(stream,text_assets)
    parameters=list(model.parameters())
    if not parameters or any(not value.is_meta for value in parameters):
        raise ValueError('Expected all meta parameters from empty model construction')
    if any(value.device.type!='cpu' for value in model.buffers()):
        raise ValueError('Expected real CPU buffers from empty model construction')
    return model,flow


def sample(model,flow,request,stream,seed_offset=0,observation=None):
    import torch
    data=source_free_condition(request,stream)
    condition={'obs_x0':torch.as_tensor(data['obs_x0'],device='cuda'),
               'obs_mask':torch.as_tensor(data['obs_mask'],device='cuda'),
               'y':{k:torch.as_tensor(v,device='cuda') if isinstance(v,np.ndarray) else v
                    for k,v in data['y'].items()}}
    if observation is not None:
        if observation.shape!=condition['obs_x0'].shape or not torch.isfinite(observation).all():
            raise ValueError('Invalid finite observation negative control')
        condition['obs_x0']=observation
    assert not condition['obs_mask'].any()
    seed=(request.seed+seed_offset)%(2**32)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    noise=torch.randn_like(condition['obs_x0'])
    with torch.inference_mode():
        result,_=flow.decode(model,noise=noise,keyframe_mask=condition['obs_mask'],
                            x_embed=condition['obs_x0'],model_kwargs=condition,
                            ode_package='torchdiffeq',ode_stepnum=request.steps)
    if result.shape!=noise.shape or not torch.isfinite(result).all():
        raise ValueError('Generated stream has invalid shape or nonfinite values')
    return result[0,:,:request.frames].T.detach().cpu().numpy().copy()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--text',action='append',required=True,help='Repeat for separate new sentences')
    parser.add_argument('--frames',type=int,default=128,help='Explicit output length; not learned from text')
    parser.add_argument('--steps',type=int,default=50)
    parser.add_argument('--seed',type=int,default=123)
    parser.add_argument('--source',type=Path,default=Path('G:/Archive/Zenodo/SignSparK/source'))
    parser.add_argument('--source-manifest',type=Path,default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument('--text-assets',type=Path,default=Path('G:/Archive/Zenodo/SignSparK/text-assets'))
    parser.add_argument('--models',type=Path,default=Path('G:/Archive/Zenodo/SignSparK/segmented-checkpoints'))
    parser.add_argument('--base-archive',type=Path,default=Path('E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip'))
    parser.add_argument('--source-presentation',type=Path,default=DEFAULT_SOURCE_PRESENTATION)
    parser.add_argument('--smplx-model',type=Path,default=Path('G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        requests=[GenerationRequest(t,args.frames,args.steps,args.seed) for t in args.text]
        if len(requests)>8 or len(set(args.text))!=len(args.text):raise ValueError('Use one to eight distinct requests')
        if args.output.exists():raise ValueError('Use a new output directory to preserve evidence')
        base_sha=verify_base_archive(args.base_archive)
        source_sha=verify_source(args.source,args.source_manifest)
        text_sha=verify_text_assets(args.text_assets)
        presentation_inputs=verify_presentation_inputs(smplx_model=args.smplx_model,source_presentation=args.source_presentation)
        for stream,(size,_) in MODEL_FILES.items():
            path=args.models/stream/'ema_0.9999_200000.pt'
            if not path.is_file() or path.stat().st_size!=size:
                raise ValueError('Verified pretrained checkpoint not ready: '+stream)
    except (OSError,ValueError,KeyError) as exc:parser.error(str(exc))

    os.environ['WANDB_MODE']='disabled';os.environ['HF_HUB_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    sys.path.insert(0,str(args.source.resolve()))
    import torch
    if not torch.cuda.is_available():parser.error('This bounded worker requires the validated CUDA runtime')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    from slbridge.animation.signspark_codec import decode_signspark_streams,CODEC_VERSION
    from slbridge.animation.gesture import refine_gestures
    from slbridge.animation.soke import write_candidate_archive
    from slbridge.animation.registry import MotionRegistry

    args.output.mkdir(parents=True)
    manifest={'schema':'custom-csl-signspark-generation-v1','state':'running',
              'source_commit':SOURCE_COMMIT,'source_manifest_sha256':source_sha,
              'base_archive_sha256':base_sha,
              'presentation_inputs':presentation_inputs,
              'model_revision':MODEL_REVISION,'text_assets_manifest_sha256':text_sha,
              'requests':[asdict(r) for r in requests],'streams':{},'raw_streams':[],'codec':CODEC_VERSION,
              'parameter_initialization':'accelerate-meta-parameters-cpu-buffers-v1',
              'checkpoint_loads':{},'memory_snapshots':[],
              'target_motion':None,'observed_keyframes':0,'length_source':'explicit request',
              'text_guidance_scale':-1,'language_review':'NOT_RUN','motion_review':'NOT_RUN'}
    report=args.output/'generation-manifest.json';json_write(report,manifest)
    generated=[{} for _ in requests]
    try:
        for stream in ('body','hand','face'):
            start=time.monotonic();size,digest=MODEL_FILES[stream]
            checkpoint=args.models/stream/'ema_0.9999_200000.pt'
            # Authenticate before constructing the large architecture. Loader
            # independently checks again to protect the subsequent deserialization.
            verified_file(checkpoint,digest,size)
            record_memory(manifest,report,torch,stream,'before_construct')
            model,flow=empty_model(stream,args.text_assets)
            record_memory(manifest,report,torch,stream,'after_construct')
            load_report=load_strict_checkpoint(model,checkpoint,expected_sha256=digest,expected_bytes=size,assign=True)
            manifest['checkpoint_loads'][stream]=load_report.copy()
            record_memory(manifest,report,torch,stream,'after_load')
            torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            record_memory(manifest,report,torch,stream,'before_cuda')
            model.eval().requires_grad_(False).to('cuda')
            record_memory(manifest,report,torch,stream,'after_cuda')
            roles=[('left',1),('right',2)] if stream=='hand' else [(stream,0 if stream=='body' else 3)]
            for role,offset in roles:
                for index,request in enumerate(requests):
                    generated[index][role]=sample(model,flow,request,stream,offset)
                    manifest['raw_streams'].append(save_raw_stream(args.output,index,role,generated[index][role]))
                    json_write(report,manifest)
            if stream=='body':
                unrelated=torch.randn((1,STREAM_WIDTHS[stream],304),device='cuda')*3
                control=sample(model,flow,requests[0],stream,observation=unrelated)
                difference=float(np.max(np.abs(control-generated[0]['body'])))
                if difference>1e-5:raise ValueError('Masked observation negative control changed generated output')
                manifest['masked_observation_control_max_difference']=difference
            torch.cuda.synchronize()
            load_report.update(seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
            manifest['streams'][stream]=load_report;json_write(report,manifest)
            print(json.dumps({'stream':stream,**load_report}),flush=True)
            del model,flow;gc.collect();torch.cuda.empty_cache()
            record_memory(manifest,report,torch,stream,'after_cleanup')
        clips=[]
        for index,(request,raw) in enumerate(zip(requests,generated)):
            motion=decode_signspark_streams(raw['body'],raw['left'],raw['right'],raw['face'],left_hand_convention='model-right')
            if float(np.max(np.abs(np.diff(motion.poses,axis=0))))<=1e-5:
                raise ValueError('Generated candidate contains no verified pose variation')
            refined,audit,mask=refine_gestures(motion.poses,30.)
            identity='generated-csl-'+hashlib.sha256(json.dumps(asdict(request),ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:16]
            target=args.output/(identity+'.npz')
            np.savez_compressed(target,**raw,effective_poses=motion.poses,expression=motion.expression,
                                refined_poses=refined,gesture_repair_mask=mask)
            raw_sha=hashlib.sha256(target.read_bytes()).hexdigest()
            clip={'id':identity,'text':request.text,'aliases':[],'fps':30.,'frames':request.frames,
                  'quality_warning':audit['after']['mapped_max_step_degrees']>30,
                  'poses':motion.poses.tolist(),'refined_poses':refined.tolist(),'gesture_refinement':audit,
                  'source_kind':'pretrained_text_conditioned_generation','model_output':'SignSparK text-only flow',
                  'source_sha256':hashlib.sha256(json.dumps(asdict(request),sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                  'source_model_sha256':{s:v[1] for s,v in MODEL_FILES.items()},'generated_parameters_sha256':raw_sha,
                  'model_revision':MODEL_REVISION,'generation_source_commit':SOURCE_COMMIT,'generation_codec':CODEC_VERSION,
                  'target_motion':None,'observed_keyframes':0,'explicit_output_frames':request.frames,
                  'experimental':True,'human_approved':False,'language_review':'unverified','motion_review':'unverified',
                  'unmapped_target_channels':['expression','jaw'],'expression_channels_retained':50}
            clip=opt_in_generated_jaw(clip,smplx_model=args.smplx_model,source_presentation=args.source_presentation)
            json_write(args.output/(identity+'.json'),clip);clips.append(clip)
        if len(requests)>1:
            difference=float(np.max(np.abs(generated[0]['body']-generated[1]['body'])))
            if difference==0:raise ValueError('Different texts did not change same-seed body output')
            manifest['different_text_body_max_difference']=difference
        archive=args.output/'candidate-motion.zip'
        # The external file may have changed during a long inference run.
        verify_base_archive(args.base_archive)
        write_candidate_archive(archive,args.base_archive,clips,'SignSparK pretrained custom CSL candidates; semantic/motion review NOT_RUN')
        digest=hashlib.sha256(archive.read_bytes()).hexdigest()
        registry=MotionRegistry(archive,digest)
        presentation=build_avatar_presentation(args.output/'presentation',assets=registry.catalog()['assets'],
                                               source_presentation=args.source_presentation)
        manifest.update(state='complete',candidate_archive_sha256=digest,asset_ids=[a['id'] for a in registry.catalog()['assets']],
                        presentation_manifest_sha256=hashlib.sha256((args.output/'presentation/manifest.json').read_bytes()).hexdigest(),
                        nonmanual_profile=clips[0]['nonmanual_candidate']['profile'],mouth_glb_export_supported=False,
                        target_avatar_sha256=presentation['files'][presentation['avatar']])
        json_write(report,manifest);print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    except Exception as exc:
        manifest.update(state='failed',error=type(exc).__name__+': '+str(exc))
        step=manifest.get('active_step',{})
        try:
            record_memory(manifest,report,torch,step.get('stream'),'failure',failed_after=step.get('stage'))
        except Exception as diagnostic_error:
            try:
                print('Failed to persist failure memory diagnostic; preserving original error: '+
                      type(diagnostic_error).__name__+': '+str(diagnostic_error),file=sys.stderr,flush=True)
            except Exception:
                pass  # Even a closed/broken stderr must not replace the primary error.
        raise
    return 0


if __name__=='__main__':raise SystemExit(main())
