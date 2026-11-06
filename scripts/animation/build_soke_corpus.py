"""Build a traceable, unreviewed CSL-Daily sentence corpus from SOKE fittings."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from slbridge.animation.soke import scan_archive,load_sequence,effective_poses,write_candidate_archive,video_binding,select_seed_rows
from slbridge.animation.gesture import refine_gestures,GESTURE_PROFILE
from slbridge.animation.soke_pickle import parse_pickle
from slbridge.animation.registry import MotionRegistry

BASE_SHA='4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5'
PROCESSING='soke-upper133-handmean-wrist-ik-candidate-v1'
ATTRIBUTION='SOKE CSL-Daily 拟合候选；动作与词义未审核。数据、人体模板按各自来源条款使用。SOKE 仓库许可 CC BY-NC-ND 4.0。'
WORDS=['你好','谢谢','开会','会议','工作','今天','明天','调查','学校','医院','天气','吃饭','朋友','不要','不能','哪里','什么','怎么','喜欢','帮助']


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while data:=stream.read(8*1024*1024):digest.update(data)
    return digest.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def annotations(directory):
    rows=[];sources={}
    for split in ('train','val','test'):
        path=directory/('csl_clean.'+split);sources[split]=sha(path)
        values=parse_pickle(gzip.decompress(path.read_bytes()))
        if not isinstance(values,list):raise ValueError('Expected annotation list')
        for source in values:
            row={k:source[k] for k in ('name','text','gloss','num_frames','signer')}
            if not isinstance(row['text'],str) or type(row['num_frames']) is not int or not 1<=row['num_frames']<=3000:raise ValueError('Invalid annotation row')
            rows.append(row|{'split':split})
    if len({r['name'] for r in rows})!=len(rows):raise ValueError('Duplicate annotation IDs')
    return rows,sources


def probe(path,ffprobe):
    command=[str(ffprobe),'-v','error','-count_frames','-select_streams','v:0','-show_entries',
             'stream=avg_frame_rate,nb_read_frames,width,height','-of','json',str(path)]
    value=json.loads(subprocess.check_output(command,text=True,encoding='utf-8',timeout=30))['streams'][0]
    fps=float(Fraction(value['avg_frame_rate']))
    if not 0<fps<=120:raise ValueError('Invalid original frame rate')
    return {'fps':fps,'frames':int(value['nb_read_frames']),'width':value['width'],'height':value['height']}


def max_step(poses):
    angle=np.linalg.norm(poses,axis=-1,keepdims=True)
    unit=np.divide(poses,angle,out=np.zeros_like(poses),where=angle>1e-8)
    quaternion=np.concatenate([unit*np.sin(angle/2),np.cos(angle/2)],axis=-1)
    dots=np.clip(np.abs(np.sum(quaternion[1:]*quaternion[:-1],axis=-1)),0,1)
    return float(np.degrees(2*np.arccos(dots)).max()) if len(poses)>1 else 0.0


def bake(output,archive,avatar,blender):
    presentation=output/'presentation';source=presentation/'source';source.mkdir(parents=True)
    registry=MotionRegistry(archive,sha(archive))
    clean=presentation/'avatar-clean.glb';shutil.copyfile(avatar,clean)
    (source/'three.mjs').write_bytes(registry.resource('three.module.js')[0])
    pilots=[]
    for asset in registry.catalog()['assets']:
        motion=registry.motion(asset['id']);path=source/(asset['id']+'.json');write(path,motion)
        pilot={'asset_id':asset['id'],'source_motion_sha256':asset['sha256'],'motion':str(path),'text':asset['text']}
        if motion['clip'].get('gesture_refinement'):pilot['temporal_profile']=GESTURE_PROFILE
        pilots.append(pilot)
    build={'output':str(presentation),'model':str(clean),'three':str(source/'three.mjs'),'pilots':pilots,'bake_only':True}
    build_path=source/'build-input.json';write(build_path,build)
    write(presentation/'production-report.json',{'avatar_reused_from':str(avatar),'avatar_input_sha256':sha(avatar),
        'motion_review':'NOT_RUN','language_review':'NOT_RUN'})
    with (output/'retarget.log').open('w',encoding='utf-8') as log:
        subprocess.run(['node',str(ROOT/'scripts/animation/export_tracks.mjs'),str(build_path)],stdout=log,stderr=subprocess.STDOUT,check=True)
    helper=ROOT/'scripts/animation/build_presentation.py'
    with (output/'blender.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(blender),'--background','--python',str(helper),'--','--blender-stage',str(build_path)],stdout=log,stderr=subprocess.STDOUT,check=True)
    spec=importlib.util.spec_from_file_location('soke_presentation_helpers',helper);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    avatar_sha=sha(clean);processing=GESTURE_PROFILE if any('temporal_profile' in p for p in pilots) else PROCESSING
    for pilot in pilots:
        binding={
            'asset_id':pilot['asset_id'],'source_motion_sha256':pilot['source_motion_sha256'],
            'avatar_sha256':avatar_sha,'processing_version':processing}
        if 'temporal_profile' in pilot:binding['temporal_profile']=pilot['temporal_profile']
        module.bind_export(presentation/(pilot['asset_id']+'.glb'),binding)
    files=[clean,*[presentation/(p['asset_id']+'.glb') for p in pilots]]
    write(presentation/'manifest.json',{'schema_version':1,'avatar':clean.name,'processing_version':processing,
        'attribution':'Godette (Rigged), zahlenmaler / CC BY 4.0; existing SignLanguage-Bridge presentation derivative',
        'source_archive_sha256':registry.sha256,'rig_profile':json.loads((presentation/(pilots[0]['asset_id']+'.tracks.json')).read_text())['rigProfile'],
        'files':{p.name:sha(p) for p in files},'pilots':[{k:v for k,v in p.items() if k!='motion'}|{'export_file':p['asset_id']+'.glb'} for p in pilots]})
    from slbridge.animation.presentation import PresentationBundle
    PresentationBundle(presentation,registry.catalog()['assets'])
    return str(presentation)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive','indices','videos','base-archive','smplx-model','output','ffprobe'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--count',type=int,default=20)
    parser.add_argument('--blender',type=Path);parser.add_argument('--avatar',type=Path)
    parser.add_argument('--seed-corpus',type=Path,help='Retain identities from an existing selected.json/summary.json corpus')
    parser.add_argument('--refine-gestures',action='store_true',help='Add reversible refined poses and retain originals')
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Use a new output directory to retain prior evidence')
    if not 1<=args.count<=100:parser.error('Pilot count must be 1..100')
    if bool(args.blender)!=bool(args.avatar):parser.error('Both --blender and --avatar are required for baking')
    base=MotionRegistry(args.base_archive,BASE_SHA)
    base_skeleton=base.motion(base.catalog()['assets'][0]['id'])
    with np.load(args.smplx_model,allow_pickle=False) as model:
        mean=np.stack([model['hands_meanl'].reshape(45),model['hands_meanr'].reshape(45)]).astype(np.float32)
        rest=np.asarray(model['J_regressor']@model['v_template'])
    if not np.allclose(rest,base_skeleton['restJoints'],atol=2e-6,rtol=0):raise ValueError('Template rest joints differ from the existing motion skeleton')
    rows,index_hashes=annotations(args.indices)
    seeds=select_seed_rows(rows,json.loads((args.seed_corpus/'selected.json').read_text(encoding='utf-8')),args.count) if args.seed_corpus else []
    eligible=[r for r in rows if 40<=r['num_frames']<=260 and (args.videos/(r['name']+'.mp4')).exists()]
    candidates=list(seeds);texts={r['text'] for r in seeds};pool_signers={r['signer'] for r in seeds}
    for word in WORDS:
        matches=[r for r in eligible if word in r['text'] and r['text'] not in texts]
        for _ in range(min(max(3,int(np.ceil(args.count/len(WORDS)))+1),len(matches))):
            row=min(matches,key=lambda r:(r['signer'] in pool_signers,abs(r['num_frames']-140),r['name']))
            candidates.append(row);texts.add(row['text']);pool_signers.add(row['signer'])
            matches=[r for r in matches if r['text'] not in texts]
            if not matches:break
    for row in eligible:
        if len(candidates)>=max(120,args.count*3):break
        if row['text'] not in texts:candidates.append(row);texts.add(row['text'])
    print('Scanning source ZIP directory',flush=True)
    inventory,members=scan_archive(args.source_archive,{r['name'] for r in candidates})
    source_sha=sha(args.source_archive);model_sha=sha(args.smplx_model)
    if args.seed_corpus and json.loads((args.seed_corpus/'summary.json').read_text())['source_archive_sha256']!=source_sha:
        raise ValueError('Seed corpus belongs to another source archive')
    args.output.mkdir(parents=True);sequences=args.output/'sequences';sequences.mkdir()
    converted=[];clips=[];excluded=[];chosen_texts=set()
    # One representative per keyword first, then additional unique sentences.
    ordered=list(seeds);used_ids={r['name'] for r in seeds};picked_signers={r['signer'] for r in seeds}
    for word in WORDS:
        choices=[r for r in candidates if word in r['text'] and r['name'] not in used_ids]
        if choices:
            row=min(choices,key=lambda r:(r['signer'] in picked_signers,r['name']))
            ordered.append(row);used_ids.add(row['name']);picked_signers.add(row['signer'])
    ordered.extend(r for r in candidates if r['name'] not in used_ids)
    for row in ordered:
        if len(converted)>=args.count:break
        if row['text'] in chosen_texts:continue
        identity=row['name'];video=args.videos/(identity+'.mp4')
        try:
            parameters,parameter_sha=load_sequence(args.source_archive,members.get(identity,[]),row['num_frames'])
            video_info=probe(video,args.ffprobe)
        except (ValueError,OSError,subprocess.SubprocessError) as exc:
            excluded.append({'id':identity,'reason':str(exc)});continue
        poses=effective_poses(parameters,mean);step=max_step(poses);asset_id='soke-'+identity
        binding=video_binding(video_info,len(poses),30.0)
        target=sequences/(identity+'.npz')
        features=np.concatenate([parameters['smplx_body_pose'][:,33:],parameters['smplx_lhand_pose'],parameters['smplx_rhand_pose'],parameters['smplx_jaw_pose'],parameters['smplx_expr']],axis=-1)
        refinement={};extra_arrays={}
        if args.refine_gestures:
            refined,audit,mask=refine_gestures(poses,30.0)
            refinement=audit;extra_arrays={'refined_poses':refined,'gesture_repair_mask':mask}
        np.savez_compressed(target,**parameters,effective_poses=poses,features133_raw=features,frame_numbers=np.arange(len(poses),dtype=np.int32),**extra_arrays)
        video_sha=sha(video)
        record=row|{'asset_id':asset_id,'fps':30.0,'frames':len(poses),'parameter_file':'sequences/'+target.name,
            'parameter_file_sha256':sha(target),'source_parameters_sha256':parameter_sha,'video_path':str(video),
            'video_sha256':video_sha,'video_info':video_info,'video_binding':binding,'source_archive_sha256':source_sha,'template_sha256':model_sha,
            'representation':'raw-smplx-parameters-plus-effective-upper133-55-axis-angle','max_adjacent_rotation_degrees':step,
            'motion_review':'unverified','language_review':'unverified','semantic_accuracy_certified':False}
        if refinement:record['gesture_refinement']=refinement
        write(sequences/(identity+'.json'),record);converted.append(record);chosen_texts.add(row['text'])
        clips.append({'schema':'soke-csl-candidate-v1','id':asset_id,'text':row['text'],'aliases':[],
            'fps':30.0,'frames':len(poses),'poses':poses.tolist(),'source_sha256':video_sha,
            'fitted_sha256':parameter_sha,'source_id':identity,'source_dataset':'SOKE/CSL-Daily',
            'source_archive_sha256':source_sha,'template_sha256':model_sha,'quality_warning':step>30,
            'split':row['split'],'experimental':True,'human_approved':False,'model_output':'published_smplx_fit',
            'source_video_binding':binding,
            'unmapped_target_channels':['expression','jaw','shape','camera_translation'],
            'playback_canonicalization':'root and joints 1..11 zero; eyes rest; hand mean added once'})
        if refinement:
            clips[-1].update(refined_poses=refined.tolist(),gesture_refinement=refinement,
                             quality_warning=refinement['after']['mapped_max_step_degrees']>30)
        print('Converted',identity,row['text'],len(poses),flush=True)
    write(args.output/'excluded-pilot-candidates.json',excluded)
    if len(converted)!=args.count:raise ValueError('Insufficient complete/probed candidates; partial evidence retained')
    if not {r['name'] for r in seeds}<={r['name'] for r in converted}:raise ValueError('A required seed could not be retained')
    complete=set();index_ids={r['name'] for r in rows};found=inventory['sequences']
    for row in rows:
        entry=found.get(row['name'])
        if entry and entry['frames']==row['num_frames'] and entry['first_frame']==0 and not entry['missing_frame_numbers']:complete.add(row['name'])
    converted_ids={r['name'] for r in converted}
    with (args.output/'corpus.jsonl').open('w',encoding='utf-8') as stream:
        for row in rows:
            identity=row['name'];entry=found.get(identity)
            state='converted_candidate' if identity in converted_ids else 'complete_frames_unprocessed' if identity in complete else 'incomplete_frames' if entry else 'missing_3d_sequence'
            stream.write(json.dumps(row|{'state':state,'archive_frame_count':entry['frames'] if entry else 0,
                'local_video_filename_exists':(args.videos/(identity+'.mp4')).exists(),'language_review':'unverified'},ensure_ascii=False)+'\n')
    write(args.output/'inventory.json',inventory)
    archive=args.output/'candidate-motion.zip';write_candidate_archive(archive,args.base_archive,clips,ATTRIBUTION)
    (args.output/'candidate-motion.zip.sha256').write_text(sha(archive)+'  candidate-motion.zip\n',encoding='ascii')
    checked=MotionRegistry(archive,sha(archive))
    summary={'schema':'soke-csl-corpus-build-v1','research_only':True,'source_archive':str(args.source_archive),
        'source_archive_sha256':source_sha,'template_file':str(args.smplx_model),'template_sha256':model_sha,
        'index_sha256':index_hashes,'indexed_samples':len(rows),'archive_sequences':len(found),
        'complete_frame_ranges':len(complete),'incomplete_sequences':len(index_ids&set(found))-len(complete),
        'missing_sequences':sorted(index_ids-set(found)),'actual_frame_files':sum(r['frames'] for r in found.values()),
        'converted':len(converted),'converted_frames':sum(r['frames'] for r in converted),'local_video_frame_counts_probed':len(converted),
        'local_video_frame_count_matches':sum(r['video_binding']['frame_count_match'] for r in converted),
        'local_video_temporal_alignment':'NOT_RUN; local originals are uncropped; no automatic stretching/truncation',
        'pose_fps':30.0,'pose_fps_basis':'CSL-Daily CVPR2021 paper section 4.1 native FPS; release timestamps not independently verified',
        'selected_signers':sorted({r['signer'] for r in converted}),
        'candidate_registry_assets':len(checked.catalog()['assets']),'candidate_archive_sha256':checked.sha256,
        'excluded_pilot_candidates':excluded,'motion_review':'NOT_RUN','language_review':'NOT_RUN',
        'full_archive_member_crc_audit':'NOT_RUN; selected sequences only','face_target_channels':'raw fields retained; Godette facial coefficients not retargeted'}
    summary['retained_seed_ids']=[r['name'] for r in seeds]
    if args.refine_gestures:
        audits=[r['gesture_refinement'] for r in converted]
        seed_ids={s['name'] for s in seeds};seed_audits=[r['gesture_refinement'] for r in converted if r['name'] in seed_ids]
        summary['gesture_refinement']={'profile':GESTURE_PROFILE,'refined_clips':len(audits),
            'isolated_repaired_joint_frames':sum(a['isolated_repaired_joint_frames'] for a in audits),
            'mapped_events_over60_before':sum(a['before']['mapped_step_events_over60'] for a in audits),
            'mapped_events_over60_after':sum(a['after']['mapped_step_events_over60'] for a in audits),
            'seed_events_over60_before':sum(a['before']['mapped_step_events_over60'] for a in seed_audits),
            'seed_events_over60_after':sum(a['after']['mapped_step_events_over60'] for a in seed_audits),
            'retained_seed_clips':len(seed_audits),'semantic_review':'NOT_RUN'}
    write(args.output/'selected.json',converted);write(args.output/'summary.json',summary)
    if args.blender:
        summary['presentation']=bake(args.output,archive,args.avatar,args.blender)
        write(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
