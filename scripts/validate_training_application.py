"""Application probe on verified train videos; references stay in evaluation."""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import gzip
import json
from pathlib import Path
import pickle
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_realtime_pipeline import read,save,sha,one

ORIGINAL=ROOT/'artifacts/unisign-csldaily-train-657-20261025'
ANNOTATIONS=Path('F:/model_weigth/data/csl-daily')
VIDEOS=Path('G:/Archive/Zenodo/csl-daliy')
LEGACY=ROOT/'artifacts/slrt-100-train-fit-20261025'
UNI_SHA='73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c'


def choose(rows,n=30):
    buckets=defaultdict(list)
    for row in sorted(rows,key=lambda x:x['sample_id']):
        if row['source_present'] and 4<=row['end']-row['start']<=200:
            buckets[row['signer']].append(row)
    selected=[]
    while len(selected)<n:
        before=len(selected)
        for signer in sorted(buckets):
            if buckets[signer] and len(selected)<n:selected.append(buckets[signer].pop(0))
        if len(selected)==before:raise ValueError('Not enough eligible training videos')
    return selected


def write_clip(source,target,start,end):
    """Decode the original clock; [start,end), no approximate keyframe seeking."""
    import cv2
    cap=cv2.VideoCapture(str(source))
    fps=float(cap.get(cv2.CAP_PROP_FPS));total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened() or not 10<=fps<=30 or not 0<=start<end<=total or width!=height:
        cap.release();raise ValueError('Invalid source video or crop protocol')
    target.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(512,512))
    count=0;first=None
    try:
        if not writer.isOpened():raise ValueError('MP4V writer unavailable')
        for index in range(end):
            if index<start:
                if not cap.grab():raise ValueError('Source ended before crop')
                continue
            ok,image=cap.read()
            if not ok:raise ValueError('Source ended inside crop')
            image=cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
            writer.write(image);count+=1
            if first is None:first=image.copy()
    finally:cap.release();writer.release()
    if count!=end-start:raise ValueError('Crop count mismatch')
    return dict(fps=fps,frames=count,raw_frames=total,raw_size=[width,height]),first


def prepare(output):
    import cv2
    import numpy as np
    if (output/'manifest.json').exists():raise ValueError('Use a new output directory')
    rows=read(output/'pose-inventory.json');selected=choose(rows)
    annotations={}
    for split in ['train','dev','test']:
        with gzip.open(ANNOTATIONS/f'csl-daily.{split}','rb') as f:
            annotations[split]={r['name']:r for r in pickle.load(f)}
    selected_ids={r['sample_id'] for r in selected}
    assert selected_ids<=annotations['train'].keys()
    assert not selected_ids.intersection(annotations['dev']) and not selected_ids.intersection(annotations['test'])
    # Freeze the sample IDs before any current model requests or outcome lookup.
    save(output/'selection.json',dict(rule='first30_signer_round_robin_span_le200_no_word_or_prediction_filter',
         samples=selected,eligible=sum(4<=r['end']-r['start']<=200 and r['source_present'] for r in rows)))
    samples=[];references=[];first_frame=None
    for row in selected:
        name=row['sample_id'];source=VIDEOS/f'{name}.mp4';video=output/'inputs'/f'{name}.mp4'
        info,first=write_clip(source,video,row['start'],row['end'])
        assert info['raw_frames']==row['pose_frames']
        pose=ORIGINAL/'pose'/f'{name}.pkl'
        sample=dict(sample_id=name,case_id=name,kind='positive',source_split='train',signer_id=row['signer'],
            video=str(video),video_sha256=sha(video),source_video=str(source),source_video_sha256=sha(source),
            pose_sha256=sha(pose),crop_start=row['start'],crop_end_exclusive=row['end'],**info)
        samples.append(sample)
        ref=annotations['train'][name]
        references.append(dict(sample_id=name,gt_gloss=ref['gloss'],ref_sentence_upstream=ref['text']))
        if first_frame is None:first_frame=first
        print(json.dumps({'prepared':name,'frames':info['frames']}),flush=True)
    for name,image in [('black',np.zeros_like(first_frame)),('static_signer',first_frame)]:
        video=output/'inputs'/f'{name}.mp4'
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(512,512));assert writer.isOpened()
        for _ in range(90):writer.write(image)
        writer.release()
        samples.append(dict(sample_id=name,case_id=name,kind='control',video=str(video),video_sha256=sha(video),fps=30,frames=90))
    # Predeclared crop probes depend on a duration mismatch, never on prediction quality.
    probes=[]
    for path in sorted((LEGACY/'alignment').glob('*.json')):
        old=read(path);row=next(r for r in rows if r['sample_id']==path.stem)
        if abs(row['end']-row['start']-old['length'])<100:continue
        name=path.stem;assert name in annotations['train']
        for variant,start,end in [('official',row['start'],row['end']),('legacy',old['best_start'],old['best_start']+old['length'])]:
            video=output/'crop-inputs'/f'{name}--{variant}.mp4';source=VIDEOS/f'{name}.mp4'
            info,_=write_clip(source,video,start,end)
            probes.append(dict(sample_id=name,case_id=f'{name}--{variant}',kind='crop_probe',source_split='train',
                 variant=variant,video=str(video),video_sha256=sha(video),source_video_sha256=sha(source),crop_start=start,crop_end_exclusive=end,**info))
        if len(probes)==4:break
    assert len(probes)==4
    for name in {x['sample_id'] for x in probes}:
        ref=annotations['train'][name];references.append(dict(sample_id=name,gt_gloss=ref['gloss'],ref_sentence_upstream=ref['text']))
    save(output/'eval/references.json',references,evaluation=True)
    save(output/'manifest.json',dict(protocol='training_application_probe_v1',created_at=datetime.now(timezone.utc).isoformat(),
         data_scope='verified_train_members_short_video_application_probe_not_new_user_validation',
         inventory_count=len(rows),selection_sha256=sha(output/'selection.json'),samples=samples,crop_probes=probes,
         annotation_hashes={split:sha(ANNOTATIONS/f'csl-daily.{split}') for split in annotations},
         source_pose_baseline='imported_from_same_checkpoint_657_run_not_rerun',llm_assistance=False,
         image_protocol='author_pose_start_end_exclusive_512_square_MP4V_original_fps',realtime_tail_seconds=1.2,
         script_sha256=sha(__file__),camera_imitation='NOT_RUN'))


async def run(output):
    import httpx
    manifest=read(output/'manifest.json');assert manifest['script_sha256']==sha(__file__)
    async with httpx.AsyncClient(trust_env=False,timeout=60) as client:
        status=(await client.get('http://127.0.0.1:8879/v1/communication/realtime')).json()
        assert status['ready'] and not status['busy']
        assert status['pipeline']['translation_model_id']=='uni-sign-csl-daily-pose-only-182913ca'
        models=(await client.get('http://127.0.0.1:8765/v3/models')).json()
        assert models['capabilities']['csl']['model_version']==UNI_SHA
        save(output/'runtime.json',dict(live=status,uni_checkpoint_sha256=UNI_SHA,script_sha256=sha(__file__),
            code_sha256={str(p.relative_to(ROOT)):sha(p) for p in list((ROOT/'slbridge').glob('realtime_*.py'))+
              [ROOT/'slbridge/uni_sign.py',ROOT/'slbridge/slrt/runtime.py',ROOT/'slbridge/slrt/pose.py'] }))
        for index,sample in enumerate(manifest['crop_probes']+manifest['samples']):
            path=output/'outputs'/f"{sample['case_id']}.json"
            if path.exists():raise ValueError('Refusing to overwrite a sample result')
            video=Path(sample['video']);assert sha(video)==sample['video_sha256']
            started=time.perf_counter()
            try:
                response=await client.post('http://127.0.0.1:8879/v1/communication/recognitions',
                    params=dict(sign_language='csl',request_id='train-probe-'+sample['case_id'],filename=video.name),
                    content=video.read_bytes(),headers={'Content-Type':'video/mp4'})
                response.raise_for_status();data=response.json()
                clip=dict(request_status='ok',response=data,text=data['text'],http_ms=(time.perf_counter()-started)*1000)
            except Exception as error:clip=dict(request_status='error',error=type(error).__name__,text='')
            live=None
            if sample['kind']!='crop_probe':
                live=await one(sample,'ws://127.0.0.1:8879/v1/communication/realtime',index+1,1.2)
            record=dict(sample_id=sample['sample_id'],case_id=sample['case_id'],clip=clip,live=live)
            save(path,record)
            print(json.dumps(dict(case_id=sample['case_id'],clip_status=clip['request_status'],clip_text=clip['text'],
                  live_status=live['request_status'] if live else None,
                  live_text=''.join(e['text'] for e in live['events'] if e['type']=='sentence' and e.get('status')=='committed') if live else None),ensure_ascii=True),flush=True)


def live_result(data):
    words={};sentences=[];first=None;resets=Counter();rejections=[]
    for event in data['events']:
        kind=event['type']
        if kind=='word':
            key=(event['utterance_id'],event['event_id'])
            if event['status']=='final':words[key]=event['gloss']
            elif event['status']=='retracted':words.pop(key,None)
            if first is None and event['status']=='partial':first=event['received_ms']
        elif kind=='sentence':
            if event['status']=='committed':sentences.append(event)
            elif event['status']=='rejected':rejections.append(event.get('reason'))
        elif kind=='utterance_reset':
            words={k:v for k,v in words.items() if k[0]!=event['utterance_id']};resets[event['reason']]+=1
    return dict(request_status=data['request_status'],text=''.join(e['text'] for e in sentences),
                predicted_gloss=list(words.values()),first_word_ms=first,resets=dict(resets),rejections=rejections,
                extra_after_source_end_ms=sentences[-1]['received_ms']-data['source_capture_end_ms'] if sentences else None)


def evaluate(output):
    from experiments.p1.eval import metrics as metrics
    import numpy as np
    manifest=read(output/'manifest.json');refs={r['sample_id']:r for r in read(output/'eval/references.json')}
    old_summary=read(ORIGINAL/'output/summary.json');assert old_summary['checkpoint_sha256']==UNI_SHA
    old={r['sample']:r for r in (json.loads(line) for line in (ORIGINAL/'output/predictions.jsonl').read_text(encoding='utf-8').splitlines())}
    rows=[];controls=[];probes=[]
    for sample in manifest['samples']+manifest['crop_probes']:
        record=read(output/'outputs'/f"{sample['case_id']}.json")
        if sample['kind']=='control':controls.append(record);continue
        ref=refs[sample['sample_id']];baseline=old[sample['sample_id']]
        assert baseline['status']=='ok' and metrics.normalize_text(baseline['reference'])==metrics.normalize_text(ref['ref_sentence_upstream'])
        row=dict(sample,**ref,official_pose_imported=baseline['prediction'],clip=record['clip'],
                 live=live_result(record['live']) if record['live'] else None)
        for name,pred in [('official_pose',baseline['prediction']),('clip',record['clip']['text']),('realtime',row['live']['text'] if row['live'] else '')]:
            row[name+'_exact']=metrics.normalize_text(pred)==metrics.normalize_text(ref['ref_sentence_upstream'])
        (probes if sample['kind']=='crop_probe' else rows).append(row)
    def text_metrics(name):
        texts=[r['official_pose_imported'] if name=='official_pose' else r['clip']['text'] if name=='clip' else r['live']['text'] for r in rows]
        return dict(metrics.corpus_metrics_cslt([(r['ref_sentence_upstream'],t) for r,t in zip(rows,texts)]),
                    nonempty=sum(bool(t) for t in texts))
    first=[r['live']['first_word_ms'] for r in rows if r['live']['first_word_ms'] is not None]
    extra=[r['live']['extra_after_source_end_ms'] for r in rows if r['live']['extra_after_source_end_ms'] is not None]
    statuses={name:sum(r[name]['request_status']=='ok' for r in rows) for name in ['clip','live']}
    summary=dict(n=len(rows),split='train',signers=len({r['signer_id'] for r in rows}),request_success=statuses,
        official_pose_imported=text_metrics('official_pose'),video_clip=text_metrics('clip'),realtime=text_metrics('realtime'),
        realtime_words=metrics.corpus_wer([(r['gt_gloss'],' '.join(r['live']['predicted_gloss'])) for r in rows]) if statuses['live']==len(rows) else None,
        first_word_ms={f'p{q}':float(np.percentile(first,q)) if first else None for q in [50,95]},
        sentence_extra_after_original_eof_ms={f'p{q}':float(np.percentile(extra,q)) if extra else None for q in [50,95]},
        camera_imitation='NOT_RUN',llm_assistance=False,controls=controls,
        crop_probes=[{k:r[k] for k in ['sample_id','variant','crop_start','crop_end_exclusive','frames','ref_sentence_upstream','official_pose_imported','clip']} for r in probes])
    save(output/'eval/records.json',rows,evaluation=True);save(output/'eval/crop-probes.json',probes,evaluation=True)
    save(output/'eval/summary.json',summary,evaluation=True)
    with (output/'eval/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','expected','official_pose_imported','video_clip','realtime','slrt_words','clip_exact','realtime_exact'])
        for r in rows:writer.writerow([r['sample_id'],r['ref_sentence_upstream'],r['official_pose_imported'],r['clip']['text'],r['live']['text'],' '.join(r['live']['predicted_gloss']),r['clip_exact'],r['realtime_exact']])
    print(json.dumps({k:v for k,v in summary.items() if k not in ['controls','crop_probes']},ensure_ascii=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','run','evaluate'])
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/train-application-validation-v1');args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'artifacts'):parser.error('Use an artifact directory')
    if args.phase=='prepare':prepare(output)
    elif args.phase=='run':asyncio.run(run(output))
    else:evaluate(output)
