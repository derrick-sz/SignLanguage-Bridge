"""Real paced WebSocket replay; GT is confined to the evaluate phase."""
import argparse
import asyncio
import base64
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.p1.gt_guard import assert_gt_clean


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data,evaluation=False):
    if not evaluation:assert_gt_clean(data,context=str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


async def one(sample,ws_url,epoch,tail_seconds,*,assistance=False,advice_url=None):
    import cv2
    from websockets.asyncio.client import connect
    from slbridge.slrt.backend import video_packets
    events=[];lateness=[];sent=0;capture_end=None;error=None;ready_wait=None
    packets=video_packets(sample['video'])
    try:
        requested=time.perf_counter()
        async with connect(ws_url,proxy=None,max_size=262144,max_queue=64) as socket:
            start=dict(type='start',protocol='csl-realtime-v1',sign_language='csl',source_epoch=epoch,fps=sample['fps'])
            if assistance:start['assistance']=True
            await socket.send(json.dumps(start))
            ready=json.loads(await asyncio.wait_for(socket.recv(),20))
            if ready['type']!='ready':raise RuntimeError('Live runtime did not become ready')
            ready_wait=(time.perf_counter()-requested)*1000
            started=time.perf_counter()
            async def receive():
                while True:
                    value=json.loads(await socket.recv())
                    value['received_ms']=(time.perf_counter()-started)*1000
                    events.append(value)
                    if len(events)>4096:raise RuntimeError('Unbounded live event count')
                    if value['type'] in {'finished','stopped','error'}:return
            receiver=asyncio.create_task(receive())
            origin=None;last=None
            async def send(image,index,stamp):
                nonlocal sent
                due=started+stamp
                await asyncio.sleep(max(0,due-time.perf_counter()))
                lateness.append(max(0,time.perf_counter()-due)*1000)
                ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90])
                if not ok:raise ValueError('JPEG encoding failed')
                await socket.send(json.dumps(dict(type='frame',index=index,timestamp=stamp,source_epoch=epoch,
                    capture_ms=(time.perf_counter()-started)*1000,jpeg=base64.b64encode(encoded).decode('ascii'))))
                sent+=1
            for packet in packets:
                if receiver.done():break
                if origin is None:origin=packet.timestamp
                await send(packet.image,packet.frame_index,packet.timestamp-origin)
                last=packet
            original_end=sample['frames']/sample['fps']
            capture_end=original_end*1000
            if last is not None:
                for offset in range(int(round(tail_seconds*sample['fps']))):
                    if receiver.done():break
                    index=last.frame_index+1+offset
                    await send(last.image,index,original_end+offset/sample['fps'])
            if not receiver.done():
                await asyncio.sleep(max(0,started+original_end+tail_seconds-time.perf_counter()))
                await socket.send(json.dumps({'type':'finish'}))
            await asyncio.wait_for(receiver,30)
            if not events or events[-1]['type']!='finished':
                error='live_session_not_finished'
    except Exception as exc:
        error=type(exc).__name__
    finally:
        packets.close()
        if 'receiver' in locals() and not receiver.done():
            receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
    advice=[]
    if assistance and advice_url:
        import httpx
        async with httpx.AsyncClient(trust_env=False,timeout=3) as client:
            for event in events:
                if event['type']!='assistance' or event.get('status')!='pending':continue
                import re
                job_id=event.get('job_id','')
                if not re.fullmatch(r'[a-f0-9]{32}',job_id):raise ValueError('Invalid generated advice job')
                deadline=time.monotonic()+12
                while True:
                    try:
                        response=await client.get(advice_url+'/'+job_id);response.raise_for_status();job=response.json()
                        assert all(job[key]==event[key] for key in ['session_id','source_epoch','utterance_id','revision'])
                    except Exception as exc:
                        job=dict(status='poll_error',reason=type(exc).__name__);break
                    if job['status'] not in {'pending','running'} or time.monotonic()>=deadline:break
                    await asyncio.sleep(.15)
                advice.append(job)
    return dict(sample_id=sample['sample_id'],request_status='error' if error else 'ok',error=error,
        source_capture_end_ms=capture_end,tail_seconds=tail_seconds,sent_frames=sent,
        ready_wait_ms=ready_wait,
        ready=ready if 'ready' in locals() else None,events=events,advice=advice,
        producer_max_lateness_ms=max(lateness) if lateness else None)


async def run(a):
    import httpx
    parsed=urlsplit(a.base_url)
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or not parsed.port:
        raise ValueError('Use a loopback HTTP service')
    if (a.output/'manifest.json').exists():raise ValueError('Use a fresh output directory')
    source=read(a.source_run/'manifest.json')
    positives=[s for s in source['samples'] if s['kind']=='positive' and (a.all or s['replay'])]
    if a.limit:positives=positives[:a.limit]
    samples=positives+[s for s in source['samples'] if s['kind']!='positive']
    prefix='/v1/communication/realtime' if a.product else '/v3/realtime'
    with httpx.Client(trust_env=False,timeout=10) as client:
        health=client.get(a.base_url+prefix).json()
    assert health['enabled'] and health['ready'] and health['protocol']=='csl-realtime-v1'
    assert health['pipeline']['translation_model_id']=='uni-sign-csl-daily-pose-only-182913ca'
    assert health['pipeline']['slrt_checkpoint_sha256']=='e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d'
    save(a.output/'manifest.json',dict(protocol='paced_live_ws_replay_v1',data_scope=source['data_scope'],
         source_manifest_sha256=sha(a.source_run/'manifest.json'),samples=samples,health=health,
         image_protocol='original_512_square_to_opencv_jpeg_quality90',tail_seconds=a.tail_seconds,
         assistance=a.assist,
         script_sha256=sha(__file__),camera_and_new_signers='NOT_RUN'))
    ws_url='ws://'+parsed.netloc+prefix
    for i,sample in enumerate(samples):
        assert sha(sample['video'])==sample['video_sha256']
        result=await one(sample,ws_url,i+1,a.tail_seconds,assistance=a.assist,
                         advice_url=a.base_url+prefix+'/assistance')
        save(a.output/'outputs'/f"{sample['sample_id']}.json",result)
        print(json.dumps(dict(sample_id=sample['sample_id'],status=result['request_status'],
            sent_frames=result['sent_frames'],last=result['events'][-1]['type'] if result['events'] else None,
            sentences=[e['text'] for e in result['events'] if e['type']=='sentence' and e['status']=='committed'],
            advice_statuses=[j['status'] for j in result['advice']]),
            ensure_ascii=True),flush=True)


def evaluate(a):
    import numpy as np
    from experiments.p1.eval import metrics as m
    manifest=read(a.output/'manifest.json');assert sha(a.source_run/'manifest.json')==manifest['source_manifest_sha256']
    original={r['sample_id']:r for r in read(a.source_run/'eval/records.json')}
    records=[];controls=[]
    for s in manifest['samples']:
        output=read(a.output/'outputs'/f"{s['sample_id']}.json")
        words={};sentences=[];first=None;queue=[];errors=[];summaries=[]
        for e in output['events']:
            if e['type']=='word':
                key=(e['utterance_id'],e['event_id'])
                if e['status']=='final':words[key]=e['gloss']
                if e['status']=='partial' and first is None:first=e['received_ms']
            elif e['type']=='sentence' and e['status']=='committed':sentences.append(e)
            elif e['type']=='ack':queue.append(e['queue_ms'])
            elif e['type']=='error':errors.append(e.get('code'))
            elif e['type']=='summary':summaries.append(e)
        row=dict(sample_id=s['sample_id'],request_status=output['request_status'],first_word_ms=first,
                 ready_wait_ms=output.get('ready_wait_ms'),
                 first_word_total_ms=first+output['ready_wait_ms'] if first is not None and output.get('ready_wait_ms') is not None else None,
                 text=''.join(e['text'] for e in sentences),predicted_gloss=list(words.values()),
                 sentence_events=sentences,queue_p95_ms=float(np.percentile(queue,95)) if queue else None,
                 errors=errors,summary=summaries[-1] if summaries else None,
                 extra_after_source_end_ms=sentences[-1]['received_ms']-output['source_capture_end_ms'] if sentences else None)
        if s['kind']=='positive':records.append(dict(original[s['sample_id']],live=row))
        else:controls.append(row)
    first=[r['live']['first_word_ms'] for r in records if r['live']['first_word_ms'] is not None]
    total_first=[r['live']['first_word_total_ms'] for r in records if r['live']['first_word_total_ms'] is not None]
    extra=[r['live']['extra_after_source_end_ms'] for r in records if r['live']['extra_after_source_end_ms'] is not None]
    ok=all(r['live']['request_status']=='ok' for r in records)
    summary=dict(n=len(records),successful_requests=sum(r['live']['request_status']=='ok' for r in records),
        nonempty_sentences=sum(bool(r['live']['text']) for r in records),
        text_metrics=m.corpus_metrics_cslt([(r['ref_sentence_upstream'],r['live']['text']) for r in records]),
        word_metrics=m.corpus_wer([(r['gt_gloss'],' '.join(r['live']['predicted_gloss'])) for r in records]) if ok else None,
        first_word_ms={f'p{q}':float(np.percentile(first,q)) if first else None for q in (50,95)},
        first_word_total_ms={f'p{q}':float(np.percentile(total_first,q)) if total_first else None for q in (50,95)},
        last_sentence_extra_after_source_end_ms={f'p{q}':float(np.percentile(extra,q)) if extra else None for q in (50,95)},
        conditional_latency_n=len(extra),controls=controls,data_scope=manifest['data_scope'],camera_and_new_signers='NOT_RUN')
    save(a.output/'eval/records.json',records,evaluation=True)
    save(a.output/'eval/summary.json',summary,evaluation=True)
    print(json.dumps({k:v for k,v in summary.items() if k!='controls'},ensure_ascii=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['run','evaluate'])
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--base-url',default='http://127.0.0.1:8765');p.add_argument('--product',action='store_true')
    p.add_argument('--all',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--tail-seconds',type=float,default=1.2)
    p.add_argument('--assist',action='store_true',help='Opt in to generated-text-only advice and poll separately after native finish')
    a=p.parse_args();a.source_run=a.source_run.resolve();a.output=a.output.resolve()
    if not a.output.is_relative_to(ROOT/'artifacts') or not 0<=a.tail_seconds<=2:p.error('Invalid output/tail')
    if a.phase=='run':asyncio.run(run(a))
    else:evaluate(a)
