"""Paced single-session synthetic continuity probe; not a camera/semantic test."""
import argparse
import asyncio
import base64
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validate_realtime_pipeline import read,save,sha


async def run(args):
    import cv2
    import httpx
    import numpy as np
    import psutil
    from slbridge.slrt.backend import video_packets
    from websockets.asyncio.client import connect
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'artifacts') or output.exists():raise ValueError('Use fresh artifact output')
    parsed=urlsplit(args.base_url)
    if parsed.scheme!='http' or parsed.hostname not in {'127.0.0.1','localhost'} or not parsed.port:
        raise ValueError('Use loopback HTTP product')
    source=read(args.source_run/'manifest.json')
    state=read(args.runtime_state)
    process_ids=[state['recognition_pid'],state['product_pid']]
    samples=[s for s in source['samples'] if s['kind']=='positive' and s['replay']][:2]
    assert len(samples)==2 and all(s['fps']==30 and sha(s['video'])==s['video_sha256'] for s in samples)
    prefix='/v1/communication/realtime'
    with httpx.Client(trust_env=False,timeout=5) as client:health=client.get(args.base_url+prefix).json()
    assert health['ready'] and not health['busy']
    save(output/'manifest.json',dict(protocol='synthetic_single_ws_continuity_v1',samples=samples,
        source_manifest_sha256=sha(args.source_run/'manifest.json'),health=health,duration_target_seconds=args.seconds,
        fps=30,tail_seconds=1.2,manual_boundary_after_tail=True,black_separator_seconds=.5,
        image_protocol='original_512_square_to_opencv_jpeg_quality90',llm_assistance=False,
        camera_and_new_signers='NOT_RUN',script_sha256=sha(__file__)))
    count=0;cycles=0;events=[];queues=[];pending=[];lateness=[];resources=[];terminal=None
    started=None;last_resource=-60
    def resource():
        processes=[]
        for pid in process_ids:
            try:
                p=psutil.Process(pid)
                memory=p.memory_info()
                processes.append(dict(pid=p.pid,rss=memory.rss,vms=memory.vms))
            except (psutil.AccessDenied,psutil.NoSuchProcess):pass
        resources.append(dict(elapsed_seconds=time.perf_counter()-started,processes=processes,
                              host_available=psutil.virtual_memory().available))
    ws='ws://'+parsed.netloc+prefix
    with (output/'events.jsonl').open('w',encoding='utf-8') as log:
        async with connect(ws,proxy=None,max_size=262144,max_queue=64) as socket:
            await socket.send(json.dumps(dict(type='start',protocol='csl-realtime-v1',sign_language='csl',source_epoch=1,fps=30)))
            ready=json.loads(await asyncio.wait_for(socket.recv(),20));assert ready['type']=='ready'
            started=time.perf_counter();resource()
            async def receive():
                nonlocal terminal
                while True:
                    event=json.loads(await socket.recv());event['received_ms']=(time.perf_counter()-started)*1000
                    if event['type']=='ack':
                        queues.append(event['queue_ms']);pending.append(event['pending_frames'])
                    else:events.append(event)
                    log.write(json.dumps(event,ensure_ascii=False)+'\n')
                    if event['type'] in {'error','stopped','finished'}:terminal=event;return
            receiver=asyncio.create_task(receive())
            async def send(image):
                nonlocal count,last_resource
                stamp=count/30;due=started+stamp
                await asyncio.sleep(max(0,due-time.perf_counter()))
                late=max(0,time.perf_counter()-due)*1000
                lateness.append(late)
                if late>250:raise RuntimeError('Probe producer lost real-time pacing; do not burst stale frames')
                ok,jpeg=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,90]);assert ok
                await socket.send(json.dumps(dict(type='frame',index=count,timestamp=stamp,source_epoch=1,
                    capture_ms=(time.perf_counter()-started)*1000,jpeg=base64.b64encode(jpeg).decode('ascii'))))
                count+=1
                if time.perf_counter()-started-last_resource>=60:
                    resource();last_resource=time.perf_counter()-started;log.flush()
                    print(json.dumps(dict(elapsed_seconds=last_resource,frames=count,cycles=cycles,
                                          queue_max_ms=max(queues,default=0))),flush=True)
            try:
                while time.perf_counter()-started<args.seconds and not receiver.done():
                    sample=samples[cycles%len(samples)];packets=video_packets(sample['video']);last=None
                    try:
                        for packet in packets:
                            if receiver.done():break
                            await send(packet.image);last=packet.image
                    finally:packets.close()
                    if receiver.done():break
                    if last is not None:
                        for _ in range(36):
                            if receiver.done():break
                            await send(last)
                    if receiver.done():break
                    await socket.send(json.dumps({'type':'boundary'}))
                    for _ in range(15):
                        if receiver.done():break
                        await send(np.zeros((512,512,3),dtype=np.uint8))
                    cycles+=1
                if not receiver.done():await socket.send(json.dumps({'type':'finish'}))
                await asyncio.wait_for(receiver,30)
            finally:
                if not receiver.done():receiver.cancel();await asyncio.gather(receiver,return_exceptions=True)
                resource()
    summary=dict(elapsed_seconds=time.perf_counter()-started,sent_frames=count,ack_frames=len(queues),cycles=cycles,
        terminal=terminal,queue_p50_ms=float(np.percentile(queues,50)) if queues else None,
        queue_p95_ms=float(np.percentile(queues,95)) if queues else None,max_pending_frames=max(pending,default=0),
        producer_max_lateness_ms=max(lateness,default=0),
        committed_sentences=sum(e['type']=='sentence' and e.get('status')=='committed' for e in events),
        errors=[e for e in events if e['type']=='error'],
        native_summary=next((e for e in reversed(events) if e['type']=='summary'),None),resources=resources,
        camera_and_semantics='NOT_RUN')
    save(output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='resources'}),flush=True)
    if not terminal or terminal['type']!='finished' or len(queues)!=count:raise RuntimeError('Continuity probe did not complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seconds',type=float,default=600)
    parser.add_argument('--base-url',default='http://127.0.0.1:8879')
    parser.add_argument('--runtime-state',type=Path,default=ROOT/'artifacts/product-bridge-20261108/runtime-state.json')
    args=parser.parse_args()
    if not 10<=args.seconds<=1200:parser.error('Duration outside bounded probe range')
    asyncio.run(run(args))
