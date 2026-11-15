"""Compare frozen WS replay runs without feeding reference text to inference."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def values(rows):
    return dict(n=len(rows), p50=float(np.percentile(rows,50)) if rows else None,
                p95=float(np.percentile(rows,95)) if rows else None,
                max=max(rows) if rows else None)


def analyze(path):
    manifest = read(path/'manifest.json')
    first, ready, total, queue, stages, serial, outputs = [], [], [], [], {}, [], {}
    statuses = []
    for sample in manifest['samples']:
        result = read(path/'outputs'/f"{sample['sample_id']}.json")
        events = result['events']; statuses.append(result['request_status'])
        words = [e for e in events if e['type']=='word' and e['status'] in {'partial','final'}]
        sentences = [e['text'] for e in events if e['type']=='sentence' and e.get('status')=='committed']
        semantic = [{k:e[k] for k in ('type','utterance_id','event_id','status','gloss','text','reason') if k in e}
                    for e in events if e['type'] in {'word','sentence','utterance_reset'}]
        outputs[sample['sample_id']] = dict(video_sha256=sample['video_sha256'], semantic=semantic, sentences=sentences)
        if words and sample['kind']=='positive':
            first.append(words[0]['received_ms'])
            if result.get('ready_wait_ms') is not None:
                ready.append(result['ready_wait_ms']);total.append(words[0]['received_ms']+result['ready_wait_ms'])
        queue.extend(e['queue_ms'] for e in events if e['type']=='ack')
        summary = next((e for e in reversed(events) if e['type']=='summary'),None)
        if summary:
            for name,row in summary['timings'].items():
                aggregate = stages.setdefault(name,dict(total_ms=0.,count=0))
                aggregate['total_ms']+=row['total_ms'];aggregate['count']+=row['count']
            if summary['frames_processed']:
                serial.append(sum(summary['timings'][name]['total_ms'] for name in
                                  ('pose_ms','frame_preprocess_ms','window_total_ms'))/summary['frames_processed'])
    for row in stages.values(): row['mean_ms']=row['total_ms']/row['count'] if row['count'] else None
    return dict(requests=len(statuses), completed=statuses.count('ok'), first_word_capture_ms=values(first),
                ready_wait_ms=values(ready), first_word_total_ms=values(total), queue_ms=values(queue),
                native_stages=stages, serial_native_ms_per_frame=values(serial)), outputs, manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();before,b,mb=analyze(a.before);after,c,mc=analyze(a.after)
    assert set(b)==set(c) and mb['source_manifest_sha256']==mc['source_manifest_sha256']
    assert mb['tail_seconds']==mc['tail_seconds'] and mb['image_protocol']==mc['image_protocol']
    assert all(b[k]['video_sha256']==c[k]['video_sha256'] for k in b)
    comparisons=[dict(sample_id=k,semantic_events_identical=b[k]['semantic']==c[k]['semantic'],
                      sentences_before=b[k]['sentences'],sentences_after=c[k]['sentences']) for k in b]
    report=dict(before=before,after=after,comparisons=comparisons,
                semantic_identical=sum(row['semantic_events_identical'] for row in comparisons),
                scope='same_fps_jpeg_video_replay_not_physical_camera_or_product_accuracy')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='comparisons'},ensure_ascii=True))


if __name__=='__main__': main()
