"""Verify local demo bytes and run frame-only WS; labels enter evaluation only."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from validate_realtime_pipeline import one, read, save
from experiments.p1.gt_guard import assert_gt_clean
from experiments.p1.eval.metrics import normalize_text


async def run(args):
    import httpx
    base='http://127.0.0.1:8879';folder=args.root.resolve()
    local=read(folder/'catalog.json');outputs=folder/'recognition'
    if outputs.exists():raise ValueError('Use fresh recognition output')
    outputs.mkdir()
    with httpx.Client(trust_env=False,timeout=10) as client:
        public=client.get(base+'/v1/communication/demos').json()
        status=client.get(base+'/v1/communication/realtime').json()
        assert status['ready'] and not status['busy'] and public['enabled']
        assert public['catalog_sha256']==hashlib.sha256((folder/'catalog.json').read_bytes()).hexdigest()
        for row in public['items']:
            response=client.get(base+row['media_url']);response.raise_for_status()
            assert hashlib.sha256(response.content).hexdigest()==row['sha256']
            partial=client.get(base+row['media_url'],headers={'Range':'bytes=0-15'})
            assert partial.status_code==206 and partial.content==response.content[:16]
    (folder/'catalog-response.json').write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding='utf-8')
    results=[]
    for index,row in enumerate(local['items']):
        sample=dict(sample_id=row['id'],video=str(folder/row['media']),fps=row['fps'],frames=row['frames'])
        assert_gt_clean(sample,context='demo visual inference input')
        save(outputs/f"{row['id']}-input.json",sample)
        result=await one(sample,'ws://127.0.0.1:8879/v1/communication/realtime',index+1,0)
        save(outputs/f"{row['id']}-output.json",result)
        sentences=[e for e in result['events'] if e['type']=='sentence' and e.get('status')=='committed']
        text=''.join(e['text'] for e in sentences)
        first=next((e['received_ms'] for e in result['events'] if e['type']=='word' and e.get('status')=='partial'),None)
        summary=dict(id=row['id'],kind=row['kind'],request_status=result['request_status'],text=text,
                     normalized_label_match=normalize_text(text)==normalize_text(row['annotation']),
                     first_word_total_ms=first+result['ready_wait_ms'] if first is not None else None,
                     review_hints=[e['review_hint'] for e in sentences if e.get('review_hint')],
                     frame_gaps=next((e['dropped_input_frames'] for e in result['events'] if e['type']=='summary'),None))
        results.append(summary);print(json.dumps(summary,ensure_ascii=True),flush=True)
    save(outputs/'evaluation.json',dict(results=results,scope='selected_train_h264_replay_not_new_signer_accuracy'),evaluation=True)
    assert all(r['request_status']=='ok' for r in results)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();asyncio.run(run(a))
