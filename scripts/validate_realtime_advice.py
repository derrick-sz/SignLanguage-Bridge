"""Exercise optional advice on actual prediction-only records, without GT."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from slbridge.llm_assistance import AdviceQueue,LLMConfig
from experiments.p1.gt_guard import assert_gt_clean


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/realtime-csl-engineering-20261109/product-ws-32-v2/outputs')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    if not out.is_relative_to(ROOT/'artifacts'):raise ValueError('Use an artifact output directory')
    if out.exists():raise ValueError('Use a fresh output directory')
    config=LLMConfig(enabled=True)
    q=AdviceQueue(config);assert q.enabled,'Configured LLM credentials unavailable'
    out.mkdir(parents=True)
    results=[]
    try:
        for index,sample_id in enumerate(['S000257_P0000_T00','S000295_P0000_T00','S000275_P0004_T00']):
            path=source/f'{sample_id}.json'
            record=json.loads(path.read_text(encoding='utf-8'))
            assert_gt_clean(record)
            sentence=[e for e in record['events'] if e['type']=='sentence' and e['status']=='committed'][-1]
            payload=dict(glosses=sentence['predicted_gloss'],text=sentence['text'])
            assert_gt_clean(payload)
            job=q.submit(session_id=f'prediction-smoke-{index}',source_epoch=0,utterance_id=0,revision=1,**payload)
            deadline=time.monotonic()+config.timeout_seconds+2
            while q.get(job)['status'] in {'pending','running'} and time.monotonic()<deadline:
                time.sleep(.05)
            result=dict(sample_id=sample_id,source_prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input=payload,job=q.get(job),config=asdict(config),credential_ref=config.credentials_ref)
            assert_gt_clean(result)
            (out/f'{sample_id}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
            results.append(result)
            print(json.dumps({'sample_id':sample_id,'status':result['job']['status'],'reason':result['job'].get('reason'),
                  'latency_ms':result['job'].get('latency_ms')},ensure_ascii=True),flush=True)
    finally:
        q.close()
    summary=dict(n=len(results),statuses=[r['job']['status'] for r in results],
                 credential_value_saved=False,references_read=False,automatic_replacement=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__':main()
