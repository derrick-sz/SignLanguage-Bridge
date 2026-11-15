"""Compare the live Uni-Sign route on the exact SLRT/LLM experiment videos."""
import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.p1.gt_guard import assert_gt_clean

EXPECTED='uni-sign-csl-daily-pose-only-182913ca'
CHECKPOINT='73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,data,*,evaluation=False):
    if not evaluation:assert_gt_clean(data,context=str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def infer(source,out):
    import httpx
    manifest=read(source/'manifest.json')
    if (out/'manifest.json').exists():raise ValueError('Use a new output directory; no silent resume')
    with httpx.Client(base_url='http://127.0.0.1:8765',timeout=100,trust_env=False) as client:
        health=client.get('/v3/health').json()
        active=health['status']['csl']['active']
        assert active['model_id']==EXPECTED and active['task_type']=='text'
        assert active['runtime']['checkpoint_sha256']==CHECKPOINT
        expected_version=health['capabilities']['csl']['model_version']
        identity=dict(model_id=EXPECTED,model_version=expected_version,checkpoint_sha256=CHECKPOINT,
            generation=active['generation'],runtime=active['runtime'])
        write(out/'manifest.json',dict(created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
            parent_manifest=str(source/'manifest.json'),parent_manifest_sha256=digest(source/'manifest.json'),
            source_scope=manifest['data_scope'],samples=manifest['samples'],identity=identity,
            code_sha256=digest(__file__),kind='same_input_unisign_completed_clip_comparison'))
        for item in manifest['samples']:
            assert digest(item['video'])==item['video_sha256']
            started=time.perf_counter();attempts=0
            try:
                while True:
                    attempts+=1
                    negotiated=client.post('/v3/models/negotiate',json={'sign_language':'csl','mode':'clip'})
                    negotiated.raise_for_status();n=negotiated.json();c=n['input_contract'];cap=n['capabilities']
                    assert cap['model_id']==EXPECTED and cap['model_version']==expected_version
                    assert n['model_generation']==identity['generation']
                    fields=dict(sign_language='csl',mode='clip',request_id='scene-unisign-'+uuid.uuid4().hex,
                        representation_id=c['representation_id'],preprocessing_version=c['preprocessing_version'],
                        task_type='text',expected_generation=str(n['model_generation']))
                    with Path(item['video']).open('rb') as stream:
                        response=client.post('/v3/recognitions',data=fields,
                            files={'file':(Path(item['video']).name,stream,'video/mp4')})
                    if response.status_code==429 and attempts<4:
                        time.sleep(1);continue
                    response.raise_for_status();data=response.json()
                    assert data['model_id']==EXPECTED and data['model_version']==expected_version
                    assert data['model_generation']==identity['generation']
                    assert data['preprocessing_version']==c['preprocessing_version']
                    assert data['result']['task_type']=='text' and data['result']['sign_language']=='csl'
                    record=dict(sample_id=item['sample_id'],request_status='ok',response=data,
                        text=(data['result'].get('prediction') or {}).get('text',''),attempts=attempts,
                        http_ms=(time.perf_counter()-started)*1000)
                    break
            except Exception as error:
                record=dict(sample_id=item['sample_id'],request_status='error',text='',attempts=attempts,
                    error_type=type(error).__name__,http_ms=(time.perf_counter()-started)*1000)
            write(out/'outputs'/f"{item['sample_id']}.json",record)
            print(json.dumps({k:record[k] for k in ('sample_id','request_status','text','http_ms')},ensure_ascii=True),flush=True)
            time.sleep(.1)


def evaluate(source,out):
    from experiments.p1.eval import metrics as m
    import numpy as np
    manifest=read(out/'manifest.json')
    assert digest(source/'manifest.json')==manifest['parent_manifest_sha256']
    original={r['sample_id']:r for r in read(source/'eval/records.json')}
    records=[];controls=[]
    for item in manifest['samples']:
        record=read(out/'outputs'/f"{item['sample_id']}.json")
        if item['kind']!='positive':controls.append(record);continue
        old=original[item['sample_id']]
        row=dict(old,unisign=record)
        target=m.normalize_text(row['ref_sentence_upstream'])
        row['unisign_edit']=m.edit_similarity(target,m.normalize_text(record['text']))
        row['raw_edit']=m.edit_similarity(target,m.normalize_text(row['raw_text']))
        row['organized_edit']=m.edit_similarity(target,m.normalize_text(row['llm']['text']))
        records.append(row)
    assert len(records)==30
    def metrics(rows,key):
        return m.corpus_metrics_cslt([(r['ref_sentence_upstream'],key(r)) for r in rows])
    accepted=[r for r in records if r['llm']['status']=='draft']
    def difference(rows,other):
        return m.bootstrap_by_group(rows,lambda group:statistics.mean(r['unisign_edit']-r[other] for r in group),seed=20261109)
    summary=dict(n=len(records),request_errors=sum(r['unisign']['request_status']=='error' for r in records),
        nonempty=sum(bool(r['unisign']['text'].strip()) for r in records),
        unisign=metrics(records,lambda r:r['unisign']['text']),
        slrt_gloss_join=metrics(records,lambda r:r['raw_text']),
        slrt_llm_all_requests=metrics(records,lambda r:r['llm']['text']),
        same_llm_accepted_subset=dict(n=len(accepted),unisign=metrics(accepted,lambda r:r['unisign']['text']),
            llm=metrics(accepted,lambda r:r['llm']['text'])),
        unisign_edit_delta_vs_slrt_join=difference(records,'raw_edit'),
        unisign_edit_delta_vs_llm_all=difference(records,'organized_edit'),
        unisign_edit_delta_vs_llm_accepted=difference(accepted,'organized_edit'),
        per_sample_vs_llm_all=CounterComparison(records,'organized_edit'),
        per_sample_vs_llm_accepted=CounterComparison(accepted,'organized_edit'),
        http_ms={f'p{q}':float(np.percentile([r['unisign']['http_ms'] for r in records],q)) for q in (50,95)},
        controls=controls,data_scope=manifest['source_scope'],camera_and_new_signers='NOT_RUN')
    write(out/'eval/records.json',records,evaluation=True)
    write(out/'eval/summary.json',summary,evaluation=True)
    with (out/'eval/comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','scene','reference_sentence','recognized_gloss','SLRT_LLM_status','SLRT_LLM_text','UniSign_text','UniSign_http_ms'])
        for r in records:w.writerow([r['sample_id'],r['scene'],r['gt_sentence_official'],' '.join(r['predicted_gloss']),r['llm']['status'],r['llm']['text'],r['unisign']['text'],r['unisign']['http_ms']])
    print(json.dumps({k:v for k,v in summary.items() if k!='controls'},ensure_ascii=True))


def CounterComparison(rows,other):
    delta=[r['unisign_edit']-r[other] for r in rows]
    return dict(better=sum(x>1e-9 for x in delta),same=sum(abs(x)<=1e-9 for x in delta),worse=sum(x< -1e-9 for x in delta))


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['infer','evaluate'])
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();source=args.source_run.resolve();out=args.output.resolve()
    if not source.is_relative_to(ROOT/'artifacts') or not out.is_relative_to(ROOT/'artifacts') or source==out:
        raise ValueError('Use distinct artifact input and output directories')
    out.mkdir(parents=True,exist_ok=True)
    globals()[args.phase](source,out)


if __name__=='__main__':main()
