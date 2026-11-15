"""Evaluate paired review-hint runs without dropping flagged predictions."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def semantic(events):
    keys=('type','utterance_id','event_id','status','gloss','text','reason')
    return [{k:e[k] for k in keys if k in e} for e in events
            if e['type'] in {'word','sentence','utterance_reset'}]


def compare(before,after):
    bm=read(before/'manifest.json');am=read(after/'manifest.json')
    # A run may replay the frozen baseline manifest itself (train v2), or
    # replay the same source manifest as a derived baseline (test WS v5).
    expected_sources={sha(before/'manifest.json'),bm.get('source_manifest_sha256')}
    assert am['source_manifest_sha256'] in expected_sources
    old={s['sample_id']:s for s in bm['samples']}
    comparisons=[];flagged=[];controls=[]
    for sample in am['samples']:
        name=sample['sample_id'];assert sample['video_sha256']==old[name]['video_sha256']
        raw=read(before/'outputs'/f'{name}.json');b=raw.get('live') or raw
        a=read(after/'outputs'/f'{name}.json')
        sentence=lambda r:''.join(e['text'] for e in r['events']
                                  if e['type']=='sentence' and e.get('status')=='committed')
        row=dict(sample_id=name,kind=sample['kind'],request_before=b['request_status'],request_after=a['request_status'],
                 semantic_identical=semantic(b['events'])==semantic(a['events']),
                 native_text_before=sentence(b),native_text_after=sentence(a))
        comparisons.append(row)
        for e in a['events']:
            if e['type']=='sentence' and e.get('review_hint'):
                flagged.append(dict(sample_id=name,utterance_id=e['utterance_id'],text=e['text'],
                                    word_evidence=e['predicted_gloss'],hint=e['review_hint']))
        if sample['kind']!='positive': controls.append(row)
    return dict(n=len(comparisons),completed=sum(r['request_after']=='ok' for r in comparisons),
                semantic_identical=sum(r['semantic_identical'] for r in comparisons),
                flagged=flagged,controls=controls,comparisons=comparisons,
                evaluation=read(after/'eval/summary.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    report=dict(train=compare(root/'artifacts/train-application-validation-v2',a.root/'train'),
                test=compare(root/'artifacts/realtime-csl-engineering-20261109/product-ws-32-v5',a.root/'test'),
                interpretation='hints_do_not_change_native_predictions_or_remove_rows_not_semantic_accuracy')
    (a.root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for name in ('train','test'):
        r=report[name];e=r['evaluation']
        print(json.dumps(dict(group=name,n=r['n'],completed=r['completed'],semantic_identical=r['semantic_identical'],
              flagged=r['flagged'],text_metrics=e['text_metrics'],first_word_total_ms=e['first_word_total_ms']),ensure_ascii=True))


if __name__=='__main__':main()
