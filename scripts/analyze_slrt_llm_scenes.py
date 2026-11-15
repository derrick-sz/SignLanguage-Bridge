"""Evaluation-only reporting. Reads paired labels; never calls a model."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.p1.eval import metrics as m


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True)
    run=p.parse_args().run_dir.resolve()
    if not run.is_relative_to(ROOT/'artifacts'):
        raise ValueError('Evaluation outputs must stay under repository artifacts')
    records=json.loads((run/'eval/records.json').read_text(encoding='utf-8'))
    summary=json.loads((run/'eval/summary.json').read_text(encoding='utf-8'))
    manifest=json.loads((run/'manifest.json').read_text(encoding='utf-8'))
    by_id={r['sample_id']:r for r in records}
    assert len(by_id)==len(records)==30
    accepted=[r for r in records if r['llm']['status']=='draft']
    for r in records:
        a=m.edit_similarity(m.normalize_text(r['ref_sentence_upstream']),m.normalize_text(r['raw_text']))
        b=m.edit_similarity(m.normalize_text(r['ref_sentence_upstream']),m.normalize_text(r['llm']['text']))
        r['char_edit_delta']=b-a
    analysis=dict(n=30,unique_reference_sentences=len({r['gt_sentence_official'] for r in records}),
        draft_n=len(accepted),draft_coverage=len(accepted)/len(records),
        same_accepted_subset=dict(n=len(accepted),
            raw=m.corpus_metrics_cslt([(r['ref_sentence_upstream'],r['raw_text']) for r in accepted]),
            organized=m.corpus_metrics_cslt([(r['ref_sentence_upstream'],r['llm']['text']) for r in accepted])),
        char_edit_delta_all_requests=m.bootstrap_by_group(records,
            lambda xs:statistics.mean(r['char_edit_delta'] for r in xs),seed=20261109),
        char_edit_delta_accepted_subset=m.bootstrap_by_group(accepted,
            lambda xs:statistics.mean(r['char_edit_delta'] for r in xs),seed=20261109),
        per_scene={scene:dict(n=sum(r['scene']==scene for r in records),
            word_errors=m.corpus_wer([(r['gt_gloss'],' '.join(r['predicted_gloss'])) for r in records if r['scene']==scene]),
            llm_status=Counter(r['llm']['status'] for r in records if r['scene']==scene)) for scene in manifest['scenes']},
        evidence_format_postaudit=dict(empty_evidence_drafts=sum(not r['llm'].get('used_token_indices') for r in accepted)))
    replay=[]
    for item in manifest['samples']:
        if not item['replay']:continue
        d=json.loads((run/'replay'/f"{item['sample_id']}.json").read_text(encoding='utf-8'))
        base=next(x for x in summary['replay'] if x['sample_id']==item['sample_id']).copy()
        # Historical drafts can be valid when published and later superseded.
        # Only the newest matching revision is the outcome at stream end.
        latest=max(d['calls'],key=lambda c:c['revision']) if d['calls'] else None
        assert latest is None or latest['tokens']==d['result'].get('gloss_sequence',[])
        current=latest['result'] if latest and not latest['stale'] else None
        base.update(latest_status=current['status'] if current else 'missing',
            latest_text=current['text'] if current and current['status']=='draft' else '',
            latest_result_ms=latest['completed_ms'] if latest else None,
            latest_draft_after_capture_eof_ms=(latest['completed_ms']-d['capture_end_ms'])
                if latest and current['status']=='draft' else None,
            historical_last_draft_is_obsolete=bool(base['last_draft']) and not bool(current and current['status']=='draft'))
        if item['kind']=='positive':
            base['word_errors']=m.wer_breakdown(by_id[item['sample_id']]['gt_gloss'],' '.join(base['tokens']))
            base['same_as_offline_words']=base['tokens']==by_id[item['sample_id']]['predicted_gloss']
        replay.append(base)
    positive=[r for r in replay if r['kind']=='positive']
    samples={r['sample_id']:r for r in manifest['samples']}
    import numpy as np
    analysis['replay']=dict(positive_n=len(positive),natural_eof_n=sum(r['natural_eof'] for r in positive),
        before_eof_word_n=sum(r['final_before_capture_eof'] for r in positive),
        streams_with_queue_drop=sum(r['queue_dropped']>0 for r in positive),
        queue_dropped_total=sum(r['queue_dropped'] for r in positive),
        exact_offline_word_agreement=sum(r['same_as_offline_words'] for r in positive),
        draft_before_capture_eof_n=sum(r['first_draft_ms'] is not None and r['first_draft_ms']<r['capture_end_ms'] for r in positive),
        frames_captured_total=sum(samples[r['sample_id']]['frames'] for r in positive),
        frames_received_total=sum(r['frames_received'] for r in positive),
        latest_status_counts=Counter(r['latest_status'] for r in positive),
        first_final_ms={f'p{q}':float(np.percentile([r['first_final_ms'] for r in positive if r['first_final_ms'] is not None],q)) for q in (50,95)},
        latest_valid_draft_after_capture_eof_ms={f'p{q}':float(np.percentile([r['latest_draft_after_capture_eof_ms'] for r in positive if r['latest_draft_after_capture_eof_ms'] is not None],q)) for q in (50,95)},
        corpus_word_errors=m.corpus_wer([(by_id[r['sample_id']]['gt_gloss'],' '.join(r['tokens'])) for r in positive]),
        same_six_offline_word_errors=m.corpus_wer([(by_id[r['sample_id']]['gt_gloss'],' '.join(by_id[r['sample_id']]['predicted_gloss'])) for r in positive]),
        records=replay)
    with (run/'eval/comparison.csv').open('w',encoding='utf-8-sig',newline='') as out:
        writer=csv.writer(out)
        writer.writerow(['sample_id','scene','reference_sentence','reference_gloss','recognized_gloss','llm_status','llm_text','llm_latency_ms'])
        for r in records:writer.writerow([r['sample_id'],r['scene'],r['gt_sentence_official'],r['gt_gloss'],
            ' '.join(r['predicted_gloss']),r['llm']['status'],r['llm']['text'],r['llm']['latency_ms']])
    if (run/'eval/selection_audit.json').exists():
        analysis['selection_postaudit']=json.loads((run/'eval/selection_audit.json').read_text(encoding='utf-8'))
    if (run/'eval/ai_factual_audit.json').exists():
        audit=json.loads((run/'eval/ai_factual_audit.json').read_text(encoding='utf-8'))
        annotations={r['sample_id']:r for r in audit['annotations']}
        assert set(annotations)==set(by_id)
        analysis['ai_audit_counts']=audit['counts']
        with (run/'eval/comparison-reviewed.csv').open('w',encoding='utf-8-sig',newline='') as out:
            writer=csv.writer(out)
            writer.writerow(['sample_id','scene','reference_sentence','reference_gloss','recognized_gloss','llm_status','llm_text','llm_latency_ms','AI_review_category','AI_review_note'])
            for r in records:
                a=annotations[r['sample_id']]
                writer.writerow([r['sample_id'],r['scene'],r['gt_sentence_official'],r['gt_gloss'],' '.join(r['predicted_gloss']),r['llm']['status'],r['llm']['text'],r['llm']['latency_ms'],a['category'],a['note']])
    (run/'eval/analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(analysis,ensure_ascii=True))


if __name__=='__main__':main()
