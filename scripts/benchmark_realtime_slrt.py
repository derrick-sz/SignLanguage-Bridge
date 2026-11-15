"""Same-input diagnostic ablations; references are read only by evaluate()."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.p1.gt_guard import assert_gt_clean


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, data, evaluation=False):
    if not evaluation:
        assert_gt_clean(data, context=str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def run(source, out, variants=None):
    import numpy as np
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend
    from slbridge.slrt.capture import SLRTCaptureSession
    if (out/'manifest.json').exists():
        raise ValueError('Use a fresh output directory')
    samples = [s for s in read(source/'manifest.json')['samples'] if s['replay']]
    configs = [('baseline', 'float32', 1, 1, 7), ('fp16', 'float16', 1, 1, 7),
               ('detect5', 'float16', 5, 1, 7), ('stride2', 'float16', 5, 2, 7),
               ('stride3', 'float16', 5, 3, 7), ('stride3vote5', 'float16', 5, 3, 5)]
    if variants:
        configs = [c for c in configs if c[0] in variants]
    save(out/'manifest.json', dict(protocol='slrt-throughput-ablation-v1',
         source_manifest_sha256=sha(source/'manifest.json'), samples=samples,
         code_sha256={p: sha(ROOT/p) for p in ['scripts/benchmark_realtime_slrt.py',
             'slbridge/slrt/runtime.py', 'slbridge/slrt/assets.py', 'slbridge/slrt/pose.py',
             'slbridge/slrt/stream.py', 'slbridge/slrt/capture.py']},
         variants=configs, data_scope='retrospective_seen_1086_diagnostic_not_blind'))
    for variant, precision, detection, stride, votes in configs:
        config = SLRTConfig(device='cuda', pose_device='cuda', pose_backend='rtmpose-experimental',
                            precision=precision, pose_detect_stride=detection, stride=stride, vote_size=votes)
        backend = SLRTBackend(config)
        active_session = None
        try:
            begin = time.perf_counter()
            backend.runtime.predict(np.zeros((16, 320, 320, 3), np.uint8), np.zeros((16, 133, 3), np.float32))
            save(out/variant/'runtime.json', dict(config=config.to_dict(), runtime=backend.runtime.info,
                 warmup_ms=(time.perf_counter()-begin)*1000))
            for sample in samples:
                assert sha(sample['video']) == sample['video_sha256']
                complete = backend.recognize_video(sample['video'])
                save(out/variant/'offline'/f"{sample['sample_id']}.json", complete)
                events = []
                started = time.perf_counter()
                def sink(event):
                    events.append(dict(event, delivered_ms=(time.perf_counter()-started)*1000))
                active_session = SLRTCaptureSession(backend, video=sample['video'], event_sink=sink)
                active_session.start()
                deadline = time.perf_counter()+60
                while active_session.is_running and time.perf_counter() < deadline:
                    time.sleep(.02)
                timed_out = active_session.is_running
                active_session.stop(timeout=10)
                if active_session.is_running:
                    raise RuntimeError('Native capture did not stop; keep backend alive')
                result = active_session.result()
                save(out/variant/'replay'/f"{sample['sample_id']}.json", dict(sample_id=sample['sample_id'],
                     timed_out=timed_out, result=result, events=events,
                     total_wall_ms=(time.perf_counter()-started)*1000))
                print(json.dumps(dict(variant=variant, sample_id=sample['sample_id'],
                     drops=result.get('queue_dropped'), error=result.get('error'),
                     gloss=result.get('gloss_sequence'), first_ms=result.get('first_stable_wall_latency_ms')),
                     ensure_ascii=True), flush=True)
        finally:
            if active_session is None or not active_session.is_running:
                backend.close()


def evaluate(source, out):
    import numpy as np
    from experiments.p1.eval import metrics as m
    manifest = read(out/'manifest.json')
    assert sha(source/'manifest.json') == manifest['source_manifest_sha256']
    refs = {r['sample_id']: r for r in read(source/'eval/records.json')}
    summaries = {}
    for variant, *_ in manifest['variants']:
        pairs = []; offline_pairs = []; cases = []; first = []
        for sample in manifest['samples']:
            offline = read(out/variant/'offline'/f"{sample['sample_id']}.json")
            replay = read(out/variant/'replay'/f"{sample['sample_id']}.json")
            r = replay['result']
            assert not replay['timed_out'] and r['error'] is None and r['status'] == 'final'
            row = dict(sample_id=sample['sample_id'], frames=sample['frames'],
                       drops=r['queue_dropped'], offline=offline['gloss_sequence'], replay=r['gloss_sequence'],
                       first_ms=r.get('first_stable_wall_latency_ms'), timings=r['timings'])
            if sample['kind'] == 'positive':
                target = refs[sample['sample_id']]['gt_gloss']
                pairs.append((target, ' '.join(r['gloss_sequence'])))
                offline_pairs.append((target, ' '.join(offline['gloss_sequence'])))
                if row['first_ms'] is not None:
                    first.append(row['first_ms'])
            cases.append(row)
        summaries[variant] = dict(replay_wer=m.corpus_wer(pairs), offline_wer=m.corpus_wer(offline_pairs),
             dropped_frames=sum(r['drops'] for r in cases),
             same_words=sum(r['offline'] == r['replay'] for r in cases),
             first_ms={f'p{q}': float(np.percentile(first, q)) if first else None for q in (50,95)}, cases=cases)
    save(out/'eval/summary.json', summaries, evaluation=True)
    print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in summaries.items()},ensure_ascii=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['run','evaluate'])
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variants',nargs='+',choices=['baseline','fp16','detect5','stride2','stride3','stride3vote5'])
    a = parser.parse_args()
    if not a.output.resolve().is_relative_to(ROOT/'artifacts'):
        parser.error('Output must be a fresh artifacts directory')
    if a.phase=='run':run(a.source_run.resolve(),a.output.resolve(),a.variants)
    else:evaluate(a.source_run.resolve(),a.output.resolve())
