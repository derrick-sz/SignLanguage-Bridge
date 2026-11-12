"""Independent beam-score diagnostic on frozen live utterances; no product changes.

Stages must run sequentially: prepare -> extract -> infer -> evaluate. References
are opened only by evaluate. Frozen historical boundaries are not new streaming
segmentation or camera evidence.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# The existing benchmark imports another local script by its CLI module name.
sys.path.insert(1, str(ROOT / 'scripts'))
from experiments.p1.gt_guard import assert_gt_clean
from scripts.validate_realtime_pipeline import read, save, sha

SOURCES = {group: ROOT / 'artifacts/realtime-utterance-optimization-20261111' / group
           for group in ('train', 'test')}
SLRT_CONFIG = ROOT / 'artifacts/realtime-csl-engineering-20261109/slrt-local.json'
SCORE_KIND = 'length_penalized_beam_log_score_not_correctness_probability'
EXECUTION_SOURCE_SHA256 = sha(__file__)


def verify_checkpoint(config):
    actual_bytes = config.checkpoint.stat().st_size
    actual_sha256 = sha(config.checkpoint)
    if actual_bytes != config.checkpoint_bytes or actual_sha256 != config.checkpoint_sha256:
        raise ValueError('Actual Uni-Sign checkpoint does not match the pinned bundle')
    return dict(expected_bytes=config.checkpoint_bytes, actual_bytes=actual_bytes,
                expected_sha256=config.checkpoint_sha256, actual_sha256=actual_sha256,
                bundle_manifest_sha256=sha(config.bundle_dir/'manifest.json'))


def segments(record):
    if record['request_status'] != 'ok':
        raise ValueError('Historical replay did not complete')
    result = []
    for event in record['events']:
        if event['type'] != 'sentence' or event.get('status') != 'committed':
            continue
        first, last = event['frame_start'], event['frame_end']
        if type(first) is not int or type(last) is not int or not 0 <= first <= last:
            raise ValueError('Invalid frozen sentence boundary')
        result.append(dict(utterance_id=event['utterance_id'], first=first, last=last,
                           historical_text=event['text'], reason=event['reason']))
    return result


def prepare(output):
    if output.exists() or not output.is_relative_to(ROOT / 'artifacts'):
        raise ValueError('Use a fresh output under artifacts')
    samples, sources = [], {}
    for group, source in SOURCES.items():
        manifest = read(source / 'manifest.json')
        sources[group] = dict(path=str(source), sha256=sha(source / 'manifest.json'))
        positives = [row for row in manifest['samples'] if row['kind'] == 'positive']
        if len(positives) != 30 or len(manifest['samples']) != 32:
            raise ValueError('Expected the complete frozen 30+2 batch')
        for row in manifest['samples']:
            path = source / 'outputs' / (row['sample_id'] + '.json')
            if sha(row['video']) != row['video_sha256']:
                raise ValueError('Frozen video identity changed')
            old = read(path)
            samples.append(dict(group=group, sample_id=row['sample_id'], kind=row['kind'],
                video=row['video'], video_sha256=row['video_sha256'], fps=row['fps'], frames=row['frames'],
                tail_seconds=manifest['tail_seconds'], baseline_output=str(path), baseline_sha256=sha(path),
                segments=segments(old)))
    save(output / 'manifest.json', dict(protocol='unisign_beam_score_diagnostic_v1',
        created_at=datetime.now(timezone.utc).isoformat(), sources=sources, samples=samples,
        slrt_config=str(SLRT_CONFIG), slrt_config_sha256=sha(SLRT_CONFIG),
        script_sha256=sha(__file__), data_scope='frozen_seen_train_and_test_not_blind_or_semantic_evaluation',
        image_protocol='original_512_square_opencv_jpeg90_FrameDecoder_shared320_ref512',
        model_input_excludes_reference=True))
    print(json.dumps(dict(samples=len(samples), positive=60, controls=4)))


def memory_preflight(output, stage):
    from scripts.benchmark_unisign_generation import windows_memory_budget
    budget = windows_memory_budget()
    save(output / f'memory-{stage}-before.json', budget)
    if budget is not None and budget['commit_free_bytes'] < 10 * 1024**3:
        raise RuntimeError('Need 10 GiB free Windows commit; release only the verified idle task services first')


def extract(output):
    memory_preflight(output, 'extract')
    import cv2
    import numpy as np
    from slbridge.realtime_protocol import FrameDecoder
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import video_packets
    from slbridge.slrt.pose import RTMPoseExperimentalFrontend
    manifest = read(output / 'manifest.json')
    if (output / 'pose-manifest.json').exists():
        raise ValueError('Extraction already exists; use a fresh diagnostic')
    if sha(manifest['slrt_config']) != manifest['slrt_config_sha256']:
        raise ValueError('SLRT config identity changed')
    config = SLRTConfig.from_file(manifest['slrt_config'])
    frontend = RTMPoseExperimentalFrontend(config)
    cases = []
    try:
        frontend.extractor.warmup_live()
        for sample in manifest['samples']:
            if sha(sample['video']) != sample['video_sha256'] or sha(sample['baseline_output']) != sample['baseline_sha256']:
                raise ValueError('Frozen source changed')
            frontend.extractor.reset_detection()
            decoder = FrameDecoder(source_epoch=1, source_id='score-diagnostic')
            points = []
            def one(image, index, stamp):
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise ValueError('JPEG encoding failed')
                frame = decoder.decode(dict(type='frame', index=index, timestamp=stamp, source_epoch=1,
                    capture_ms=stamp*1000, jpeg=base64.b64encode(encoded).decode('ascii'))).packet
                value = frontend.extract(frame.image, frame_index=index, timestamp=stamp)
                points.append(value.copy() if value is not None else None)
            packets = video_packets(sample['video'])
            origin = last = None
            try:
                for packet in packets:
                    if origin is None:
                        origin = packet.timestamp
                    one(packet.image, packet.frame_index, packet.timestamp-origin)
                    last = packet
            finally:
                packets.close()
            if last is None or len(points) != sample['frames']:
                raise ValueError('Decoded frame count differs from frozen input')
            for offset in range(round(sample['tail_seconds']*sample['fps'])):
                one(last.image, last.frame_index+1+offset, sample['frames']/sample['fps']+offset/sample['fps'])
            chosen = list(sample['segments'])
            # A stationary control can still yield fluent text if translated
            # directly. This explicit bypass tests score behavior, not live gating.
            if sample['kind'] != 'positive' and not chosen and all(p is not None for p in points):
                chosen.append(dict(utterance_id=-1, first=0, last=len(points)-1,
                    historical_text=None, reason='direct_control_pose_not_live_translation'))
            extracted = []
            for segment in chosen:
                selected = points[segment['first']:segment['last']+1]
                if len(selected) != segment['last']-segment['first']+1 or any(p is None for p in selected):
                    raise ValueError('Frozen translated segment contains newly invalid poses')
                name = f"{sample['group']}--{sample['sample_id']}--u{segment['utterance_id']}"
                path = output / 'poses' / (name+'.npz')
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, points=np.stack(selected), indices=np.arange(segment['first'], segment['last']+1))
                extracted.append(dict(segment, id=name, pose_file=str(path), pose_sha256=sha(path), frames=len(selected)))
            case = dict(group=sample['group'], sample_id=sample['sample_id'], kind=sample['kind'],
                frames=len(points), invalid_pose_frames=sum(p is None for p in points), segments=extracted)
            cases.append(case)
            print(json.dumps(dict(group=sample['group'], sample_id=sample['sample_id'], segments=len(extracted))), flush=True)
    finally:
        frontend.close()
    save(output / 'pose-manifest.json', dict(source_manifest_sha256=sha(output/'manifest.json'),
        frontend=frontend.info, cases=cases, newly_extracted=True, frozen_boundaries=True))


def score_metadata(result, *, length_penalty):
    scores = result.sequences_scores.detach().cpu().tolist()
    if len(scores) != 1 or not math.isfinite(scores[0]) or scores[0] > 1e-5:
        raise ValueError('Expected one finite nonpositive beam log score')
    if result.sequences.ndim != 2 or result.sequences.shape[0] != 1 or result.sequences.shape[1] <= 1 or not result.scores:
        raise ValueError('Invalid beam generation output')
    return dict(beam_score=float(scores[0]), score_kind=SCORE_KIND, length_penalty=float(length_penalty),
                generated_tokens=int(result.sequences.shape[1]-1), score_steps=len(result.scores),
                temporary_score_bytes=sum(row.numel()*row.element_size() for row in result.scores))


@contextmanager
def generation_capture(model, *, scored):
    """Wrap only this diagnostic's mT5 instance, restoring any previous override."""
    target = model.mt5_model
    absent = object()
    previous = target.__dict__.get('generate', absent)
    original = target.generate
    captured = {}
    def generate(*args, **kwargs):
        if scored:
            if 'output_scores' in kwargs or 'return_dict_in_generate' in kwargs:
                raise ValueError('Unexpected upstream score override')
            result = original(*args, **kwargs, output_scores=True, return_dict_in_generate=True)
            captured.update(score_metadata(result, length_penalty=target.generation_config.length_penalty))
            sequences = result.sequences
        else:
            sequences = original(*args, **kwargs)
        captured['tokens'] = sequences.detach().cpu().tolist()
        return sequences
    target.generate = generate
    try:
        yield captured
    finally:
        if previous is absent:
            del target.generate
        else:
            target.generate = previous


def infer(output, *, repeats):
    memory_preflight(output, 'infer')
    import numpy as np
    from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig
    from slbridge.uni_sign_preprocess import normalize_rtmpose_pixels, prepare_pose_window
    if importlib.metadata.version('transformers') != '4.40.0':
        raise RuntimeError('This diagnostic was inspected for Transformers 4.40.0 only')
    poses = read(output / 'pose-manifest.json')
    if poses['source_manifest_sha256'] != sha(output/'manifest.json') or (output/'outputs.jsonl').exists():
        raise ValueError('Source changed or inference already started; use fresh output')
    config = UniSignConfig.from_bundle(ROOT/'models/external/uni_sign/model_bundle')
    checkpoint_identity = verify_checkpoint(config)
    runtime = TorchUniSignRuntime(config, device='cuda', dtype='bfloat16')
    try:
        runtime.load()
        import torch
        torch.set_num_threads(4)
        save(output/'runtime.json', dict(runtime=runtime.runtime_info, checkpoint=checkpoint_identity,
            runtime_source_sha256=sha(ROOT/'slbridge/uni_sign.py'), upstream_source_sha256=sha(config.source_dir/'models.py'),
            script_sha256=EXECUTION_SOURCE_SHA256,
            transformers=importlib.metadata.version('transformers'), repeats=repeats,
            num_beams=4, max_new_tokens=100, score_kind=SCORE_KIND))
    except BaseException:
        runtime.close()
        raise
    rows = []
    try:
        for case in poses['cases']:
            for segment in case['segments']:
                if sha(segment['pose_file']) != segment['pose_sha256']:
                    raise ValueError('Pose input changed')
                with np.load(segment['pose_file'], allow_pickle=False) as data:
                    points = data['points']
                    assert np.array_equal(data['indices'], np.arange(segment['first'], segment['last']+1))
                normalized = normalize_rtmpose_pixels(points[:,:,:2], 512, 512, mode='isotropic_short_side')
                pose = prepare_pose_window(normalized, points[:,:,2], max_frames=256)
                for scored in (False, True):
                    with generation_capture(runtime._model, scored=scored):
                        runtime.infer(pose, num_beams=4, max_new_tokens=100)
                for repeat in range(repeats):
                    pair = []
                    for scored in ((False, True) if repeat % 2 == 0 else (True, False)):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        with generation_capture(runtime._model, scored=scored) as captured:
                            started = time.perf_counter()
                            text, timings = runtime.infer(pose, num_beams=4, max_new_tokens=100)
                            elapsed = (time.perf_counter()-started)*1000
                        row = dict(group=case['group'], sample_id=case['sample_id'], kind=case['kind'],
                            segment_id=segment['id'], reason=segment['reason'], repeat=repeat,
                            mode='scored' if scored else 'baseline', text=text, **captured, **timings,
                            wall_ms=elapsed, peak_extra_bytes=torch.cuda.max_memory_allocated()-allocated,
                            historical_text_identical=None if segment['historical_text'] is None else text==segment['historical_text'])
                        assert_gt_clean(row, context='beam-score diagnostic output')
                        with (output/'outputs.jsonl').open('a',encoding='utf-8') as stream:
                            stream.write(json.dumps(row,ensure_ascii=False)+'\n')
                        pair.append(row);rows.append(row)
                    if pair[0]['tokens'] != pair[1]['tokens'] or pair[0]['text'] != pair[1]['text']:
                        raise RuntimeError('Score collection changed generated tokens/text')
                print(json.dumps(dict(segment_id=segment['id'], text=text, paired_tokens_identical=True)), flush=True)
    finally:
        runtime.close()
    summary = dict(pairs=len(rows)//2, segments=len({r['segment_id'] for r in rows}), paired_tokens_identical=True,
        historical_text_checks=sum(r['historical_text_identical'] is not None for r in rows),
        historical_text_identical=sum(r['historical_text_identical'] is True for r in rows), modes={})
    for mode in ('baseline', 'scored'):
        group = [row for row in rows if row['mode']==mode]
        summary['modes'][mode] = {key: dict(mean=float(np.mean([r[key] for r in group])),
            p50=float(np.percentile([r[key] for r in group],50)),p95=float(np.percentile([r[key] for r in group],95)))
            for key in ('wall_ms','model_total_ms','generation_ms','peak_extra_bytes')}
    save(output/'summary.json',summary)


def auc(bad_scores, good_scores):
    if not bad_scores or not good_scores:
        return None
    return sum((a>b)+.5*(a==b) for a in bad_scores for b in good_scores)/(len(bad_scores)*len(good_scores))


def evaluate(output):
    """Only this worker reads quality proxies; none are model inputs."""
    from experiments.p1.eval.metrics import normalize_text
    manifest = read(output/'manifest.json')
    poses = read(output/'pose-manifest.json')
    summary = read(output/'summary.json')
    assert summary['paired_tokens_identical']
    rows = [json.loads(line) for line in (output/'outputs.jsonl').read_text(encoding='utf-8').splitlines()]
    scored = [row for row in rows if row['mode']=='scored' and row['repeat']==0]
    by_segment = {row['segment_id']:row for row in scored}
    assert len(rows) == summary['pairs']*2
    assert len(by_segment) == sum(len(case['segments']) for case in poses['cases'])
    results, controls = [], []
    for group, source in SOURCES.items():
        assert sha(source/'manifest.json') == manifest['sources'][group]['sha256']
        refs = {row['sample_id']:row for row in read(source/'eval/records.json')}
        for case in (case for case in poses['cases'] if case['group']==group):
            predictions = [by_segment[segment['id']] for segment in case['segments']]
            if case['kind'] != 'positive':
                controls.append(dict(group=group,sample_id=case['sample_id'],segments=len(predictions),
                    bypass_live_gating=bool(predictions),predictions=predictions))
                continue
            record = refs[case['sample_id']]
            target = record['ref_sentence_upstream']
            text = ''.join(row['text'] for row in predictions)
            values = [row['beam_score'] for row in predictions]
            results.append(dict(group=group,sample_id=case['sample_id'],prediction=text,reference=target,
                exact=normalize_text(text)==normalize_text(target),
                minimum_beam_score=min(values) if values else None,predictions=predictions))
    report = dict(scope='retrospective_exact_match_proxy_not_human_semantics_or_calibrated_probability',groups={},controls=controls,
        inputs=dict(model_outputs_sha256=sha(output/'outputs.jsonl'),pose_manifest_sha256=sha(output/'pose-manifest.json'),
            reference_sha256={group:sha(source/'eval/records.json') for group,source in SOURCES.items()},
            evaluation_script_sha256=EXECUTION_SOURCE_SHA256))
    for group in SOURCES:
        cases = [row for row in results if row['group']==group]
        bad = [-row['minimum_beam_score'] for row in cases if not row['exact'] and row['minimum_beam_score'] is not None]
        good = [-row['minimum_beam_score'] for row in cases if row['exact'] and row['minimum_beam_score'] is not None]
        report['groups'][group] = dict(n=len(cases),exact=sum(row['exact'] for row in cases),
            scored_cases=len(bad)+len(good),risk_auc_exact_proxy=auc(bad,good),records=cases)
    save(output/'eval/report.json',report,evaluation=True)
    print(json.dumps({name:{k:v for k,v in value.items() if k!='records'} for name,value in report['groups'].items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','extract','infer','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args = parser.parse_args()
    if args.repeats not in (1,2,3) or not args.output.resolve().is_relative_to(ROOT/'artifacts'):
        parser.error('Use bounded repetitions and an artifacts output')
    if args.stage=='infer':
        infer(args.output.resolve(),repeats=args.repeats)
    else:
        globals()[args.stage](args.output.resolve())
