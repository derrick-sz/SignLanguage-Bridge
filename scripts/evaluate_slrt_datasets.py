"""Batch SLRT Gloss evaluation on local datasets: per-video WER and latency.

Historical guardrails applied (docs/agents/model-evaluation-pitfalls.md):
canonical recorded orientation by default; any direction change is an explicit
per-signer table or an explicit --mirror-all diagnostic probe; each run writes
its own output directory so protocols are never mixed inside one JSONL; WER
uses the same token Levenshtein as the SLRT CLI with no text normalization and
reference OOV against the model vocabulary is reported separately, never fixed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from slbridge._io import configure_utf8_streams, canonical_hash
from slbridge.slrt.assets import SLRTConfig, load_vocabulary, sha256
from slbridge.slrt.backend import SLRTBackend, video_packets
from slbridge.slrt.cli import _wer
from slbridge.slrt.pose import make_frontend
from slbridge.slrt.preprocess import mirror_keypoints_133
from slbridge.slrt.stream import SLRTStream

LABEL_DIR = Path(r"D:\Zenodo\CE-CSL\CE-CSL\label")
VIDEO_DIR = Path(r"D:\Zenodo\CE-CSL\CE-CSL\video")


def parse_labels(split):
    path = LABEL_DIR / f"{split}.csv"
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows[0][:2] != ["Number", "Translator"]:
        raise ValueError(f"unexpected label header in {path}: {rows[0]}")
    items = []
    for row in rows[1:]:
        number, translator = row[0], row[1]
        gloss = [t.strip() for t in row[3].split("/") if t.strip()]
        video = VIDEO_DIR / split / translator / f"{number}.mp4"
        if not video.is_file():
            raise FileNotFoundError(f"label row without video: {video}")
        items.append(dict(sample_id=number, translator=translator, video=video, reference=gloss))
    return items


def stratified(items, limit):
    """Round-robin over translators (sorted ids inside each) for a balanced subset."""
    if not limit or limit >= len(items):
        return list(items)
    by_translator = {}
    for item in sorted(items, key=lambda i: i["sample_id"]):
        by_translator.setdefault(item["translator"], []).append(item)
    selected, cursor = [], 0
    while len(selected) < limit:
        progressed = False
        for name in sorted(by_translator):
            bucket = by_translator[name]
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                progressed = True
                if len(selected) == limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def direction_table(path):
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not set(data) <= set("ABCDEFGHIJKL") or not all(v in {"preserve", "mirror"} for v in data.values()):
        raise ValueError("direction table must map translator letters to preserve|mirror")
    return data


class _MirroredFrontend:
    """Receive mirrored RGB, extract original pose, then mirror it once.

    The outer packet transform and this frontend form one paired transform.
    """

    def __init__(self, inner):
        self.inner = inner

    def extract(self, bgr, *, frame_index, timestamp):
        keypoints = self.inner.extract(bgr[:, ::-1, :], frame_index=frame_index, timestamp=timestamp)
        if keypoints is None:
            return None
        return mirror_keypoints_133(keypoints[None])[0]

    def close(self):
        self.inner.close()

    @property
    def info(self):
        return dict(self.inner.info, orientation="mirror", experimental=True)


def oriented_packet(packet, *, mirror):
    if not mirror:
        return packet
    import numpy as np
    return replace(packet,image=np.ascontiguousarray(packet.image[:,::-1]))


class _NoCloseFrontend:
    """Share one frontend across per-video streams; stream.close() must not kill it."""

    def __init__(self, inner):
        self.inner = inner

    def extract(self, bgr, *, frame_index, timestamp):
        return self.inner.extract(bgr, frame_index=frame_index, timestamp=timestamp)

    def close(self):
        pass

    @property
    def info(self):
        return self.inner.info


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def run_protocol(args, config, items, table):
    root=Path(__file__).resolve().parents[1]
    sources=[Path(__file__).resolve()]+[root/'slbridge/slrt'/name for name in
        ('assets.py','pose.py','preprocess.py','network.py','_s3d_base.py','runtime.py','stream.py','decoder.py','backend.py','cli.py')]
    sources.append(root/'slbridge/uni_sign.py')
    return dict(schema_version=2,config=config.to_dict(),split=args.split,
                checkpoint_sha256=sha256(config.checkpoint_path),vocabulary_sha256=sha256(config.vocabulary_path),
                label_sha256=sha256(LABEL_DIR/f'{args.split}.csv'),
                sources={str(p.relative_to(root)):sha256(p) for p in sources},
                inputs=[dict(sample_id=i['sample_id'],video_sha256=sha256(i['video']),reference=i['reference'],
                             direction='mirror' if args.mirror_all or table.get(i['translator'],'preserve')=='mirror' else 'preserve')
                        for i in items])


def bind_runtime_protocol(path, static, runtime_info, pose_info):
    resolved={k:runtime_info[k] for k in ('device','dtype','torch_version','checkpoint_sha256','vocabulary_sha256')}
    expected=dict(static=static,resolved_runtime=resolved,pose=pose_info)
    expected['protocol_sha256']=canonical_hash(expected)
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=expected:
            raise ValueError('SLRT run protocol mismatch; use a new output directory')
    else:
        path.write_text(json.dumps(expected,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return expected['protocol_sha256']


def validate_records(records, protocol, *, split):
    """Bind every resumed/scored row to its input, protocol and recomputed WER."""
    expected = None
    if protocol is not None:
        payload = {k: v for k, v in protocol.items() if k != 'protocol_sha256'}
        if canonical_hash(payload) != protocol.get('protocol_sha256'):
            raise ValueError('Invalid run protocol hash')
        static = protocol['static']
        if static['split'] != split:
            raise ValueError('Run protocol split mismatch')
        expected = {item['sample_id']: item for item in static['inputs']}
        if len(expected) != len(static['inputs']):
            raise ValueError('Duplicate sample IDs in run protocol')
    seen = set()
    for record in records:
        sample_id = record['sample_id']
        if sample_id in seen:
            raise ValueError(f'Duplicate prediction for {sample_id}')
        seen.add(sample_id)
        if expected is not None:
            item = expected.get(sample_id)
            if item is None:
                raise ValueError(f'Prediction outside input manifest: {sample_id}')
            required = dict(split=split, run_protocol_sha256=protocol['protocol_sha256'],
                            video_sha=item['video_sha256'], reference=item['reference'],
                            direction=item['direction'])
            for field, value in required.items():
                if record.get(field) != value:
                    raise ValueError(f'Prediction {sample_id}: {field} does not match run protocol')
            if record.get('error') and record.get('wer') is not None:
                raise ValueError(f'Failed prediction must have null WER: {sample_id}')
            if not record.get('error') and record.get('status') != 'final':
                raise ValueError(f'Successful video prediction must be final: {sample_id}')
        if not record.get('error'):
            if record.get('wer') != _wer(record['reference'], record['hypothesis']):
                raise ValueError(f'Prediction WER disagrees with tokens: {sample_id}')
    return seen


def evaluate(args):
    config = SLRTConfig(pose_backend=args.pose_backend, device=args.device, pose_device=args.pose_device)
    vocabulary = load_vocabulary(config.vocabulary_path)
    vocab_glosses = set(vocabulary)
    items = stratified(parse_labels(args.split), args.limit)
    table = direction_table(args.direction_table)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / f"detail-{args.split}.jsonl"
    protocol_path=out_dir/'run-protocol.json'
    static=run_protocol(args,config,items,table)
    if detail_path.exists() and detail_path.stat().st_size:
        if not args.resume:
            raise ValueError('Evaluation output exists; use a new directory or explicit --resume')
        if not protocol_path.is_file():
            raise ValueError('Legacy predictions have no bound run protocol; use a new output directory')
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding='utf-8')).get('static')!=static:
        raise ValueError('SLRT input/model/source protocol mismatch; use a new output directory')
    done = set()
    if args.resume and detail_path.exists():
        previous = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.is_file() else None
        done = validate_records(previous, protocol, split=args.split)
        print(f"resume: {len(done)} videos already evaluated", flush=True)
    with detail_path.open("a" if args.resume else "w", encoding="utf-8") as detail:
        backend = SLRTBackend(config)
        # One pose frontend for the whole batch: re-creating it per video would
        # rebuild ONNX sessions hundreds of times and repeatedly mutate DLL paths.
        shared_frontend = None
        try:
            shared_frontend = make_frontend(config)
            protocol_id=bind_runtime_protocol(protocol_path,static,backend.runtime.info,shared_frontend.info)
        except Exception:
            if shared_frontend is not None:shared_frontend.close()
            backend.close()
            raise
        started = perf_counter()
        try:
            for item in items:
                if item["sample_id"] in done:
                    continue
                mirror = args.mirror_all or table.get(item["translator"], "preserve") == "mirror"
                inner = _MirroredFrontend(shared_frontend) if mirror else shared_frontend
                window_totals = []
                begin = perf_counter()
                error = None
                result = None
                stream = SLRTStream(backend.runtime, _NoCloseFrontend(inner), backend.config,
                                    event_sink=None,
                                    window_sink=(lambda w: window_totals.append(w["window_total_ms"])))
                try:
                    packets = video_packets(item["video"])
                    try:
                        while True:
                            try:
                                packet = next(packets)
                            except StopIteration:
                                break
                            stream.feed(oriented_packet(packet,mirror=mirror))
                        stream.flush()
                    except Exception as e:  # one bad video must not kill the batch
                        error = f"{type(e).__name__}: {e}"
                    if not stream._closed:
                        result = stream.result()
                finally:
                    stream.close()
                wall = perf_counter() - begin
                if result is None:
                    print(f"{item['sample_id']} FAILED before result: {error}", flush=True)
                    continue
                hypothesis = result["gloss_sequence"]
                wer = None if error else _wer(item["reference"], hypothesis)
                oov = [t for t in item["reference"] if t not in vocab_glosses]
                record = dict(schema_version=1, split=args.split, sample_id=item["sample_id"],
                              run_protocol_sha256=protocol_id,
                              translator=item["translator"], direction="mirror" if mirror else "preserve",
                              video_sha=sha256(item["video"]), reference=item["reference"],
                              hypothesis=hypothesis, wer=wer, reference_oov_tokens=sorted(oov),
                              reference_oov_count=len(oov), frames_received=result["frames_received"],
                              windows_processed=result["windows_processed"],
                              quality_rejected_frames=result["quality_rejected_frames"],
                              dropped_frames=result["dropped_frames"], status=result["status"],
                              first_stable_required_context_ms=result["first_stable_required_context_ms"],
                              timings=result["timings"], window_total_ms_values=window_totals,
                              wall_seconds=wall, error=error)
                detail.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                detail.flush()
                edits = wer["substitutions"] + wer["deletions"] + wer["insertions"] if wer else 'NA'
                print(f"{item['sample_id']} {item['translator']} {'M' if mirror else 'P'} "
                      f"wer={wer['wer'] if wer and wer['wer'] is not None else 'NA'} edits={edits} "
                      f"hyp={' '.join(hypothesis) or '(empty)'} wall={wall:.1f}s"
                      + (f" ERROR {error}" if error else ""), flush=True)
        finally:
            shared_frontend.close()
            backend.close()
            summarize(detail_path, out_dir, args, elapsed=perf_counter() - started)


def summarize(detail_path, out_dir, args, elapsed=None):
    records = [json.loads(line) for line in detail_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    protocol_path = out_dir / 'run-protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8')) if protocol_path.is_file() else None
    actual = validate_records(records, protocol, split=args.split)
    expected = {item['sample_id'] for item in protocol['static']['inputs']} if protocol else None
    missing = sorted(expected - actual) if expected is not None else None
    failed = [r for r in records if r.get('error')]
    successful = [r for r in records if not r.get('error')]
    total_ref = total_edits = total_s = total_d = total_i = 0
    total_oov = oov_deletions = empty = 0
    per_translator = {}
    windows, pose, fps, first_final = [], [], [], []
    for r in successful:
        w = r["wer"]
        total_ref += w["reference_tokens"]
        total_edits += w["substitutions"] + w["deletions"] + w["insertions"]
        total_s += w["substitutions"]
        total_d += w["deletions"]
        total_i += w["insertions"]
        total_oov += r["reference_oov_count"]
        oov_deletions += min(r["reference_oov_count"], w["deletions"])
        empty += not r["hypothesis"]
        bucket = per_translator.setdefault(r["translator"], dict(n=0, ref=0, edits=0, mirror=0, mirror_ref=0))
        bucket["n"] += 1
        bucket["ref"] += w["reference_tokens"]
        bucket["edits"] += w["substitutions"] + w["deletions"] + w["insertions"]
        if r["direction"] == "mirror":
            bucket["mirror"] += w["substitutions"] + w["deletions"] + w["insertions"]
            bucket["mirror_ref"] += w["reference_tokens"]
        windows += r.get("window_total_ms_values") or []
        timing = r.get("timings") or {}
        if timing.get("pose_ms", {}).get("mean") is not None:
            pose.append(timing["pose_ms"]["mean"])
        if r["wall_seconds"]:
            fps.append(r["frames_received"] / r["wall_seconds"])
        if r.get("first_stable_required_context_ms") is not None:
            first_final.append(r["first_stable_required_context_ms"])
    corpus_wer = total_edits / total_ref if total_ref else None
    complete = protocol is not None and not missing and not failed
    summary = dict(schema_version=1, split=args.split, videos=len(records),
                   pose_backend=args.pose_backend, device=args.device,
                   mirror_all=args.mirror_all,
                   direction_table=json.loads(Path(args.direction_table).read_text(encoding="utf-8"))
                   if args.direction_table else ("all mirror (diagnostic)" if args.mirror_all else "all preserve (canonical)"),
                   run_protocol_sha256=protocol['protocol_sha256'] if protocol else None,
                   corpus_wer=corpus_wer if complete else None,
                   successful_subset_corpus_wer=corpus_wer,
                   failed_videos=len(failed),successful_videos=len(successful),
                   evaluation_complete=complete,
                   coverage_verified=protocol is not None,
                   expected_videos=len(expected) if expected is not None else None,
                   missing_sample_ids=missing,
                   failure_details=[dict(sample_id=r['sample_id'],error=r['error']) for r in failed],
                   substitutions=total_s, deletions=total_d, insertions=total_i,
                   reference_tokens=total_ref,
                   reference_oov_count=total_oov,
                   reference_oov_rate=total_oov / total_ref if total_ref else None,
                   oov_deletions_share_of_edits_upper_bound=oov_deletions / total_edits if total_edits else None,
                   empty_output_videos=empty,
                   window_total_ms=dict(mean=sum(windows) / len(windows), p50=percentile(windows, .5),
                                        p95=percentile(windows, .95)) if windows else None,
                   pose_ms_median=percentile(pose, .5),
                   processing_fps=dict(median=percentile(fps, .5), p95=percentile(fps, .95),
                                       min=min(fps)) if fps else None,
                   first_final_source_latency_ms=dict(median=percentile(first_final, .5),
                                                      max=max(first_final)) if first_final else None,
                   per_translator_wer={k: dict(n=v["n"], wer=v["edits"] / v["ref"] if v["ref"] else None,
                                               mirror_wer=v["mirror"] / v["mirror_ref"] if v["mirror_ref"] else None)
                                       for k, v in sorted(per_translator.items())},
                   batch_wall_seconds=elapsed,
                   metric_scope="token WER against official CE-CSL gloss labels; not sentence accuracy; "
                                "cross-domain rtmpose-experimental frontend, not official HRNet")
    path = out_dir / f"summary-{args.split}.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("split", "videos", "corpus_wer", "empty_output_videos",
                                              "reference_oov_rate", "processing_fps")}, ensure_ascii=False),
          flush=True)


def main(argv=None):
    configure_utf8_streams()
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--split", choices=("dev", "test", "train"), default="test")
    p.add_argument("--limit", type=int, default=0, help="0 means all; otherwise round-robin per translator")
    p.add_argument("--pose-backend", choices=("rtmpose-experimental", "hrnet"), default="rtmpose-experimental")
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--pose-device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--direction-table", type=Path, help="JSON {translator: preserve|mirror}, calibrated on dev only")
    p.add_argument("--mirror-all", action="store_true", help="force mirror on every video (diagnostic probe)")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--resume", action="store_true", help="skip sample_ids already present in the detail JSONL")
    args = p.parse_args(argv)
    if args.mirror_all and args.direction_table:
        p.error("--mirror-all and --direction-table are mutually exclusive")
    return evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
