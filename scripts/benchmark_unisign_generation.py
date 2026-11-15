"""Count encoder calls and compare inference paths on frozen, label-free poses.

The baseline is the existing runtime. A candidate may only change extraction of
visual inputs; the pinned model's generate, beam and tokenizer stay unchanged.
This is a performance/behavior regression, not a recognition accuracy study.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import time
import weakref

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.p1.gt_guard import assert_gt_clean
from slbridge.uni_sign import TorchUniSignRuntime, UniSignConfig
from slbridge.uni_sign_preprocess import prepare_pose_window
from evaluate_unisign_csldaily_pose import load_pose


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def baseline_infer(path):
    """Load only our own saved pre-change infer function for paired execution."""
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'TorchUniSignRuntime')
    function = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'infer')
    namespace = dict(vars(sys.modules[TorchUniSignRuntime.__module__]))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['infer']


def windows_memory_budget():
    """Physical free RAM alone does not bound Windows allocation failures."""
    if sys.platform != 'win32':
        return None
    import ctypes
    from ctypes import wintypes

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD)] + [(name, ctypes.c_size_t) for name in
            ('CommitTotal', 'CommitLimit', 'CommitPeak', 'PhysicalTotal', 'PhysicalAvailable',
             'SystemCache', 'KernelTotal', 'KernelPaged', 'KernelNonpaged', 'PageSize')] + [
            (name, wintypes.DWORD) for name in ('HandleCount', 'ProcessCount', 'ThreadCount')]

    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    api = ctypes.WinDLL('psapi', use_last_error=True).GetPerformanceInfo
    api.argtypes = (ctypes.POINTER(PerformanceInfo), wintypes.DWORD)
    api.restype = wintypes.BOOL
    if not api(ctypes.byref(info), info.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(commit_free_bytes=(info.CommitLimit-info.CommitTotal)*info.PageSize,
                commit_limit_bytes=info.CommitLimit*info.PageSize,
                physical_free_bytes=info.PhysicalAvailable*info.PageSize)


def main(args):
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'artifacts') or output.exists():
        raise ValueError('Use a fresh artifacts output directory')
    output.mkdir(parents=True)
    budget = windows_memory_budget()
    (output / 'memory-before.json').write_text(json.dumps(budget, indent=2), encoding='utf-8')
    if budget is not None and budget['commit_free_bytes'] < 10 * 1024**3:
        raise RuntimeError('Need at least 10 GiB free Windows commit before loading a second model; use an isolated evaluation window')
    source = json.loads((args.source_run / 'manifest.json').read_text(encoding='utf-8-sig'))
    selected = source['samples'][:args.limit]
    rows = []
    for sample in selected:
        path = ROOT / 'artifacts/unisign-csldaily-train-657-20261025/pose' / (sample['sample_id'] + '.pkl')
        assert sha(path) == sample['pose_sha256']
        points, scores, _ = load_pose(path)
        pose = prepare_pose_window(points, scores, max_frames=256)
        identity = dict(sample_id=sample['sample_id'], pose_sha256=sample['pose_sha256'], frames=len(pose.sampled_indices))
        assert_gt_clean(identity, context='generation performance input')
        rows.append((identity, pose))
    if args.edge_cases:
        import numpy as np
        identity, source_pose = rows[0]
        for frames in (4, 256):
            indices = np.arange(frames) % len(source_pose.sampled_indices)
            diagnostic = replace(source_pose, **{name: array[indices].copy() for name, array in source_pose.parts.items()},
                                 sampled_indices=tuple(range(frames)), quality={'diagnostic': True})
            rows.append((dict(identity, sample_id=identity['sample_id']+f'--synthetic-{frames}', frames=frames), diagnostic))
    config = UniSignConfig.from_bundle(ROOT / 'models/external/uni_sign/model_bundle')
    assert sha(config.checkpoint) == config.checkpoint_sha256
    runtime = TorchUniSignRuntime(config, device='cuda', dtype='bfloat16')
    runtime.load()
    model_ref = weakref.ref(runtime._model)
    import torch
    torch.set_num_threads(4)
    counter = {'encoder': 0}
    generated = []
    handle = runtime._model.mt5_model.encoder.register_forward_hook(
        lambda *_: counter.__setitem__('encoder', counter['encoder'] + 1))
    original_generate = runtime._model.generate

    def generate(*a, **kw):
        result = original_generate(*a, **kw)
        generated.append(result.detach().cpu().tolist())
        return result

    runtime._model.generate = generate
    modes = {'baseline': baseline_infer(args.baseline_source), 'candidate': TorchUniSignRuntime.infer}
    capture_enabled = False
    captured = []

    def inspect_visual(_module, _args, kwargs):
        if capture_enabled and kwargs.get('labels') is not None:
            captured.append({key: dict(shape=list(kwargs[key].shape), dtype=str(kwargs[key].dtype),
                sha256=hashlib.sha256(kwargs[key].detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest())
                for key in ('inputs_embeds', 'attention_mask')})

    visual_handle = runtime._model.mt5_model.register_forward_pre_hook(inspect_visual, with_kwargs=True)
    manifest = dict(scope='frozen_official_train_pose_behavior_and_performance_only',
                    checkpoint_sha256=config.checkpoint_sha256, model_source_sha256=sha(config.source_dir / 'models.py'),
                    runtime_source_sha256=sha(ROOT / 'slbridge/uni_sign.py'), script_sha256=sha(__file__),
                    device='cuda', dtype='bfloat16', num_beams=4, max_new_tokens=100,
                    samples=[identity for identity, _ in rows], modes=list(modes),
                    baseline_source_sha256=sha(args.baseline_source), memory_before=budget)
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    results = []
    try:
        visual_checks = []
        for identity, pose in rows:
            hashes = {}
            for mode, infer in modes.items():
                capture_enabled = True
                captured.clear()
                infer(runtime, pose, num_beams=4, max_new_tokens=100)
                capture_enabled = False
                assert len(captured) == 1
                hashes[mode] = captured[0]
            assert all(value == next(iter(hashes.values())) for value in hashes.values())
            visual_checks.append(dict(identity, inputs=hashes, exact=True))
        (output / 'visual-inputs.json').write_text(json.dumps(visual_checks, indent=2), encoding='utf-8')
        for repeat in range(args.repeats):
            for index, (identity, pose) in enumerate(rows):
                order = list(modes.items())
                if (index + repeat) % 2:
                    order.reverse()
                pair = []
                for mode, infer in order:
                    before = counter['encoder']
                    started = time.perf_counter()
                    text, timings = infer(runtime, pose, num_beams=4, max_new_tokens=100)
                    row = dict(identity, mode=mode, repeat=repeat, text=text, tokens=generated[-1],
                               encoder_calls=counter['encoder']-before, wall_ms=(time.perf_counter()-started)*1000, **timings)
                    assert row['encoder_calls'] == (2 if mode == 'baseline' else 1)
                    assert_gt_clean(row, context='generation performance output')
                    pair.append(row)
                    results.append(row)
                    with (output / 'outputs.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                assert all(row['tokens'] == pair[0]['tokens'] and row['text'] == pair[0]['text'] for row in pair)
                print(json.dumps(dict(sample_id=identity['sample_id'], repeat=repeat,
                    values={r['mode']: {k: r[k] for k in ('encoder_calls', 'model_forward_ms', 'generation_ms', 'model_total_ms')} for r in pair},
                    exact_tokens=True)), flush=True)
    finally:
        # Removing the instance override restores the original class method.
        # Reassigning a saved bound method would keep the model alive here.
        del runtime._model.generate
        del original_generate
        visual_handle.remove()
        handle.remove()
        runtime.close()
    cleanup = dict(model_released=model_ref() is None,
                   cuda_allocated_after_close_bytes=torch.cuda.memory_allocated(),
                   cuda_reserved_after_close_bytes=torch.cuda.memory_reserved())
    (output / 'cleanup.json').write_text(json.dumps(cleanup, indent=2), encoding='utf-8')
    assert cleanup['model_released'], 'Probe retained the model after close'
    (output / 'memory-after.json').write_text(json.dumps(windows_memory_budget(), indent=2), encoding='utf-8')
    import numpy as np
    summary = dict(requests=len(results), paired_tokens_exact=True, visual_inputs_exact=True, by_mode={})
    for mode in modes:
        group = [r for r in results if r['mode'] == mode]
        summary['by_mode'][mode] = dict(n=len(group), encoder_calls=sorted({r['encoder_calls'] for r in group}),
            timings={name: dict(mean=float(np.mean([r[name] for r in group])), p50=float(np.percentile([r[name] for r in group],50)),
                p95=float(np.percentile([r[name] for r in group],95))) for name in ('model_forward_ms','generation_ms','model_total_ms','wall_ms')})
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-run', type=Path, default=ROOT / 'artifacts/train-application-validation-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--edge-cases', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.limit <= 30 or not 1 <= args.repeats <= 5:
        parser.error('Invalid bounds')
    main(args)
