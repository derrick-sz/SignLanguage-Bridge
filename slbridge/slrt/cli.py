"""Independent SLRT commands; --help does not import Torch or Qt."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from time import perf_counter

from .._io import configure_utf8_streams
from .assets import SLRTConfig


def build_parser():
    p = argparse.ArgumentParser(prog='slbridge-slrt',description='SLRT-CSL Gloss recognition; translation is a separate future module')
    p.add_argument('command',choices=('inspect','bundle','video','replay','camera','desktop'))
    p.add_argument('--config',type=Path)
    p.add_argument('--resource-dir',type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--vocabulary',type=Path)
    p.add_argument('--device',choices=('auto','cpu','cuda'))
    p.add_argument('--pose-device',choices=('auto','cpu','cuda'))
    p.add_argument('--pose-backend',choices=('hrnet','hrnet-npz','rtmpose-experimental'))
    p.add_argument('--window',type=int)
    p.add_argument('--stride',type=int)
    p.add_argument('--vote-size',type=int)
    p.add_argument('--video',type=Path)
    p.add_argument('--pose-npz',type=Path)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--seconds',type=float,default=0,help='Camera duration, 0 means until Ctrl+C')
    p.add_argument('--output',type=Path)
    p.add_argument('--events-jsonl',type=Path)
    p.add_argument('--windows-jsonl',type=Path)
    p.add_argument('--reference',help='Optional real ground truth, space-separated Gloss tokens, for single-video WER')
    p.add_argument('--require-gloss',action='store_true',help='Exit 3 on zero final Gloss; useful for positive acceptance samples')
    return p


def config_from_args(args):
    c = SLRTConfig.from_file(args.config) if args.config else SLRTConfig()
    mapping = dict(resource_dir='resource_dir',checkpoint='checkpoint',vocabulary='vocabulary',device='device',
                   pose_device='pose_device',pose_backend='pose_backend',window='window_size',stride='stride',vote_size='vote_size')
    changes = {dst:str(getattr(args,src)) if isinstance(getattr(args,src),Path) else getattr(args,src)
               for src,dst in mapping.items() if getattr(args,src) is not None}
    return replace(c,**changes)


def _write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate_output_paths(args,config):
    outputs=[Path(p).resolve() for p in (args.output,args.events_jsonl,args.windows_jsonl) if p is not None]
    inputs=[p for p in (args.video,args.pose_npz,args.config,config.checkpoint_path,config.vocabulary_path,
                        config.detector_config,config.detector_checkpoint,config.hrnet_config,config.hrnet_checkpoint) if p is not None]
    if len(outputs)!=len(set(outputs)):
        raise ValueError('JSON and JSONL output paths must be distinct')
    if set(outputs)&{Path(p).resolve() for p in inputs}:
        raise ValueError('Output paths must not overwrite source videos, poses, configs, or model assets')


def main(argv=None):
    configure_utf8_streams()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command in {'video','replay'} and args.video is None:
        parser.error('video/replay requires --video')
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error('--seconds must be finite and nonnegative')
    if args.command == 'camera' and args.video is not None:
        parser.error('Use replay to feed video in real time')
    backend = None
    session = None
    handles = []
    result = None
    safe_outputs = False
    start = perf_counter()
    first_final = None
    def sink(path, *, events=False):
        if path is None:
            def record(value):
                nonlocal first_final
                if events and value.get('status') == 'final' and first_final is None:
                    first_final = (perf_counter()-start)*1000
            return record
        path.parent.mkdir(parents=True,exist_ok=True)
        f = path.open('w',encoding='utf-8'); handles.append(f)
        def record(value):
            nonlocal first_final
            if events and value.get('status') == 'final' and first_final is None:
                first_final = (perf_counter()-start)*1000
            f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n'); f.flush()
        return record
    try:
        config = config_from_args(args)
        validate_output_paths(args,config)
        safe_outputs = True
        if args.command == 'bundle':
            if args.output is None:
                parser.error('bundle requires --output path/to/manifest.json')
            from .assets import require_file,load_vocabulary,BACKEND_ID
            require_file(config.checkpoint_path,'SLRT checkpoint',config.checkpoint_sha256)
            require_file(config.vocabulary_path,'SLRT vocabulary',config.vocabulary_sha256)
            load_vocabulary(config.vocabulary_path)
            raw=config.to_dict()
            for name in ('resource_dir','checkpoint','vocabulary','detector_config','detector_checkpoint','hrnet_config','hrnet_checkpoint','rtmpose_cache_dir'):
                if raw.get(name):raw[name]=str(Path(raw[name]).resolve())
            raw['backend']=BACKEND_ID
            _write(args.output,raw)
            print(json.dumps(dict(status='bundle_manifest_written',path=str(args.output.resolve()),frontend_readiness_verified=False)))
            return 0
        if args.command == 'desktop':
            from .desktop import launch
            return launch(config,video=args.video,camera=args.camera,pose_npz=args.pose_npz)
        from .backend import SLRTBackend
        backend = SLRTBackend(config)
        event_sink, window_sink = sink(args.events_jsonl,events=True),sink(args.windows_jsonl)
        import psutil
        proc = psutil.Process()
        cpu_start = proc.cpu_times()
        run_start = perf_counter()
        if args.command == 'inspect':
            result = dict(model=backend.runtime.info,config=config.to_dict(),status='strict_weights_loaded',
                          video_inference_verified=False,official_hrnet_verified=False)
        elif args.command == 'video':
            result = backend.recognize_video(args.video,pose_npz=args.pose_npz,event_sink=event_sink,window_sink=window_sink)
            result['first_stable_wall_latency_ms'] = first_final
            result['wall_latency_scope'] = 'command_start_including_model_and_frontend_initialization'
            if args.reference is not None:
                result['reference'] = args.reference.split()
                result['wer'] = _wer(args.reference.split(),result['gloss_sequence'])
        else:
            from .capture import SLRTCaptureSession
            session = SLRTCaptureSession(backend,camera=args.camera,
                                         video=args.video if args.command == 'replay' else None,pose_npz=args.pose_npz,
                                         event_sink=event_sink,window_sink=window_sink)
            session.start()
            try:
                from time import sleep
                while session.is_running and (not args.seconds or perf_counter()-run_start < args.seconds):
                    sleep(.05)
            except KeyboardInterrupt:
                pass
            finally:
                session.stop()
            result = session.result()
            if session.error:
                raise RuntimeError(session.error)
        cpu_end = proc.cpu_times()
        result['resources'] = dict(process_rss_mb=proc.memory_info().rss/1024**2,
                                   process_cpu_seconds=(cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system),
                                   measurement_wall_seconds=perf_counter()-run_start,
                                   rss_scope='end_snapshot_not_peak')
        result['elapsed_command_seconds'] = perf_counter()-start
        if args.output and safe_outputs:
            _write(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
        return 3 if args.require_gloss and not result.get('gloss_sequence') else 0
    except (ValueError,RuntimeError,ImportError,OSError) as error:
        result = dict(status='error',error=f'{type(error).__name__}: {error}')
        if args.output and safe_outputs:
            _write(args.output,result)
        print(json.dumps(result,ensure_ascii=False),file=sys.stderr)
        return 2
    finally:
        for handle in handles:
            handle.close()
        if backend and (session is None or not session.is_running):
            backend.close()


def _wer(reference,hypothesis):
    """Token Levenshtein with S/D/I; no accuracy label or text normalization."""
    table = [[(j,0,j,0) for j in range(len(hypothesis)+1)]]
    for i,r in enumerate(reference,1):
        row = [(i,0,0,i)]
        for j,h in enumerate(hypothesis,1):
            if r == h:
                row.append(table[i-1][j-1])
            else:
                diag = table[i-1][j-1]; ins = row[j-1]; dele = table[i-1][j]
                row.append(min((diag[0]+1,diag[1]+1,diag[2],diag[3]),
                               (ins[0]+1,ins[1],ins[2]+1,ins[3]),
                               (dele[0]+1,dele[1],dele[2],dele[3]+1)))
        table.append(row)
    edits,s,i,d = table[-1][-1]
    return dict(wer=edits/len(reference) if reference else None,substitutions=s,insertions=i,deletions=d,
                reference_tokens=len(reference),scope='single_video_user_supplied_reference')


if __name__ == '__main__':
    raise SystemExit(main())
