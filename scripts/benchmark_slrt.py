"""Run the real SLRT CLI while sampling its process and device resources."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('video','replay','camera'),default='video')
    p.add_argument('--video',type=Path)
    p.add_argument('--camera',type=int,default=0)
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--vote-size',type=int,default=7)
    p.add_argument('--pose-backend',choices=('hrnet','rtmpose-experimental'),default='hrnet')
    p.add_argument('--reference')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-m','slbridge.slrt.cli',a.mode,'--pose-backend',a.pose_backend,'--stride',str(a.stride),
         '--vote-size',str(a.vote_size),'--output',str(out/'result.json'),'--events-jsonl',str(out/'events.jsonl'),
         '--windows-jsonl',str(out/'windows.jsonl'),'--camera',str(a.camera),'--seconds',str(a.seconds)]
    if a.video:cmd+=['--video',str(a.video.resolve())]
    if a.reference is not None:cmd+=['--reference',a.reference]
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    samples=deque(maxlen=4096);gpu=deque(maxlen=4096);gpu_error=None
    sample_count=0;cpu_sum=0.;peak_rss=0.
    started=time.perf_counter();last_gpu=-1.
    with (out/'run.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,cwd=Path(__file__).resolve().parents[1])
        child=psutil.Process(process.pid);child.cpu_percent()
        while process.poll() is None:
            elapsed=time.perf_counter()-started
            try:
                sample=dict(elapsed_seconds=elapsed,rss_mb=child.memory_info().rss/1024**2,
                            cpu_percent_one_core_100=child.cpu_percent(),threads=child.num_threads())
                samples.append(sample);sample_count+=1;cpu_sum+=sample['cpu_percent_one_core_100'];peak_rss=max(peak_rss,sample['rss_mb'])
            except psutil.Error:
                pass
            if elapsed-last_gpu>=1:
                last_gpu=elapsed
                try:
                    op=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],
                                      capture_output=True,text=True,timeout=3,creationflags=flags)
                    if op.returncode==0:
                        gpu.append(dict(elapsed_seconds=elapsed,devices=op.stdout.strip().splitlines()))
                    else:gpu_error=op.stderr.strip()
                except (OSError,subprocess.TimeoutExpired) as error:gpu_error=str(error)
            time.sleep(.2)
        code=process.wait()
    result=dict(command=cmd,exit_code=code,wall_seconds=time.perf_counter()-started,
                process_samples=list(samples),device_samples=list(gpu),device_sampling_error=gpu_error,
                sample_history_scope='last_4096_samples',total_process_samples=sample_count,
                sampled_process_peak_rss_mb=peak_rss if sample_count else None,
                sampled_process_cpu_percent_mean=cpu_sum/sample_count if sample_count else None,
                gpu_measurement_scope='whole_device_including_other_processes_not_exclusive_SLRT',
                recognition_acceptance='inspect result.json tokens and real reference separately; exit0 is not semantic accuracy')
    (out/'resource-samples.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('exit_code','wall_seconds','sampled_process_peak_rss_mb','sampled_process_cpu_percent_mean')},indent=2))
    return code


if __name__=='__main__':raise SystemExit(main())
