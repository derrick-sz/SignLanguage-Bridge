"""Controlled cold-residency probe for only the verified idle model process."""
import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import time


def main():
    import httpx
    import psutil
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('Windows-only diagnostic')
    if args.output.exists():raise ValueError('Use a fresh diagnostic output')
    profile=json.loads(args.profile.read_text(encoding='utf-8-sig'));service=profile['recognition']
    state=json.loads((Path(profile['output_root'])/'runtime-state.json').read_text(encoding='utf-8-sig'))
    process=psutil.Process(state['recognition_pid']);command=process.cmdline()
    assert Path(process.exe()).resolve()==Path(service['python']).resolve()
    assert 'slbridge.api_cli' in command and Path(process.cwd()).resolve()==Path(service['workdir']).resolve()
    for option,value in [('--port',str(service['port'])),('--realtime-config',service['realtime_config'])]:
        assert any(command[i:i+2]==[option,value] for i in range(len(command)-1)),option
    with httpx.Client(trust_env=False,timeout=5) as client:
        live=client.get(f"http://127.0.0.1:{service['port']}/v3/realtime").json()
        models=client.get(f"http://127.0.0.1:{service['port']}/v3/models").json()
    assert live['ready'] and not live['busy']
    assert all(s.get('state')!='inferencing' for s in models['status'].values())
    before=process.memory_info()._asdict();created=process.create_time()
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.K32EmptyWorkingSet.argtypes=[wintypes.HANDLE];kernel.K32EmptyWorkingSet.restype=wintypes.BOOL
    kernel.CloseHandle.argtypes=[wintypes.HANDLE];kernel.CloseHandle.restype=wintypes.BOOL
    handle=kernel.OpenProcess(0x0400|0x0100,False,process.pid)
    if not handle:raise ctypes.WinError(ctypes.get_last_error())
    try:
        assert psutil.Process(process.pid).create_time()==created
        if not kernel.K32EmptyWorkingSet(handle):raise ctypes.WinError(ctypes.get_last_error())
    finally:kernel.CloseHandle(handle)
    after=process.memory_info()._asdict()
    record=dict(pid=process.pid,process_created=created,operation='K32EmptyWorkingSet_only_verified_idle_recognition',
                before=before,after=after,live=live,checked_at=time.time(),other_processes_modified=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2),encoding='utf-8')
    print({'pid':process.pid,'rss_before_mb':before['rss']/2**20,'rss_after_mb':after['rss']/2**20})


if __name__=='__main__':main()
