"""Prepare fixed, explicitly labelled train replay media, outside the package."""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    result=subprocess.run([str(a) for a in args],capture_output=True,check=True,
                          creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    return result.stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ffmpeg',type=Path,required=True);a=p.parse_args()
    output=a.output.resolve()
    if not output.is_relative_to(ROOT/'artifacts') or output==ROOT/'artifacts': raise ValueError('Use an artifact subdirectory')
    if (output/'catalog.json').exists() or (output/'media').exists(): raise ValueError('Refusing to replace demo media')
    output.mkdir(parents=True,exist_ok=True);(output/'media').mkdir()
    source=ROOT/'artifacts/train-application-validation-v2'
    manifest=read(source/'manifest.json');samples={r['sample_id']:r for r in manifest['samples']}
    reference={r['sample_id']:r for r in read(source/'eval/records.json')}
    choices=[('intro','S000011_P0000_T00','日常介绍','training_replay',0),
             ('school-reading','S000062_P0000_T00','校园交流','training_replay',0),
             ('sleep-classroom','S001663_P0005_T00','作息与课堂','training_replay',0),
             ('known-tail','S000068_P0000_T00','已知问题：句末误追加','diagnostic_replay',2)]
    items=[];evidence=[]
    for identifier,name,title,kind,tail in choices:
        sample=samples[name];original=Path(sample['video']);ref=reference[name]
        assert sample['source_split']=='train' and sha(original)==sample['video_sha256']
        if kind=='training_replay': assert ref['clip_exact'] and ref['realtime_exact']
        destination=output/'media'/f'{identifier}.mp4'
        command=[a.ffmpeg,'-hide_banner','-loglevel','error','-n','-i',original]
        if tail: command+=['-vf',f'tpad=stop_mode=clone:stop_duration={tail}']
        command+=['-an','-c:v','libx264','-threads','1','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',destination]
        run(command)
        probe=a.ffmpeg.with_name('ffprobe.exe' if os.name=='nt' else 'ffprobe')
        meta=json.loads(run([probe,'-v','error','-select_streams','v:0','-show_entries',
                             'stream=width,height,avg_frame_rate,nb_frames','-of','json',destination]))['streams'][0]
        fps=float(Fraction(meta['avg_frame_rate']));frames=int(meta['nb_frames'])
        assert abs(fps-sample['fps'])<.001 and frames==sample['frames']+round(tail*fps)
        note='已测训练回放；跟练与真人语义效果待验证。' if not tail else '含2秒静止补尾的已知问题回放，用于展示核对。'
        items.append(dict(id=identifier,title=title,annotation=ref['ref_sentence_upstream'],kind=kind,
                          source_split='train',sample_id=name,media=f'media/{identifier}.mp4',sha256=sha(destination),
                          fps=fps,frames=frames,width=int(meta['width']),height=int(meta['height']),note=note))
        evidence.append(dict(id=identifier,sample_id=name,source_video=str(original),source_sha256=sha(original),
                             source_frames=sample['frames'],tail_seconds=tail,media_sha256=sha(destination),
                             encoding='H264_crf18_yuv420p_original_fps_faststart_no_audio_no_subtitles'))
    (output/'catalog.json').write_text(json.dumps(dict(schema_version=1,items=items),ensure_ascii=False,indent=2),encoding='utf-8')
    record=dict(source_manifest_sha256=sha(source/'manifest.json'),script_sha256=sha(__file__),items=evidence,
                role='local_training_and_diagnostic_replay_not_product_accuracy',dataset_url='https://ustc-slr.github.io/datasets/2021_csl_daily/')
    (output/'build-record.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps([dict(id=i['id'],frames=i['frames'],fps=i['fps'],bytes=(output/i['media']).stat().st_size) for i in items]))


if __name__=='__main__':main()
