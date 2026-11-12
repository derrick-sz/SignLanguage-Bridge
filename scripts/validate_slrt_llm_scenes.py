"""Bounded, GT-separated SLRT/LLM experiment. Production services stay intact."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.p1.gt_guard import assert_gt_clean

MODEL = 'deepseek-v4.1-flash'
SYSTEM = '''你是手语识别词序列的中文语言整理器。输入只有模型实际识别到的词，不是标准答案。
使用范围是饮食休息、时间出行、人物日常交流。允许调整语序和补必要助词。
禁止添加输入不支持的人物、否定、数字、时间、动作、对象；不能凭场景猜词或纠正视觉识别。
证据不足、孤立名词不能组成明确表达时返回insufficient，text为空。
只输出一个JSON对象：{"status":"draft或insufficient","text":"中文句子或空串","used_token_indices":[使用的零基词下标]}。
保持简短；不输出解释、推理或Markdown。'''
SCENES = {
    'daily_need': ['喝', '水', '吃', '睡觉', '休息', '饿', '累', '帮助', '谢谢'],
    'time_travel': ['明天', '昨天', '早上', '下午', '学校', '医院', '出差'],
    'person_daily': ['名字', '几岁', '年龄', '朋友', '老师', '爸爸', '妈妈', '同学'],
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value, *, clean=True):
    if clean:
        assert_gt_clean(value, context=str(path))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def log(**value):
    print(json.dumps(value, ensure_ascii=True), flush=True)


def prepare(run):
    import cv2
    import numpy as np
    if (run / 'manifest.json').exists():
        raise ValueError('Refusing to replace an existing protocol/sample selection')
    metadata = {r['sample_id']: r for r in rows(ROOT / 'experiments/p1/datasets/manifest_1086.csv')}
    references = rows(ROOT / 'experiments/p1/references/gt_1086.csv')
    old = {r['sample_id'] for r in rows(ROOT / 'experiments/p1/datasets/pilot_50.csv')}
    selected, used = [], set()
    for scene, keywords in SCENES.items():
        candidates = [r for r in references if r['sample_id'] not in old and r['sample_id'] not in used
            and r['sample_id'] in metadata and len(r['gt_gloss'].split()) <= 8
            and int(metadata[r['sample_id']]['frames']) <= 180
            and any(k in r['gt_sentence_official'] for k in keywords)
            and Path(metadata[r['sample_id']]['trimmed_video_path']).is_file()]
        buckets = {}
        for r in sorted(candidates, key=lambda r: r['sample_id']):
            buckets.setdefault(r['signer_id'], []).append(r)
        chosen = []
        while len(chosen) < 10:
            advanced = False
            for signer in sorted(buckets):
                if buckets[signer] and len(chosen) < 10:
                    chosen.append(buckets[signer].pop(0)); advanced = True
            if not advanced:
                raise ValueError(f'Not enough samples for {scene}')
        for index, r in enumerate(chosen):
            used.add(r['sample_id'])
            selected.append((r, scene, index < 2))
    manifest_samples, eval_rows = [], []
    directory = run / 'inputs'
    directory.mkdir(parents=True)
    first_frame = None
    for r, scene, replay in selected:
        source = Path(metadata[r['sample_id']]['trimmed_video_path'])
        assert digest(source) == metadata[r['sample_id']]['trimmed_video_sha256']
        target = directory / (r['sample_id'] + '.mp4')
        cap = cv2.VideoCapture(str(source)); fps = float(cap.get(cv2.CAP_PROP_FPS))
        declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), fps, (512, 512))
        assert writer.isOpened()
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok: break
                frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_LINEAR)
                if first_frame is None: first_frame = frame.copy()
                writer.write(frame); count += 1
        finally:
            cap.release(); writer.release()
        assert count == declared == int(metadata[r['sample_id']]['frames'])
        manifest_samples.append(dict(sample_id=r['sample_id'], scene=scene, signer_id=r['signer_id'],
            source_split='test', source_video=str(source), source_sha256=digest(source), video=str(target),
            video_sha256=digest(target), fps=fps, frames=count, replay=replay, kind='positive'))
        eval_rows.append(r)
    for name, image in [('black', np.zeros_like(first_frame)), ('static_signer', first_frame)]:
        target = directory / (name + '.mp4')
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*'mp4v'), 30, (512,512))
        assert writer.isOpened()
        for _ in range(90): writer.write(image)
        writer.release()
        manifest_samples.append(dict(sample_id=name, scene='control', signer_id='synthetic', video=str(target),
            video_sha256=digest(target), fps=30, frames=90, replay=True, kind=name))
    from slbridge.slrt.assets import PROTOCOL, CHECKPOINT_SHA256, VOCAB_SHA256
    manifest = dict(protocol='slrt-llm-scenes-v1', created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
        data_scope='retrospective_seen_1086_scene_subset_not_independent_signers', scenes=SCENES,
        selection=dict(per_scene=10, max_gloss_words=8, max_frames=180, exclude_old_pilot_50=True,
            order='sample_id_then_signer_round_robin', replay_per_scene=2),
        input_protocol='derived_512_square_mp4v_original_fps', samples=manifest_samples,
        slrt=dict(protocol=PROTOCOL, device='cuda', pose_backend='rtmpose-experimental', pose_device='cuda',
            window_size=16, stride=1, vote_size=7, queue_size=32, frame_cache=True,
            checkpoint_sha256=CHECKPOINT_SHA256, vocabulary_sha256=VOCAB_SHA256),
        llm=dict(provider='wb-proxy', base_url='http://127.0.0.1:8788/v1', model=MODEL,
            temperature=0, max_tokens=4096, timeout_seconds=35, retries=0,
            credentials_ref='WB_PROXY_API_KEY@dsh-credentials', system_sha256=hashlib.sha256(SYSTEM.encode()).hexdigest()),
        source_hashes={str(p.relative_to(ROOT)): digest(p) for p in [Path(__file__),
            ROOT/'slbridge/slrt/runtime.py',ROOT/'slbridge/slrt/stream.py',ROOT/'slbridge/slrt/capture.py',
            ROOT/'slbridge/slrt/pose.py',ROOT/'slbridge/slrt/preprocess.py',ROOT/'slbridge/slrt/frame_cache.py'] if p.exists()})
    write(run/'manifest.json', manifest)
    write(run/'eval/selected_references.json', eval_rows, clean=False)
    write(run/'prompt.json', {'system':SYSTEM})
    log(stage='prepare', positives=30, replay_positives=6, controls=2, run_dir=str(run))


def backend_llm():
    from experiments.p1.e02.llm_fusion.backends.openai_compat import OpenAICompatBackend
    return OpenAICompatBackend(model=MODEL, enabled=True, max_retries=0, timeout_seconds=35)


def organize_tokens(backend, tokens):
    payload = {'tokens': tokens}
    assert_gt_clean(payload)
    if not tokens:
        return dict(status='insufficient', text='', called=False, latency_ms=0., used_token_indices=[], reason='no_tokens')
    result = backend.generate([{'role':'system','content':SYSTEM},
        {'role':'user','content':json.dumps(payload,ensure_ascii=False)}],group='scene_organizer',max_tokens=4096)
    data = result.to_dict()
    record = dict(called=True, latency_ms=data['latency_ms'], tokens_in=data['tokens_in'], tokens_out=data['tokens_out'],
        model_version=data['model_version'], finish_reason=data['extra'].get('finish_reason'), raw_text=data['text'],
        status='error', text='', error=data['error'])
    if data['status'] == 'ok':
        try:
            parsed = json.loads(data['text'])
            status, text, indices = parsed['status'], parsed['text'], parsed['used_token_indices']
            if (status not in {'draft','insufficient'} or not isinstance(text,str)
                    or not isinstance(indices,list) or any(type(i) is not int or i<0 or i>=len(tokens) for i in indices)
                    or len(set(indices)) != len(indices) or (status=='draft' and (not text.strip() or not indices))
                    or (status=='insufficient' and text.strip())):
                raise ValueError('Invalid organizer schema')
            record.update(status=status,text=text,used_token_indices=indices,error=None)
        except (ValueError,KeyError,TypeError):
            record['error']='invalid_json_or_schema'
    return record


def smoke(run):
    backend=backend_llm()
    records=[]
    for name,tokens in [('sleep',['我','想','睡觉']),('isolated_color',['绿色']),('negation',['我','不','喝','水'])]:
        result=organize_tokens(backend,tokens)
        records.append(dict(sample_id=name,tokens=tokens,result=result,kind='synthetic_contract'))
        log(stage='smoke',sample_id=name,result=result)
    write(run/'smoke.json',records)


def slrt_backend():
    import numpy as np
    from slbridge.slrt.assets import SLRTConfig
    from slbridge.slrt.backend import SLRTBackend
    backend=SLRTBackend(SLRTConfig(device='cuda',pose_device='cuda',pose_backend='rtmpose-experimental'))
    backend.runtime.predict(np.zeros((16,320,320,3),dtype=np.uint8), np.zeros((16,133,3),dtype=np.float32))
    return backend


def recognize(run):
    manifest=read(run/'manifest.json'); backend=slrt_backend()
    write(run/'runtime.json',backend.runtime.info)
    try:
        for item in manifest['samples']:
            path=run/'offline'/f"{item['sample_id']}.json"
            if path.exists(): raise ValueError('Refusing to mix/reuse recognition without a new protocol run')
            assert digest(item['video'])==item['video_sha256']
            events=[]; started=time.perf_counter()
            def sink(e): events.append(dict(e,wall_elapsed_ms=(time.perf_counter()-started)*1000))
            try:
                result=backend.recognize_video(item['video'],event_sink=sink)
                record=dict(sample_id=item['sample_id'],status='ok',result=result,events=events)
            except Exception as error:
                record=dict(sample_id=item['sample_id'],status='error',error_type=type(error).__name__,error=str(error)[:500])
            write(path,record)
            log(stage='recognize',sample_id=item['sample_id'],status=record['status'],
                words=record.get('result',{}).get('gloss_sequence'),wall_ms=(time.perf_counter()-started)*1000)
    finally: backend.close()


def organize(run):
    manifest=read(run/'manifest.json'); backend=backend_llm()
    for item in manifest['samples']:
        source=read(run/'offline'/f"{item['sample_id']}.json")
        tokens=source.get('result',{}).get('gloss_sequence',[])
        result=organize_tokens(backend,tokens)
        write(run/'organized'/f"{item['sample_id']}.json",dict(sample_id=item['sample_id'],tokens=tokens,result=result))
        log(stage='organize',sample_id=item['sample_id'],status=result['status'],latency_ms=result['latency_ms'],text=result['text'])


class IncrementalOrganizer:
    """One in flight + latest pending; source revisions never wait for the LLM."""
    def __init__(self,backend,start):
        self.backend,self.start=backend,start
        self.condition=threading.Condition(); self.pending=None; self.latest=[]; self.revision=0
        self.closed=False; self.calls=[]; self.published=[]; self.scheduled=set()
        self.thread=threading.Thread(target=self._run,daemon=True); self.thread.start()

    def offer(self,tokens,*,eof=False):
        with self.condition:
            if list(tokens)!=self.latest:
                self.latest=list(tokens); self.revision+=1
            if (eof or (len(tokens)>=2 and len(tokens)%2==0)) and self.revision not in self.scheduled:
                self.scheduled.add(self.revision)
                self.pending=(self.revision,list(tokens),(time.perf_counter()-self.start)*1000)
                self.condition.notify_all()
            if eof: self.closed=True; self.condition.notify_all()

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda:self.pending is not None or self.closed)
                if self.pending is None: return
                revision,tokens,queued_ms=self.pending; self.pending=None
            call_started=time.perf_counter()
            try:
                result=organize_tokens(self.backend,tokens)
            except Exception as error:
                # Preserve an observable failed attempt if a provider raises
                # before returning its structured result; never copy secrets.
                result=dict(status='error',text='',called=None,
                    latency_ms=(time.perf_counter()-call_started)*1000,
                    error='backend_exception',error_type=type(error).__name__)
            record=dict(revision=revision,tokens=tokens,queued_ms=queued_ms,
                completed_ms=(time.perf_counter()-self.start)*1000,result=result)
            with self.condition:
                record['stale']=revision!=self.revision
                self.calls.append(record)
                if not record['stale']: self.published.append(record)

    def finish(self,tokens,*,natural_eof=True):
        if natural_eof:
            self.offer(tokens,eof=True)
        else:
            with self.condition:
                self.revision+=1; self.pending=None; self.closed=True
                self.condition.notify_all()
        self.thread.join(timeout=110)
        if self.thread.is_alive(): raise TimeoutError('Organizer did not drain within bounded deadline')


def replay(run):
    from slbridge.slrt.capture import SLRTCaptureSession
    manifest=read(run/'manifest.json'); backend=slrt_backend(); active_session=None
    try:
        for item in manifest['samples']:
            if not item['replay']: continue
            assert digest(item['video'])==item['video_sha256']
            events=[]; finalized={}; started=time.perf_counter()
            organizer=IncrementalOrganizer(backend_llm(),started)
            def sink(e):
                e=dict(e,session_wall_elapsed_ms=e.get('wall_elapsed_ms'),
                    wall_elapsed_ms=(time.perf_counter()-started)*1000)
                events.append(e)
                if e['status']=='final':
                    finalized[e['event_id']]=e['gloss']
                    organizer.offer(list(finalized.values()))
            class TimedSession(SLRTCaptureSession):
                capture_start_ms=None; capture_end_ms=None
                def _capture_loop(self):
                    self.capture_start_ms=(time.perf_counter()-started)*1000
                    try: return super()._capture_loop()
                    finally: self.capture_end_ms=(time.perf_counter()-started)*1000
            session=TimedSession(backend,video=item['video'],event_sink=sink)
            active_session=session
            session.start()
            while session.is_running and time.perf_counter()-started<90:
                session.poll(); time.sleep(.05)
            timed_out=session.is_running
            if timed_out:
                try:
                    session.stop(timeout=10)
                except Exception as error:
                    organizer.finish([],natural_eof=False)
                    write(run/'replay'/f"{item['sample_id']}.json",dict(sample_id=item['sample_id'],
                        timed_out=True,natural_eof=False,error_type=type(error).__name__,
                        native_release_pending=session.is_running,result=session.result(),events=events,
                        calls=organizer.calls,published=organizer.published,
                        capture_start_ms=session.capture_start_ms,capture_end_ms=session.capture_end_ms,
                        total_wall_ms=(time.perf_counter()-started)*1000))
                    raise
            result=session.result()
            natural_eof=not timed_out and not result.get('error') and result.get('status')=='final'
            organizer.finish(result.get('gloss_sequence',[]),natural_eof=natural_eof)
            record=dict(sample_id=item['sample_id'],result=result,events=events,calls=organizer.calls,
                timed_out=timed_out,natural_eof=natural_eof,
                published=organizer.published,capture_start_ms=session.capture_start_ms,capture_end_ms=session.capture_end_ms,
                total_wall_ms=(time.perf_counter()-started)*1000)
            write(run/'replay'/f"{item['sample_id']}.json",record)
            log(stage='replay',sample_id=item['sample_id'],words=result.get('gloss_sequence'),
                queue_dropped=result.get('queue_dropped'),llm_calls=len(organizer.calls),
                published=len(organizer.published),wall_ms=record['total_wall_ms'])
    finally:
        if active_session is None or not active_session.is_running:
            backend.close()
        else:
            log(stage='replay_abort',error='native_release_pending_backend_kept_alive')


def evaluate(run):
    from experiments.p1.eval import metrics as m
    import numpy as np
    manifest=read(run/'manifest.json')
    refs={r['sample_id']:r for r in read(run/'eval/selected_references.json')}
    records=[]
    for item in manifest['samples']:
        if item['kind']!='positive': continue
        raw=read(run/'offline'/f"{item['sample_id']}.json")
        llm=read(run/'organized'/f"{item['sample_id']}.json")['result']
        tokens=raw.get('result',{}).get('gloss_sequence',[])
        record=dict(refs[item['sample_id']],scene=item['scene'],
            predicted_gloss=tokens,raw_text=' '.join(tokens),llm=llm)
        record['slrt_error']=raw.get('error')
        record['word_errors']=m.wer_breakdown(record['gt_gloss'],' '.join(tokens))
        records.append(record)
    raw_pairs=[(r['ref_sentence_upstream'],r['raw_text']) for r in records]
    llm_pairs=[(r['ref_sentence_upstream'],r['llm']['text'] if r['llm']['status']=='draft' else '') for r in records]
    latency=[r['llm']['latency_ms'] for r in records if r['llm']['called']]
    summary=dict(n=len(records),data_scope=manifest['data_scope'],
        recognition_errors=sum(bool(r['slrt_error']) for r in records),
        empty_gloss=sum(not r['predicted_gloss'] for r in records),
        corpus_word_errors=m.corpus_wer([(r['gt_gloss'],' '.join(r['predicted_gloss'])) for r in records]),
        llm_status_counts={s:sum(r['llm']['status']==s for r in records) for s in ('draft','insufficient','error')},
        raw_gloss_text_metrics=m.corpus_metrics_cslt(raw_pairs),
        organized_text_metrics_all_requests_errors_as_empty=m.corpus_metrics_cslt(llm_pairs),
        llm_latency_ms={f'p{q}':float(np.percentile(latency,q)) if latency else None for q in (50,95)},
        tokens_in=sum(r['llm'].get('tokens_in') or 0 for r in records),
        tokens_out=sum(r['llm'].get('tokens_out') or 0 for r in records),
        independent_human_semantic_review='NOT_RUN',camera_and_new_signers='NOT_RUN')
    if summary['recognition_errors']: summary['corpus_word_errors']=None
    playback=[]
    for item in manifest['samples']:
        if not item['replay']: continue
        p=read(run/'replay'/f"{item['sample_id']}.json")
        finals=[e for e in p['events'] if e['status']=='final']
        drafts=[c for c in p['published'] if c['result']['status']=='draft']
        playback.append(dict(sample_id=item['sample_id'],kind=item['kind'],
            tokens=p['result'].get('gloss_sequence',[]),error=p['result'].get('error'),
            stream_status=p['result'].get('status'),stream_state=p['result'].get('state'),
            timed_out=p['timed_out'],natural_eof=p['natural_eof'],
            queue_dropped=p['result'].get('queue_dropped'),frames_received=p['result'].get('frames_received'),
            windows_processed=p['result'].get('windows_processed'),capture_end_ms=p['capture_end_ms'],
            first_final_ms=finals[0]['wall_elapsed_ms'] if finals else None,
            final_before_capture_eof=p['natural_eof'] and any(e['wall_elapsed_ms']<(p['capture_end_ms'] or 0) for e in finals),
            first_draft_ms=drafts[0]['completed_ms'] if drafts else None,
            last_draft=drafts[-1]['result']['text'] if drafts else '',
            llm_calls=len([c for c in p['calls'] if c['result']['called']]),
            llm_failures=sum(c['result']['status']=='error' for c in p['calls']),
            stale_discarded=sum(c['stale'] for c in p['calls'])))
    summary['replay']=playback
    write(run/'eval/records.json',records,clean=False)
    write(run/'eval/summary.json',summary,clean=False)
    log(stage='evaluate',n=summary['n'],corpus_word_errors=summary['corpus_word_errors'],
        llm_status_counts=summary['llm_status_counts'],llm_latency_ms=summary['llm_latency_ms'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','smoke','recognize','organize','replay','evaluate'])
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args(); run=args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'artifacts'): raise ValueError('Outputs must stay under repository artifacts')
    run.mkdir(parents=True,exist_ok=True)
    if args.phase!='prepare':
        manifest=read(run/'manifest.json')
        for path,expected in manifest['source_hashes'].items():
            if digest(ROOT/path)!=expected: raise ValueError(f'Source drift: {path}')
    globals()[args.phase](run)


if __name__=='__main__': main()
