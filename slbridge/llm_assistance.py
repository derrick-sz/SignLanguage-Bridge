"""Optional bounded text-only advice. It never replaces a visual translation."""
from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
import json
import inspect
import math
import os
from pathlib import Path
import re
import threading
import time
import unicodedata
from urllib.parse import urlsplit
from uuid import uuid4

_PROTECTED = re.compile(r'[不没别勿无我你他她它\d一二三四五六七八九十百千万两几多少每]|'
    r'(?:今|明|昨|前|后)天|早上|上午|中午|下午|晚上|星期|礼拜|周[一二三四五六日天]')
_FUNCTION_CHARS = set('的了是在和与把被就也都着过很请吗呢啊')
_PROMPT = ('你只做手语识别文字的辅助核对。两个视觉输出都可能错误，不能替用户裁决事实。'
 '只依据给出的实际手语词和句子草稿；不得添加人物、否定、时间、数字、动作或对象。'
 '有冲突返回risk并简短提示核对，信息不足返回insufficient，不确定时不要拼句。'
 '仅返回JSON：{"status":"suggestion或risk或insufficient","text":"建议或核对提示",'
 '"used_token_indices":[从0开始的实际词下标]}。不输出解释过程。')


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool = False
    base_url: str = 'http://127.0.0.1:8788/v1'
    model: str = 'deepseek-v4.1-flash'
    api_key_env: str = 'WB_PROXY_API_KEY'
    credentials_file: str | None = None
    credentials_ref: str = 'WB_PROXY_API_KEY'
    timeout_seconds: float = 8.
    max_tokens: int = 1024
    max_jobs: int = 64
    job_ttl_seconds: float = 120.

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'127.0.0.1','localhost','::1'} or not parsed.port
                or parsed.username or parsed.password or parsed.path.rstrip('/') != '/v1' or parsed.query or parsed.fragment):
            raise ValueError('LLM backend must be a server-owned loopback /v1 origin')
        if type(self.enabled) is not bool or not isinstance(self.model,str) or not 1 <= len(self.model) <= 128:
            raise ValueError('Invalid LLM model configuration')
        if not re.fullmatch(r'[A-Z_][A-Z0-9_]{0,79}',self.api_key_env):
            raise ValueError('Invalid API key environment variable')
        if (type(self.timeout_seconds) not in {int,float} or not math.isfinite(self.timeout_seconds)
                or not 1 <= self.timeout_seconds <= 15 or type(self.max_tokens) is not int or not 128 <= self.max_tokens <= 2048
                or type(self.max_jobs) is not int or not 4 <= self.max_jobs <= 256
                or type(self.job_ttl_seconds) not in {int,float} or not 20 <= self.job_ttl_seconds <= 3600):
            raise ValueError('Invalid LLM budget')


def _credentials(config):
    value = os.environ.get(config.api_key_env)
    if value:
        return value
    path = Path(config.credentials_file) if config.credentials_file else Path.home()/'.dsh'/'.credentials.yaml'
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
        value = (data.get('refs') or {}).get(config.credentials_ref)
        return str(value) if value else None
    except Exception:
        return None


def validate_advice(data, glosses, sentence):
    if (not isinstance(data,dict) or set(data) != {'status','text','used_token_indices'}
            or data['status'] not in {'suggestion','risk','insufficient'}
            or not isinstance(data['text'],str) or len(data['text']) > 512):
        raise ValueError('Invalid LLM advice structure')
    indices = data['used_token_indices']
    if (not isinstance(indices,list) or len(indices) > len(glosses) or
            any(type(i) is not int or not 0 <= i < len(glosses) for i in indices) or len(set(indices)) != len(indices)):
        raise ValueError('Invalid LLM word evidence')
    if data['status'] != 'suggestion':
        return {'status':data['status'],'text':data['text'] if data['status']=='risk' else ''}
    if not indices or not data['text'].strip():
        raise ValueError('Suggestion needs actual word evidence')
    normalize = lambda s:re.sub(r'\s+','',unicodedata.normalize('NFKC',s)).replace('。','.')
    proposal = data['text'].strip()
    if (_PROTECTED.search(sentence) or _PROTECTED.search(proposal)) and normalize(sentence) != normalize(proposal):
        return {'status':'risk','text':'建议改变了涉及否定、人称、数量或时间的表达，请核对原译文与动作。'}
    allowed = set(sentence+''.join(glosses)) | _FUNCTION_CHARS
    if any(ch not in allowed for ch in re.findall(r'[\u4e00-\u9fff]',proposal)):
        return {'status':'risk','text':'建议包含缺少视觉证据的新内容，原译文已保留，请核对。'}
    return {'status':'suggestion','text':proposal}


class AdviceQueue:
    """One network request in flight plus one pending, separate from native ML."""
    def __init__(self, config, *, backend=None):
        if config.enabled and backend is not None and not inspect.iscoroutinefunction(backend):
            raise TypeError('Advice backend must be async and cancellable')
        self.config = config
        self._backend = backend
        self._api_key = _credentials(config) if config.enabled and backend is None else None
        self.enabled = config.enabled and (backend is not None or bool(self._api_key))
        self._condition = threading.Condition(threading.RLock())
        self._jobs = OrderedDict()
        self._latest = {}
        self._pending = deque()
        self._closed = False
        self._thread = None
        self._loop = self._task = None
        if self.enabled:
            self._thread = threading.Thread(target=self._run,name='csl-advice',daemon=True)
            self._thread.start()

    def _forget(self, job_id):
        job = self._jobs.pop(job_id)
        key = job['_key']
        if self._latest.get(key) == job_id:
            self._latest.pop(key,None)

    def _prune(self):
        for job_id,job in list(self._jobs.items()):
            if job['status'] not in {'pending','running'} and (time.monotonic()-job['_created']>self.config.job_ttl_seconds
                                                            or len(self._jobs)>=self.config.max_jobs):
                self._forget(job_id)

    def submit(self, *, session_id, source_epoch, utterance_id, revision, glosses, text):
        if (not isinstance(session_id,str) or not 1 <= len(session_id) <= 80 or
                any(type(n) is not int for n in (source_epoch,utterance_id,revision)) or
                min(source_epoch,utterance_id)<0 or revision<1 or not isinstance(text,str) or not 1<=len(text)<=1024
                or not isinstance(glosses,list) or not 1<=len(glosses)<=64 or
                any(not isinstance(t,str) or not 1<=len(t)<=128 for t in glosses)):
            raise ValueError('Invalid generated evidence for advice')
        key = session_id,utterance_id
        with self._condition:
            self._prune()
            old = self._jobs.get(self._latest.get(key))
            if old is not None and old['revision'] >= revision:
                return old['job_id']
            job_id = uuid4().hex
            job = dict(job_id=job_id,session_id=session_id,source_epoch=source_epoch,utterance_id=utterance_id,
                       revision=revision,status='pending',text='',model=self.config.model,
                       _key=key,_created=time.monotonic(),_glosses=list(glosses),_sentence=text)
            self._jobs[job_id] = job
            self._latest[key] = job_id
            if not self.enabled or self._closed:
                job.update(status='unavailable',reason='disabled_or_missing_credentials')
            elif self._pending and self._jobs[self._pending[0]]['_key'] != key:
                job.update(status='busy',reason='advice_budget')
            else:
                if self._pending:
                    self._jobs[self._pending.popleft()].update(status='superseded',text='')
                self._pending.append(job_id)
                self._condition.notify()
            return job_id

    def get(self, job_id):
        with self._condition:
            self._prune()
            job = self._jobs.get(job_id)
            return deepcopy({k:v for k,v in job.items() if not k.startswith('_')}) if job else None

    async def _http(self, glosses, sentence):
        import httpx
        async with httpx.AsyncClient(timeout=self.config.timeout_seconds,trust_env=False) as client:
            async with client.stream('POST',self.config.base_url.rstrip('/')+'/chat/completions',
                 headers={'Authorization':'Bearer '+self._api_key},json=dict(model=self.config.model,
                 temperature=0,max_tokens=self.config.max_tokens,stream=False,
                 messages=[{'role':'system','content':_PROMPT},{'role':'user','content':json.dumps(
                     {'observed_glosses':glosses,'sentence_draft':sentence},ensure_ascii=False)}])) as response:
                response.raise_for_status()
                body=bytearray()
                async for chunk in response.aiter_bytes(chunk_size=4096):
                    if len(body)+len(chunk)>65536:
                        raise ValueError('Advice response exceeds its byte budget')
                    body.extend(chunk)
                data=json.loads(body)
        if data.get('model') != self.config.model or data['choices'][0].get('finish_reason') != 'stop':
            raise ValueError('LLM model changed or completion incomplete')
        return json.loads(data['choices'][0]['message']['content'])

    def _run(self):
        loop=asyncio.new_event_loop()
        with self._condition:self._loop=loop
        try:
            self._work(loop)
        finally:
            loop.close()
            with self._condition:self._loop=self._task=None

    def _work(self,loop):
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait(.5)
                if self._closed:
                    return
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                if self._latest.get(job['_key']) != job_id:
                    job.update(status='superseded',text='')
                    continue
                job['status'] = 'running'
                self._task=loop.create_task(asyncio.wait_for(
                    (self._backend or self._http)(job['_glosses'],job['_sentence']),
                    timeout=self.config.timeout_seconds))
            started = time.perf_counter()
            try:
                data = loop.run_until_complete(self._task)
                result = validate_advice(data,job['_glosses'],job['_sentence'])
            except asyncio.TimeoutError:
                result=dict(status='timeout',text='',reason='overall_deadline')
            except asyncio.CancelledError:
                result=dict(status='cancelled',text='')
            except Exception as error:
                result = dict(status='error',text='',reason=type(error).__name__)
            with self._condition:
                self._task=None
                if self._closed:
                    result=dict(status='cancelled',text='')
                elif self._latest.get(job['_key']) != job_id:
                    result = dict(status='superseded',text='')
                job.update(result,latency_ms=(time.perf_counter()-started)*1000)

    def close(self):
        with self._condition:
            self._closed = True
            while self._pending:
                self._jobs[self._pending.popleft()].update(status='cancelled',text='')
            self._condition.notify_all()
            if self._loop is not None and self._task is not None:
                self._loop.call_soon_threadsafe(self._task.cancel)
        if self._thread:
            self._thread.join(2)
        return self._thread is None or not self._thread.is_alive()


class SessionAdvice:
    def __init__(self, queue, session):
        self.queue,self.session = queue,session
    def submit(self, **data):
        job_id = self.queue.submit(session_id=self.session.session_id,source_epoch=self.session.source_epoch,**data)
        self.session.emit('assistance',status='pending',text='',job_id=job_id,
                          utterance_id=data['utterance_id'],revision=data['revision'])
