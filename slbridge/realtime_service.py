"""Optional single-owner live coordinator over the existing CSL model lease."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import hashlib
import math
from pathlib import Path
import threading
import time

from .realtime_protocol import PROTOCOL, MAX_MESSAGE_BYTES, MAX_SESSION_SECONDS
from .realtime_worker import LiveSession
from .runtime import RuntimeBusy
from .slrt.assets import SLRTConfig


@dataclass(frozen=True)
class RealtimeConfig:
    slrt: SLRTConfig
    pause_seconds: float = .8
    max_seconds: float = 8.
    queue_size: int = 16
    llm: dict = field(default_factory=lambda:{'enabled':False})
    source_sha256: str | None = None
    slrt_source_sha256: str | None = None

    def __post_init__(self):
        for name, low, high in [('pause_seconds',.4,2),('max_seconds',1,8)]:
            value = getattr(self,name)
            if type(value) not in {int,float} or not math.isfinite(value) or not low <= value <= high:
                raise ValueError('Invalid live '+name)
        if type(self.queue_size) is not int or not 1 <= self.queue_size <= 32:
            raise ValueError('Invalid live queue_size')
        if not isinstance(self.llm,dict) or type(self.llm.get('enabled',False)) is not bool:
            raise ValueError('Invalid live LLM configuration')
        from .llm_assistance import LLMConfig
        try:
            LLMConfig(**self.llm)
        except TypeError as error:
            raise ValueError('Unknown LLM configuration field') from error

    @classmethod
    def from_file(cls, path):
        path = Path(path).resolve()
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        allowed = {'schema_version','slrt_config_file','pause_seconds','max_seconds','queue_size','llm'}
        if (not isinstance(data,dict) or set(data)-allowed or type(data.get('schema_version')) is not int
                or data['schema_version'] != 1 or not isinstance(data.get('slrt_config_file'),str)):
            raise ValueError('Invalid realtime server configuration')
        slrt_file = Path(data['slrt_config_file'])
        if not slrt_file.is_absolute():
            slrt_file = path.parent/slrt_file
        return cls(slrt=SLRTConfig.from_file(slrt_file), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   slrt_source_sha256=hashlib.sha256(slrt_file.read_bytes()).hexdigest(), **{k:v for k,v in data.items()
                   if k not in {'schema_version','slrt_config_file'}})


class UnavailableRealtime:
    def status(self):
        return dict(enabled=True, ready=False, protocol=PROTOCOL, reason='initialization_failed')
    def create_session(self, start):
        raise RuntimeBusy('Live runtime is unavailable')
    def assistance_result(self, job_id):return None
    def close(self):pass


class RealtimeService:
    def __init__(self, manager, config, *, runtime_factory=None):
        self.manager, self.config = manager, config
        self._runtime_factory = runtime_factory
        self.runtime = None
        self._expected = None
        self._generation = None
        self._model_id = None
        self._lock = threading.RLock()
        self._active = None
        self._last_performance = None
        self._closed = False
        self.assister = None

    def warmup(self):
        import numpy as np
        with self.manager.native_session() as native:
            info = self.manager.describe() or {}
            if (info.get('sign_language') != 'csl' or not callable(getattr(native,'live_pose_extractor',None))
                    or not callable(getattr(native,'translate_pose_arrays',None))):
                raise ValueError('Live CSL requires the existing Uni-Sign pose backend')
            if self._runtime_factory is None:
                from .slrt.runtime import SLRTRuntime
                factory = SLRTRuntime
            else:
                factory = self._runtime_factory
            self.runtime = factory(self.config.slrt)
            self.runtime.predict(np.zeros((16,320,320,3),np.uint8),np.zeros((16,133,3),np.float32))
            self.runtime.clear_frame_cache()
            self._expected, self._generation = native, self.manager.generation
            self._model_id = info['model_id']
        from .llm_assistance import AdviceQueue, LLMConfig
        self.assister = AdviceQueue(LLMConfig(**self.config.llm))

    def status(self):
        with self._lock:
            ready = (not self._closed and self.runtime is not None and self._expected is not None
                     and self.manager.recognizer is self._expected and self.manager.generation == self._generation)
            return dict(enabled=True, ready=ready, busy=self._active is not None or
                 self.manager.status()['state']=='inferencing', protocol=PROTOCOL,
                 reason=None if ready else 'model_not_ready_or_changed',
                 pipeline={'id':'slrt-pause-unisign-v1','translation_model_id':self._model_id,
                           'model_generation':self._generation,'slrt_model_id':self.config.slrt.model_id,
                           'realtime_config_sha256':self.config.source_sha256,
                           'slrt_config_sha256':self.config.slrt_source_sha256,
                           'slrt_checkpoint_sha256':self.config.slrt.checkpoint_sha256,
                           'precision':self.config.slrt.precision,'pose_detect_stride':self.config.slrt.pose_detect_stride,
                           'stride':self.config.slrt.stride,'vote_size':self.config.slrt.vote_size},
                 input={'image_format':'JPEG','max_width':512,'max_height':512,'capture_fps_range':[10,30],
                        'pose_protocol':'shared_rtmpose133_square320_ref512_v1'},
                 features={'incremental_words':True,'sentence_translation':True,
                           'word_evidence_review':True,
                           'llm_assistance':bool(self.assister and self.assister.enabled),
                           'native_streaming_translation':False},
                 limits={'queue_frames':self.config.queue_size,'max_message_bytes':MAX_MESSAGE_BYTES,
                         'max_session_seconds':MAX_SESSION_SECONDS,'max_utterance_seconds':self.config.max_seconds},
                 performance={'active':self._active.performance() if self._active is not None else None,
                              'last':self._last_performance})

    def _prepare_live_worker(self, extractor):
        import numpy as np
        started = time.perf_counter()
        try:
            pose = extractor.warmup_live() if callable(getattr(extractor,'warmup_live',None)) else None
            slrt_started = time.perf_counter()
            self.runtime.predict(np.zeros((16,320,320,3),np.uint8),np.zeros((16,133,3),np.float32))
            return dict(total_ms=(time.perf_counter()-started)*1000, pose=pose,
                        slrt_ms=(time.perf_counter()-slrt_started)*1000)
        finally:
            self.runtime.clear_frame_cache()

    def create_session(self, start):
        if not 10 <= start['fps'] <= 30:
            raise ValueError('Live capture must be in [10,30] FPS')
        with self._lock:
            if not self.status()['ready'] or self._active is not None:
                raise RuntimeBusy('Live session is busy or model changed')
            # Assemble immutable capabilities before taking the backend lock.
            # A different status route may hold the manager lock while reading
            # native runtime_info; do not take it in the reverse order below.
            pipeline = self.status()['pipeline']
            @contextmanager
            def engine_factory(session):
                from .slrt.pose import RTMPoseExperimentalFrontend
                from .realtime_engine import RealtimeEngine
                from .llm_assistance import SessionAdvice
                with self.manager.native_session(expected_model=self._expected) as native:
                    with native.live_pose_extractor(detect_stride=self.config.slrt.pose_detect_stride) as extractor:
                        self.runtime.clear_frame_cache()
                        preparation = self._prepare_live_worker(extractor)
                        frontend = RTMPoseExperimentalFrontend.from_shared(self.config.slrt,extractor)
                        engine = RealtimeEngine(session,self.runtime,frontend,self.config.slrt,
                             translator=native.translate_pose_arrays,fps=start['fps'],
                             pause_seconds=self.config.pause_seconds,max_seconds=self.config.max_seconds,
                             assister=SessionAdvice(self.assister,session) if start.get('assistance',False) else None,
                             model_info=pipeline)
                        engine.info['preparation'] = preparation
                        try:
                            yield engine
                        finally:
                            engine.close()
                            self.runtime.clear_frame_cache()
            session = LiveSession(source_epoch=start['source_epoch'], engine_factory=engine_factory,
                                  queue_size=self.config.queue_size, on_done=self._release)
            self._active = session
            return session

    def _release(self, session):
        with self._lock:
            if self._active is session:
                self._last_performance = session.performance()
                self._active = None

    def assistance_result(self, job_id):
        return self.assister.get(job_id) if self.assister else None

    def close(self):
        self._closed = True
        with self._lock:
            active = self._active
        if active is not None:
            active.cancel('server_shutdown')
            if not active.join(30):
                raise RuntimeBusy('Native work still owns the live model; do not release it')
            self._release(active)
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None
        if self.assister is not None:
            if not self.assister.close():
                raise RuntimeBusy('Advice worker did not stop within its cancellation budget')
