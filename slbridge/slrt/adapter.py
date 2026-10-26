"""Register SLRT with the existing typed model manager without replacing RF."""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from .._io import canonical_hash
from ..contracts import GlossToken, GlossSequencePrediction, RecognitionResult
from ..model_adapters import (InputRepresentation,ModelCapabilities,ModelInputContract,
                             get_input_representation,register_input_representation)
from .assets import SLRTConfig,PROTOCOL,SLRTResourceError
from .backend import SLRTBackend

REPRESENTATION = 'slrt_rgb_hrnet133_v1'


class SLRTAdapter:
    def __init__(self,config):
        if config.pose_backend == 'hrnet-npz':
            raise SLRTResourceError('hrnet-npz is video-bound and supported by the standalone video CLI only; shared adapters require a live pose frontend')
        self.config = config
        try:
            get_input_representation(REPRESENTATION)
        except ValueError:
            register_input_representation(InputRepresentation(REPRESENTATION,(None,None,None,3),dtype='uint8',layout='T,H,W,C'))
        self._backend = None
        self._closed = False
        version = canonical_hash(dict(protocol=PROTOCOL,config=config.to_dict()))
        self.capabilities = ModelCapabilities(config.model_id,version,'csl','gloss_sequence',
            (ModelInputContract(REPRESENTATION,(None,None,None,3),'uint8',version,('file','captured_clip')),),
            (), 'pytorch_slrt',('cpu','cuda'),('clip',),'uncalibrated_mean_ensemble_probability')
        # Validate vocabulary and hashes before advertising a route. Torch/GPU
        # loading stays behind warmup, as with the existing optional adapter.
        from .assets import load_vocabulary,require_file
        require_file(config.checkpoint_path,'SLRT checkpoint',config.checkpoint_sha256)
        require_file(config.vocabulary_path,'SLRT vocabulary',config.vocabulary_sha256)
        self.capabilities = replace(self.capabilities,vocabulary=load_vocabulary(config.vocabulary_path))

    @property
    def recognizer(self):
        return self

    @property
    def runtime_info(self):
        return dict(backend='pytorch_slrt',config=self.config.to_dict(),
                    device=self._backend.runtime.device if self._backend is not None else None,
                    pose_readiness=getattr(self,'_pose_check',None),
                    model=self._backend.runtime.info if self._backend is not None else None,
                    stream_interface='SLRTBackend.new_stream; independent lifecycle, not shared clip scheduling')

    def _load(self):
        if self._closed:
            raise RuntimeError('SLRT adapter is closed')
        if self._backend is None:
            self._backend = SLRTBackend(self.config)

    def warmup(self):
        self._load()
        from .pose import make_frontend
        frontend = make_frontend(self.config)
        try:
            self._pose_check = frontend.info
        finally:
            frontend.close()
        import numpy as np
        self._backend.runtime.predict(np.zeros((self.config.window_size,320,320,3),np.uint8),
                                      np.zeros((self.config.window_size,133,3),np.float32))

    def infer(self,source,request,*,input_kind):
        self.capabilities.negotiate(request,input_kind=input_kind)
        request.check_active()
        self._load()
        start = perf_counter()
        if input_kind == 'file':
            report = self._backend.recognize_video(source,cancellation=request.check_active)
        elif input_kind == 'captured_clip':
            if self.config.pose_backend == 'hrnet-npz':
                raise SLRTResourceError('hrnet-npz is file-bound; captured clips require a live pose frontend')
            stream = self._backend.new_stream()
            try:
                for packet in source.frames:
                    request.check_active()
                    stream.feed(packet)
                request.check_active()
                stream.flush()
                report = stream.result()
            finally:
                stream.close()
        else:
            raise ValueError('SLRT supports file/captured_clip; prepared MediaPipe arrays are incompatible')
        request.check_active()
        tokens = tuple(GlossToken(t['token_id'],t['gloss'],t['score'],tuple(t['time_range'])) for t in report['tokens'])
        quality = dict(slrt=report,boundary_source='sliding_window_anchor_estimate',
                       decoder=f'center_majority_{self.config.vote_size}_then_run_compression',
                       raw_gloss=True,translation_available=False)
        result = RecognitionResult('recognized' if tokens else 'rejected',request.request_id,self.config.model_id,'csl',
            GlossSequencePrediction(tokens,'slrt_window_majority'),quality,
            reason=None if tokens else 'no_final_gloss',task_type='gloss_sequence',
            score_type=self.capabilities.score_type,
            time_range=(tokens[0].time_range[0],tokens[-1].time_range[1]) if tokens else None)
        timing = dict(preprocess_ms=sum(report['timings'][k]['total'] for k in ('pose_ms','frame_preprocess_ms','preprocess_ms')),
                      inference_ms=report['timings']['inference_ms']['total'],total_ms=(perf_counter()-start)*1000)
        return result,timing

    def close(self):
        self._closed = True
        if self._backend is not None:
            self._backend.close()
            self._backend = None


def load_slrt_adapter(path,options):
    config = SLRTConfig.from_file(path/'manifest.json')
    config = replace(config,device=options.device,device_id=options.device_id,cpu_threads=options.intra_op_threads)
    return SLRTAdapter(config)
