"""Explicit pose frontends; MediaPipe is never an SLRT input."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .assets import SLRTResourceError, require_file, sha256
from .preprocess import canonical_frame, select_keypoints


class HRNetFrontend:
    """Reuse the same legacy APIs and detector/pose models as upstream gen_pose.

    Install these in an isolated legacy OpenMMLab runtime. No automatic model
    downloads or mutation of the working project's dependencies.
    """
    def __init__(self, config):
        paths = {}
        for name in ('detector_config', 'detector_checkpoint', 'hrnet_config', 'hrnet_checkpoint'):
            value = getattr(config, name)
            if value is None:
                raise SLRTResourceError(f'Official HRNet frontend needs {name}; EMNLP24_Online contains no HRNet/detector assets. See docs/slrt-integration.md')
            paths[name] = require_file(Path(value), name)
        try:
            import torch  # Load native Torch DLLs before other CV runtimes.
            from mmdet.apis import init_detector, inference_detector
            from mmpose.apis import init_pose_model, inference_top_down_pose_model
        except (ImportError, OSError, AttributeError) as e:
            raise SLRTResourceError('Official HRNet requires legacy mmpose 0.x + mmdet 2.x + compatible mmcv-full; use an isolated environment') from e
        device = config.pose_device
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = f'cuda:{config.device_id}' if device == 'cuda' else device
        self.detector = init_detector(str(paths['detector_config']), str(paths['detector_checkpoint']), device=device)
        self.model = init_pose_model(str(paths['hrnet_config']), str(paths['hrnet_checkpoint']), device=device)
        if str(self.detector.CLASSES[0]).lower() != 'person':
            raise SLRTResourceError('HRNet detector must have person as COCO class 0')
        self._detect, self._pose = inference_detector, inference_top_down_pose_model
        self.info = dict(frontend='hrnet_w48_dark', experimental=False, device=device,
                         assets={k: dict(path=str(v), sha256=sha256(v)) for k,v in paths.items()},
                         coordinate_system='coco_wholebody_133_xyc_512_v1')

    def extract(self, bgr, *, frame_index, timestamp):
        from PIL import Image
        frame = np.asarray(Image.fromarray(bgr[..., ::-1]).resize((512,512), Image.Resampling.BILINEAR))[..., ::-1].copy()
        boxes = self._detect(self.detector, frame)[0]
        boxes = boxes[(boxes[:,4] >= .5) & ((boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]) >= 1600)]
        if len(boxes) == 0:
            return None
        poses = self._pose(self.model, frame, [dict(bbox=b) for b in boxes], format='xyxy')[0]
        if not poses:
            return None
        kp = np.asarray(max(poses, key=lambda p: p['bbox'][-1])['keypoints'], np.float32)
        select_keypoints(kp[None])
        return kp

    def close(self):
        self.model = self.detector = None


class RTMPoseExperimentalFrontend:
    def __init__(self, config):
        if config.device_id != 0:
            raise SLRTResourceError('Experimental RTMLib frontend currently supports device_id=0 only')
        cache = Path(config.rtmpose_cache_dir).resolve()
        # Never start a hidden large download when the cache is absent.
        expected = ('hub/checkpoints/yolox_tiny_8xb8-300e_humanart-6f3252f9.onnx',
                    'hub/checkpoints/rtmw-dw-l-m_simcc-cocktail14_270e-256x192_20231122.onnx')
        assets = [require_file(cache / name, 'Experimental RTMPose cached model') for name in expected]
        try:
            import torch
            from ..uni_sign import RtmposeExtractor
            device = config.pose_device
            if device == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # The existing wrapper handles native DLL ordering and verifies providers.
            self.extractor = RtmposeExtractor(mode='lightweight', backend='onnxruntime', device=device,
                                              cache_dir=cache, strict_cuda=config.pose_device == 'cuda')
        except (ImportError, OSError, RuntimeError) as e:
            raise SLRTResourceError(f'Experimental RTMPose frontend failed: {e}') from e
        self.info = dict(frontend='rtmpose-experimental', experimental=True,
                         coordinate_system='coco_wholebody_133_xyc_512_v1',
                         assets=[dict(path=str(p), sha256=sha256(p)) for p in assets],
                         **self.extractor.runtime_info)

    def extract(self, bgr, *, frame_index, timestamp):
        frame = canonical_frame(bgr)
        keypoints, scores = self.extractor.extract(frame)
        # No unverified multi-person identity selection across windows.
        if len(keypoints) != 1:
            return None
        kp = np.concatenate([np.asarray(keypoints[0],np.float32) * (512/320),
                             np.asarray(scores[0],np.float32).clip(0,1)[:,None]], axis=-1)
        select_keypoints(kp[None])
        if float(kp[:11,2].mean()) < .2 or float(kp[91:,2].mean()) < .1:
            return None
        return kp

    def close(self):
        self.extractor.close()


class HRNetNPZFrontend:
    def __init__(self, path, video_path):
        path = require_file(Path(path), 'HRNet NPZ')
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata_json'].item()))
            self.keypoints = data['keypoints'].copy()
            self.frame_indices = data['frame_indices'].copy()
            self.timestamps = data['timestamps'].copy()
        if meta.get('video_sha256') != sha256(Path(video_path)):
            raise SLRTResourceError('HRNet cache video SHA256 mismatch')
        if (meta.get('coordinate_system') != 'coco_wholebody_133_xyc_512_v1' or
                meta.get('frontend') != 'hrnet_w48_dark'):
            raise SLRTResourceError('HRNet cache requires declared HRNet133 x/y/confidence in 512 reference pixels')
        select_keypoints(self.keypoints)
        n = len(self.keypoints)
        if (self.frame_indices.shape != (n,) or self.frame_indices.dtype.kind not in 'iu' or self.timestamps.shape != (n,) or
                not np.array_equal(self.frame_indices,np.arange(n)) or not np.isfinite(self.timestamps).all() or
                n < 2 or (np.diff(self.timestamps) <= 0).any()):
            raise SLRTResourceError('HRNet cache must cover every source frame in monotonic order')
        self.info = dict(frontend='hrnet-npz', experimental=False, cache_sha256=sha256(path), metadata=meta)

    def extract(self, bgr, *, frame_index, timestamp):
        if frame_index >= len(self.keypoints) or abs(float(self.timestamps[frame_index])-timestamp) > .001:
            raise SLRTResourceError('HRNet cache frame index/timestamp does not match this video')
        kp = self.keypoints[frame_index].copy()
        return kp if kp[:,2].max() > 0 else None

    def close(self):
        self.keypoints = None


def make_frontend(config, *, pose_npz=None, video_path=None):
    if config.pose_backend == 'hrnet-npz':
        if pose_npz is None or video_path is None:
            raise SLRTResourceError('hrnet-npz requires --pose-npz and a bound video, cannot be used for a camera')
        return HRNetNPZFrontend(pose_npz,video_path)
    if pose_npz is not None:
        raise SLRTResourceError('--pose-npz requires pose_backend=hrnet-npz')
    return HRNetFrontend(config) if config.pose_backend == 'hrnet' else RTMPoseExperimentalFrontend(config)
