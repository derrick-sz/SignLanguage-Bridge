"""Inference-only, strict-key equivalent of the released CSL-Daily network.

Only the configured 5-block two-stream/triplehead path is included. No training
loss, pretrained K400 initialization, distributed state or label inputs.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ._s3d_base import S3D_base


class _Stream(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.backbone = S3D_base(channels, 5)


class _Lateral(nn.Module):
    def __init__(self, channels, *, up):
        super().__init__()
        if up:
            self.conv = nn.ConvTranspose3d(channels, channels, (1, 3, 3), (1, 2, 2), (0, 1, 1), (0, 1, 1), bias=False)
        else:
            self.conv = nn.Conv3d(channels, channels, (1, 3, 3), (1, 2, 2), (0, 1, 1), bias=False)


class TwoStreamS3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_stream = _Stream(3)
        self.pose_stream = _Stream(63)
        self.rgb_stream_lateral = nn.ModuleList([_Lateral(c, up=True) for c in (64, 192, 480, 832)])
        self.pose_stream_lateral = nn.ModuleList([_Lateral(c, up=False) for c in (64, 192, 480, 832)])

    def forward(self, rgb, pose):
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            rgb, pose = rgb_layer(rgb), pose_layer(pose)
            if i in (0, 3, 6, 12):
                j = (0, 3, 6, 12).index(i)
                # Both fusions use the unfused stage tensors, as in upstream.
                rgb, pose = rgb + self.rgb_stream_lateral[j].conv(pose), pose + self.pose_stream_lateral[j].conv(rgb)
        # Use the original pooling operator rather than a mathematically equal
        # reduction with different float32 accumulation on CUDA.
        def pool(x):
            x = F.avg_pool3d(x, (1,x.size(3),x.size(4)),stride=1)
            return x.view(x.size(0),x.size(1),x.size(2)).permute(0,2,1)
        return pool(rgb), pool(pose)


class _Head(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.gloss_output_layer = nn.Linear(dim, classes)

    def forward(self, x):
        return self.gloss_output_layer(x.mean(dim=1))


class _Recognition(nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.visual_backbone_twostream = TwoStreamS3D()
        self.visual_head = _Head(1024, classes)
        self.visual_head_keypoint = _Head(1024, classes)
        self.visual_head_fuse = _Head(2048, classes)

    def forward(self, rgb, heatmaps):
        rgb, kp = self.visual_backbone_twostream(rgb, heatmaps)
        heads = (self.visual_head(rgb), self.visual_head_keypoint(kp), self.visual_head_fuse(torch.cat([rgb, kp], dim=-1)))
        return torch.stack([h.softmax(-1) for h in heads]).mean(0)


class SLRTNetwork(nn.Module):
    def __init__(self, classes=2001):
        super().__init__()
        self.recognition_network = _Recognition(classes)

    def forward(self, rgb, heatmaps):
        return self.recognition_network(rgb, heatmaps)
