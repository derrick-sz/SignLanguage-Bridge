"""Create/link the upstream SLRT Online asset layout (no data is fabricated)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from slbridge.slrt.official_layout import build_layout  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-root", type=Path, default=ROOT / "Paper-algorithm/SLRT-main")
    p.add_argument("--asset-root", type=Path, default=ROOT / "models/external/EMNLP24_Online/CSL-Daily")
    p.add_argument("--pose-asset-root", type=Path, default=ROOT / "models/external/uni_sign")
    p.add_argument("--pretrained-root", type=Path, default=ROOT / "models/external/SLRT_Online/pretrained_models")
    p.add_argument("--data-source", type=Path, default=None,
                   help="optional CSL-Daily continuous package (csl-daily.train/dev/test + csl-daily-keypoints.pkl)")
    p.add_argument("--two-stream-checkpoint", type=Path, default=None,
                   help="optional TwoStream-SLR S2G best.ckpt for Online/CTC_fusion/results")
    p.add_argument("--manifest", type=Path, default=None)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = build_layout(
        args.upstream_root,
        args.asset_root,
        pose_asset_root=args.pose_asset_root,
        pretrained_root=args.pretrained_root,
        data_source=args.data_source,
        two_stream_checkpoint=args.two_stream_checkpoint,
        manifest_path=args.manifest,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
