"""Installable desktop launcher; parsing --help never imports the GUI."""
from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from ._io import configure_utf8_streams
from .runtime_options import RuntimeOptions


def default_catalog():
    """Prefer the user's editable working-directory catalog, then package seed."""
    local = Path.cwd() / "configs/libras_caption_catalog.json"
    if local.is_file():
        return local
    return resources.files("slbridge").joinpath("resources/libras_caption_catalog.json")


def default_model():
    """Models remain external and are never looked up in site-packages."""
    local = Path.cwd() / "artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009"
    return local if local.is_dir() else None


def main(argv=None):
    configure_utf8_streams()
    parser = argparse.ArgumentParser(
        prog="slbridge-desktop",
        description="Libras 动作分类与中国手语词序列识别、中文字幕",
    )
    parser.add_argument("--model", type=Path, help="RGB clip 模型包目录")
    parser.add_argument("--catalog", type=Path, help="字幕词典 JSON")
    parser.add_argument("--segment-config", type=Path, help="自动端点参数 JSON；未指定时使用未经真人验证的启发式默认值")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="模型执行设备")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--intra-op-threads", type=int, default=4)
    parser.add_argument("--cpu-conv-subnormals", choices=("flush_to_zero", "preserve"), default="flush_to_zero")
    parser.add_argument("--gpu-mem-limit-mb", type=int, default=RuntimeOptions().gpu_mem_limit_mb)
    parser.add_argument("--cuda-dll-dir", type=Path, action="append", default=[])
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video", type=Path, help="启动时打开的视频文件")
    inputs.add_argument("--camera", type=int, help="启动时打开的摄像头设备编号")
    args = parser.parse_args(argv)
    try:
        options = RuntimeOptions(device=args.device, device_id=args.device_id,
                                 intra_op_threads=args.intra_op_threads, gpu_mem_limit_mb=args.gpu_mem_limit_mb,
                                 cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir),
                                 cpu_conv_subnormals=args.cpu_conv_subnormals)
    except ValueError as error:
        parser.error(str(error))
    import sys
    from PySide6.QtWidgets import QApplication
    from .desktop import SignLanguageApp

    model = args.model if args.model is not None else default_model()
    segment_config = None
    if args.segment_config:
        from ._io import read_json
        from .segmentation import SegmentConfig
        try:
            segment_config = SegmentConfig(**read_json(args.segment_config))
        except (ValueError, TypeError, OSError) as error:
            parser.error(str(error))
    qt = QApplication.instance() or QApplication([sys.argv[0]])
    app = SignLanguageApp(model=model, video=args.video, camera=args.camera, runtime_options=options,
                          segment_config=segment_config)
    if args.catalog is not None:
        app.load_catalog(args.catalog)
    else:
        # as_file also supports zipped package resources. load_catalog consumes
        # the JSON synchronously before a temporary resource file is released.
        with resources.as_file(default_catalog()) as catalog_path:
            app.load_catalog(catalog_path)
    app.show()
    return qt.exec()


if __name__ == "__main__":
    raise SystemExit(main())
