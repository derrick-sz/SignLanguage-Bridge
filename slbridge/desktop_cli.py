"""Installable desktop launcher; parsing --help never imports the GUI."""
from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

from ._io import configure_utf8_streams
from .desktop_models import default_language_models, default_unisign_model
from .runtime_options import RuntimeOptions


def default_catalog():
    """Prefer the user's editable working-directory catalog, then package seed."""
    local = Path.cwd() / "configs/libras_caption_catalog.json"
    if local.is_file():
        return local
    return resources.files("slbridge").joinpath("resources/libras_caption_catalog.json")


def default_model():
    """Models remain external and are never looked up in site-packages."""
    local = default_language_models()["libras"]
    return local if local.is_dir() else None


def main(argv=None):
    configure_utf8_streams()
    parser = argparse.ArgumentParser(
        prog="slbridge-desktop",
        description="Libras 动作分类、中国手语词序列与 Uni-Sign 中文句子翻译",
    )
    parser.add_argument("--model", type=Path, help="RGB clip 模型包目录")
    parser.add_argument("--unisign", action="store_true", help="加载 Uni-Sign Pose-only CSL-Daily 实验模型")
    parser.add_argument("--unisign-model", type=Path, help="Uni-Sign model_bundle 目录")
    parser.add_argument("--slrt", action="store_true", help="打开独立 SLRT-CSL 连续 Gloss 面板")
    parser.add_argument("--slrt-config", type=Path, help="SLRT 配置 JSON（需 --slrt）")
    parser.add_argument("--slrt-pose-backend", choices=("hrnet", "hrnet-npz", "rtmpose-experimental"), help="SLRT 姿态前端（需 --slrt）")
    parser.add_argument("--keypoints", action="store_true", help="直接打开关键点采集与回放工作台，不加载识别模型")
    parser.add_argument("--continuous", action="store_true", help="输入与模型就绪后连续识别视频流；窗口结果会修正")
    parser.add_argument("--continuous-window-frames", type=int, help="连续识别窗口帧数（Uni-Sign 默认 48）")
    parser.add_argument("--continuous-stride-frames", type=int, help="连续识别步长帧数（Uni-Sign 默认 16）")
    parser.add_argument("--capture-fps", type=float, default=30.0, help="把窗口帧数换算为秒的标称帧率")
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
    if not args.slrt and (args.slrt_config is not None or args.slrt_pose_backend is not None):
        parser.error("--slrt-config/--slrt-pose-backend require --slrt")
    if args.slrt:
        if (args.unisign or args.unisign_model is not None or args.keypoints or args.model is not None
                or args.continuous or args.continuous_window_frames is not None or args.continuous_stride_frames is not None
                or args.catalog is not None or args.segment_config is not None):
            parser.error("--slrt uses its independent config and cannot be combined with other model/segment modes")
        from dataclasses import replace
        from .slrt.assets import SLRTConfig
        from .slrt.desktop import launch
        try:
            config = SLRTConfig.from_file(args.slrt_config) if args.slrt_config else SLRTConfig()
            config = replace(config, device=args.device, device_id=args.device_id,
                             cpu_threads=args.intra_op_threads,
                             pose_backend=args.slrt_pose_backend or config.pose_backend)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return launch(config, video=args.video, camera=args.camera or 0)
    unisign_options = (args.unisign_model is not None or args.continuous_window_frames is not None
                       or args.continuous_stride_frames is not None)
    if not args.unisign and unisign_options:
        parser.error("--unisign-model/--continuous-window-frames/--continuous-stride-frames require --unisign")
    if args.unisign and args.model is not None:
        parser.error("--unisign cannot be combined with --model")
    if args.keypoints:
        if args.model is not None or args.catalog is not None or args.segment_config is not None or args.continuous or args.unisign:
            parser.error("关键点工作台不使用识别模型、字幕词典或自动分段配置")
        if args.camera is not None and not 0 <= args.camera <= 32:
            parser.error("关键点工作台的摄像头设备编号须在 0 到 32 之间")
        import sys
        from PySide6.QtWidgets import QApplication
        from .keypoint_desktop import KeypointWorkbench
        qt = QApplication.instance() or QApplication([sys.argv[0]])
        app = KeypointWorkbench()
        if args.video:
            app.open_video(args.video)
        elif args.camera is not None:
            app.camera_index.setValue(args.camera)
            app.open_camera()
        app.show()
        return qt.exec()
    try:
        options = RuntimeOptions(device=args.device, device_id=args.device_id,
                                 intra_op_threads=args.intra_op_threads, gpu_mem_limit_mb=args.gpu_mem_limit_mb,
                                 cuda_dll_dirs=tuple(str(path.resolve()) for path in args.cuda_dll_dir),
                                 cpu_conv_subnormals=args.cpu_conv_subnormals)
    except ValueError as error:
        parser.error(str(error))
    if not 0 < args.capture_fps <= 240:
        parser.error("--capture-fps must be in (0, 240]")
    unisign_path = None
    if args.unisign:
        unisign_path = (args.unisign_model or default_unisign_model()).resolve()
        if not (unisign_path / "manifest.json").is_file():
            parser.error(f"Uni-Sign bundle is missing: {unisign_path}")
    continuous_config = None
    if args.unisign or args.continuous_window_frames is not None or args.continuous_stride_frames is not None:
        from .continuous import ContinuousConfig
        window_frames = args.continuous_window_frames or 48
        stride_frames = args.continuous_stride_frames or 16
        if not 2 <= window_frames <= 600 or not 1 <= stride_frames <= window_frames:
            parser.error("continuous window/stride must satisfy 2 <= window <= 600 and 1 <= stride <= window")
        window_seconds = window_frames / args.capture_fps
        stride_seconds = stride_frames / args.capture_fps
        if window_seconds > 8.0:
            parser.error("continuous window must not exceed the 8-second clip contract")
        try:
            continuous_config = ContinuousConfig(
                minimum_seconds=window_seconds,
                # Frame-count windows must not require N/F seconds from N
                # timestamps, whose span is only (N-1)/F. Keep the 8-second
                # clip bound while readiness uses actual received frames.
                window_seconds=8.0,
                minimum_frames=window_frames,
                update_interval_seconds=stride_seconds,
                max_result_age_seconds=max(3.0, 2 * window_seconds + 1.0),
                max_observation_gap_seconds=.5,
                max_frames=window_frames,
                max_width=640,
            )
        except ValueError as error:
            parser.error(str(error))
    import sys
    from PySide6.QtWidgets import QApplication
    from .desktop import SignLanguageApp

    model = unisign_path if args.unisign else args.model if args.model is not None else default_model()
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
                          segment_config=segment_config, unisign_model=unisign_path,
                          continuous=args.continuous, continuous_config=continuous_config,
                          defer_startup_input=bool(args.unisign and (args.video is not None or args.camera is not None)))
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
