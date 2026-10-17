"""Inspect or extract keypoint recordings without loading a recognition model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Sequence

from .keypoint_recording import (
    KeypointRecorder, KeypointRecording, RecordingLimitError, RecordingLimits,
)
from .keypoints import KeypointExtractor
from .sources import VideoSource


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="validate a complete .slkp.jsonl recording")
    inspect.add_argument("path", type=Path)
    extract = commands.add_parser("extract", help="extract all video frames into a keypoint recording")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--label", help="optional draft collection label, not a recognized sign")
    extract.add_argument("--language", help="optional draft sign-language annotation")
    extract.add_argument("--performer", help="optional draft performer identifier")
    for command in (inspect, extract):
        command.add_argument("--max-frames", type=_positive_int, default=100_000,
                             help="hard frame limit; exceeding it rejects the whole recording")
        command.add_argument("--max-bytes", type=_positive_int, default=256 * 1024 * 1024,
                             help="hard file-size limit in bytes")
    return parser


def _extract(args: argparse.Namespace) -> dict:
    recorder = None
    extractor = None
    source = VideoSource(args.video)
    try:
        source.open()
        extractor = KeypointExtractor(time_domain="media_seconds")
        annotations = {"review_status": "draft"}
        for name in ("label", "language", "performer"):
            value = getattr(args, name)
            if value is not None:
                annotations[name] = value
        metadata = {"extractor": extractor.metadata(), "source_kind": "video",
                    "source_path": str(args.video.resolve()),
                    "source_timing_mode": source.timestamp_mode,
                    "annotations": annotations}
        recorder = KeypointRecorder(args.output, metadata=metadata,
                                    max_frames=args.max_frames, max_bytes=args.max_bytes)
        while True:
            packet = source.read()
            if packet is None:
                break
            if recorder.count >= args.max_frames:
                raise RecordingLimitError("recording exceeds the frame limit")
            frame = extractor.process(packet, received_monotonic=time.perf_counter())
            recorder.append(frame)
        path = recorder.finish(reason="eof")
        summary = KeypointRecording(path, limits=RecordingLimits(
            max_frames=args.max_frames, max_bytes=args.max_bytes)).summary()
        if recorder.cleanup_error is not None:
            summary["cleanup_warning"] = recorder.cleanup_error
        return summary
    except BaseException as failure:
        if recorder is not None:
            try:
                recorder.abort()
            except Exception as cleanup:
                failure.keypoint_cleanup_error = recorder.cleanup_error or str(cleanup)
                if hasattr(failure, "add_note"):
                    failure.add_note(f"Recording cleanup also failed: {failure.keypoint_cleanup_error}")
                raise failure from cleanup
        raise
    finally:
        try:
            if extractor is not None:
                extractor.close()
        finally:
            source.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "inspect":
            result = KeypointRecording(args.path, limits=RecordingLimits(
                max_frames=args.max_frames, max_bytes=args.max_bytes)).summary()
        else:
            result = _extract(args)
    except (Exception, KeyboardInterrupt) as exc:
        error = {"error": str(exc) or type(exc).__name__, "error_type": type(exc).__name__}
        if getattr(exc, "keypoint_cleanup_error", None) is not None:
            error["cleanup_error"] = exc.keypoint_cleanup_error
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
