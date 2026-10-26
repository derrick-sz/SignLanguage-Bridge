"""E01 step 2: probe the 50 raw CSL-Daily pilot videos with ffprobe.

Writes ``experiments/p1/e01/video_probe.csv`` with measured container/stream
facts only.  The pilot CSV ``frames`` column is the *annotation* sentence length
(``csl-daily.test`` ``num_frames``), not the raw recording length; both are
written so the difference is explicit.

Run:
    $env:PYTHONPATH='E:\\githubproject\\SignLanguage-Bridge'
    .\\.conda-env\\python.exe -X utf8 experiments\\p1\\e01\\tools\\build_video_probe.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.p1 import common  # noqa: E402

PROBE = shutil.which("ffprobe") or r"E:\ffmpeg\ffmpeg-2026-04-16-git-5abc240a27-full_build\bin\ffprobe.exe"

COLUMNS = [
    "sample_id", "signer_id", "cohort", "raw_video_path", "raw_present",
    "ffprobe_path", "probe_method", "status", "frames_counted", "frames_stream_nb_frames",
    "frames_from_nb_frames", "counted_equals_declared", "width", "height", "pix_fmt",
    "codec_name", "r_frame_rate", "avg_frame_rate", "fps", "fps_source",
    "duration_s_format", "duration_from_counted_frames_s", "duration_delta_s",
    "bytes", "sha256", "pilot_annotation_frames", "annotation_equals_raw_frames",
    "note",
]


def ffprobe(path: Path) -> dict:
    command = [
        PROBE, "-v", "error", "-count_frames",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,"
        "nb_frames,nb_read_frames,duration,time_base,color_space,color_range",
        "-show_entries", "format=duration,size,format_name,bit_rate",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=300)
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed ({completed.returncode}): {completed.stderr.strip()[:400]}")
    return json.loads(completed.stdout)


def fraction(text: str | None) -> float | None:
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    den_value = float(den)
    if den_value == 0:
        return None
    return float(num) / den_value


def main() -> int:
    rows = common.read_csv_rows(common.DATASETS / "pilot_50.csv")
    output_rows = []
    for position, row in enumerate(rows, 1):
        sample = row["sample_id"].strip()
        raw = Path(row["raw_video_path"])
        record = {
            "sample_id": sample,
            "signer_id": row["signer_id"],
            "cohort": row["cohort"],
            "raw_video_path": str(raw),
            "raw_present": raw.is_file(),
            "ffprobe_path": PROBE,
            "probe_method": "ffprobe -count_frames (decoded frame count)",
            "pilot_annotation_frames": int(row["frames"]),
        }
        if not raw.is_file():
            record.update({"status": "missing_raw_video", "note": "raw video not on disk"})
            output_rows.append(record)
            continue
        try:
            info = ffprobe(raw)
        except Exception as error:  # noqa: BLE001 - recorded per sample
            record.update({"status": "probe_error", "note": f"{type(error).__name__}: {error}"[:400]})
            output_rows.append(record)
            continue
        streams = [item for item in info.get("streams", []) if item.get("codec_type") == "video"]
        if not streams:
            record.update({"status": "no_video_stream"})
            output_rows.append(record)
            continue
        stream = streams[0]
        fmt = info.get("format", {})
        counted = stream.get("nb_read_frames")
        declared = stream.get("nb_frames")
        counted_int = int(counted) if counted not in (None, "N/A") else None
        declared_int = int(declared) if declared not in (None, "N/A") else None
        r_fps = fraction(stream.get("r_frame_rate"))
        avg_fps = fraction(stream.get("avg_frame_rate"))
        fps = avg_fps or r_fps
        fps_source = ("ffprobe avg_frame_rate" if avg_fps else "ffprobe r_frame_rate")
        duration_format = float(fmt["duration"]) if fmt.get("duration") not in (None, "N/A") else None
        duration_from_frames = (counted_int / fps) if (counted_int and fps) else None
        annotation = int(row["frames"])
        record.update({
            "status": "ok",
            "frames_counted": counted_int,
            "frames_stream_nb_frames": declared_int,
            "frames_from_nb_frames": declared_int,
            "counted_equals_declared": (counted_int == declared_int) if counted_int and declared_int else None,
            "width": stream.get("width"),
            "height": stream.get("height"),
            "pix_fmt": stream.get("pix_fmt"),
            "codec_name": stream.get("codec_name"),
            "r_frame_rate": stream.get("r_frame_rate"),
            "avg_frame_rate": stream.get("avg_frame_rate"),
            "fps": round(fps, 6) if fps else None,
            "fps_source": fps_source,
            "duration_s_format": round(duration_format, 6) if duration_format else None,
            "duration_from_counted_frames_s": round(duration_from_frames, 6) if duration_from_frames else None,
            "duration_delta_s": round(duration_format - duration_from_frames, 6)
            if (duration_format and duration_from_frames) else None,
            "bytes": raw.stat().st_size,
            "sha256": common.sha256_file(raw),
            "annotation_equals_raw_frames": annotation == counted_int,
            "note": ("raw recording is longer than the annotated sentence clip; "
                     "pilot 'frames' is the annotation length, not the raw frame count"),
        })
        output_rows.append(record)
        print(json.dumps({"event": "probed", "done": position, "sample": sample,
                          "frames": counted_int, "annotation": annotation}, ensure_ascii=False),
              flush=True)

    path = common.P1 / "e01" / "video_probe.csv"
    written = common.write_csv_rows(path, COLUMNS, output_rows)
    print(json.dumps({"event": "written", "path": str(path), "rows": written}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
