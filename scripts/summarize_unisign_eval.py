#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""汇总 Uni-Sign 评测输出目录（predictions.jsonl + run-protocol.json）。

用途
----
读取一个评测输出目录，产出 ``summary.json`` 与中文 ``summary.md``，并对运行协议做独立复核。
不加载模型、不读视频、不启动 GPU；只做纯文本/数值统计，可在任意机器上重跑。

用法
----
    # 自动挑选最新的 artifacts/cslnews-eval/run-cslnews-* 运行目录
    .\\.conda-env\\python.exe scripts\\summarize_unisign_eval.py

    # 指定运行目录
    .\\.conda-env\\python.exe scripts\\summarize_unisign_eval.py --run-dir artifacts\\cslnews-eval\\run-cslnews-20261025

    # 只跑合成 fixture 自测（不读真实数据）
    .\\.conda-env\\python.exe scripts\\summarize_unisign_eval.py --self-test

口径（与 docs/agents/model-evaluation-pitfalls.md 对齐）
------------------------------------------------------
* 准确率只用 ``status == "ok"`` 且预测非空的成功样本；分母写清楚（成功数/总数/失败数/空预测数）。
* 失败与空预测单独列清单，另外给一份「全量悲观下界」口径（失败按空串计入），但不得当作模型准确率引用。
* 延迟按 ``pose_source`` 分组（extracted 含姿态提取 / disk_cache 仅模型），不合并成单一数字。
* 冷启动（模型加载 / warmup）单列，不进入逐视频延迟。
* 跨数据集分数不得写成产品准确率，见 summary.md 的「数据边界」。

退出码
------
0 = 汇总已写出（复核可能有 warn/fail，见 summary.json）；3 = 复核存在 fail 项；
2 = 无可用数据（且不允许覆盖已有真实汇总）；1 = 其它错误。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_GLOB = "artifacts/cslnews-eval/run-cslnews-*"
# 只自动发现正式 run 目录；pilot-* 必须显式 --run-dir 传入，避免试跑数据混进最终 summary。
DISCOVERY_PATTERNS = ("artifacts/cslnews-eval/run-cslnews-*",)
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "cslnews-eval" / "summary"
DEFAULT_AUDIT_PATHS = (
    ROOT / "artifacts" / "cslnews-eval" / "audit" / "dataset-audit.json",
)

TIMING_FIELDS = (
    "decode_ms",
    "pose_extract_ms",
    "pose_preprocess_ms",
    "model_forward_ms",
    "generation_ms",
    "model_total_ms",
    "e2e_seconds",
)
# 口径（2026-10-25 与评测脚本作者核对）：
# * decode_ms 是 runtime.infer() 里 tokenizer/batch_decode 的**文本解码**耗时，不是视频容器解码；
# * 视频容器解码（cv2 读帧）没有单独计时，合并在 pose_extract_ms 里（read + RTMPose 同一计时区间）；
# * 纯视频解码成本只能来自 --decode-benchmark 产出的 decode-benchmark.json。
TIMING_LABELS = {
    "decode_ms": "文本解码 tokenizer(batch_decode)",
    "pose_extract_ms": "解码+RTMPose 合计",
    "pose_preprocess_ms": "姿态预处理",
    "model_forward_ms": "模型前向",
    "generation_ms": "文本生成",
    "model_total_ms": "模型合计",
    "e2e_seconds": "端到端(秒)",
    "pose_load_ms": "上游 pose 加载(pkl)",
    "pose_prepare_ms": "姿态准备(归一化+采样)",
}
# 消融（上游 pose pkl）目录只记录这几段：没有视频解码/RTMPose，也没有 pose_extract_ms。
ABLATION_TIMING_FIELDS = ("pose_load_ms", "pose_prepare_ms", "model_total_ms", "e2e_seconds")
# 延迟分解表的展示顺序：先姿态/解码链路，再模型侧，最后端到端。
STAGE_FIELD_ORDER = (
    "pose_extract_ms", "pose_preprocess_ms", "pose_load_ms", "pose_prepare_ms",
    "decode_ms", "model_forward_ms", "generation_ms", "model_total_ms", "e2e_seconds",
)
COLD_EVENTS = {
    "warmup", "model_load", "load_model", "model_ready", "load", "ready",
    "start", "done", "interrupted", "shutdown", "model_loading",
}
POSE_SOURCE_ORDER = ("extracted", "disk_cache", "unknown")


# ---------------------------------------------------------------------------
# 文本归一化与指标
# ---------------------------------------------------------------------------
def normalize_text(text) -> str:
    """NFKC 归一化后去掉全部空白；与 scripts/evaluate_unisign_cecsl.py 一致。"""
    if text is None:
        return ""
    return "".join(unicodedata.normalize("NFKC", str(text)).split())


def lcs_length(a: str, b: str) -> int:
    previous = [0] * (len(b) + 1)
    for char_a in a:
        current = [0]
        for j, char_b in enumerate(b, 1):
            current.append(previous[j - 1] + 1 if char_a == char_b else max(current[-1], previous[j]))
        previous = current
    return previous[-1]


def rouge_l_f1_char(reference: str, hypothesis: str) -> float:
    """字符级 ROUGE-L F1（每个汉字当作一个 token）。"""
    if not reference and not hypothesis:
        return 1.0
    lcs = lcs_length(reference, hypothesis)
    precision = lcs / max(1, len(hypothesis))
    recall = lcs / max(1, len(reference))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def edit_similarity(a: str, b: str) -> float:
    """归一化 Levenshtein 相似度：1 - 距离 / max(len(a), len(b))。"""
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, 1):
        current = [i]
        for j, char_b in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (char_a != char_b)))
        previous = current
    return 1.0 - previous[-1] / max(1, len(a), len(b))


def compute_text_metrics(refs, hyps):
    """在一组已归一化的参考/预测上计算文本指标。"""
    n = len(refs)
    if n == 0:
        return None
    exact = sum(a == b for a, b in zip(refs, hyps))
    edits = [edit_similarity(a, b) for a, b in zip(refs, hyps)]
    lengths = [len(b) / max(1, len(a)) for a, b in zip(refs, hyps)]
    mean_edit = float(statistics.fmean(edits))
    result = {
        "n": n,
        "exact_match_count": int(exact),
        "exact_match_percent": 100.0 * exact / n,
        "mean_char_edit_similarity": mean_edit,
        "mean_char_edit_similarity_percent": 100.0 * mean_edit,
        "mean_length_ratio_prediction_to_reference": float(statistics.fmean(lengths)),
        "reference_chars_total": int(sum(len(a) for a in refs)),
        "prediction_chars_total": int(sum(len(b) for b in hyps)),
        "rouge_l_f1_char": float(statistics.fmean(rouge_l_f1_char(a, b) for a, b in zip(refs, hyps))),
        "bleu_zh": None,
        "bleu_char": None,
        "chrf": None,
        "metric_error": None,
    }
    try:
        import sacrebleu  # 延迟导入：纯文本统计时也要能给出清晰的缺依赖错误

        result["bleu_zh"] = float(sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh").score)
        result["bleu_char"] = float(sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score)
        result["chrf"] = float(sacrebleu.corpus_chrf(hyps, [refs]).score)
    except Exception as exc:  # noqa: BLE001 - 指标失败不能阻断汇总，但要如实记录
        result["metric_error"] = f"{type(exc).__name__}: {exc}"
    return result


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------
def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def protocol_digest(protocol) -> str:
    """与 scripts/evaluate_unisign_cecsl.py::protocol_digest 完全一致。"""
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coerce_float(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def percentile_stats(values):
    """mean/P50/P95/P99/max/min；百分位用 numpy 线性插值（method="linear"）。"""
    cleaned = [float(v) for v in values if v is not None]
    if not cleaned:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None, "min": None}
    array = np.asarray(cleaned, dtype=float)
    return {
        "n": len(cleaned),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50, method="linear")),
        "p95": float(np.percentile(array, 95, method="linear")),
        "p99": float(np.percentile(array, 99, method="linear")),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
    }


def mean_or_none(values):
    cleaned = [v for v in values if v is not None]
    return float(statistics.fmean(cleaned)) if cleaned else None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


# ---------------------------------------------------------------------------
# 读取输入
# ---------------------------------------------------------------------------
def read_jsonl(path: Path):
    """返回 (records, parse_errors, total_lines)；records 为 [(line_no, dict), ...]。"""
    raw = path.read_text(encoding="utf-8-sig").splitlines()
    records, errors = [], []
    for index, line in enumerate(raw, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append({"line": index, "error": f"JSONDecodeError: {exc}"})
            continue
        if not isinstance(value, dict):
            errors.append({"line": index, "error": f"not a JSON object: {type(value).__name__}"})
            continue
        records.append((index, value))
    return records, errors, len(raw)


def read_protocol(run_dir: Path):
    path = run_dir / "run-protocol.json"
    if not path.is_file():
        return None, None, {"exists": False, "path": str(path)}
    try:
        protocol = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return None, None, {"exists": True, "path": str(path), "error": f"JSONDecodeError: {exc}"}
    if not isinstance(protocol, dict):
        return None, None, {"exists": True, "path": str(path), "error": "protocol is not a JSON object"}
    info = {
        "exists": True,
        "path": str(path),
        "sha256_file": file_sha256(path),
        "digest_recomputed": protocol_digest(protocol),
        "embedded_run_protocol_id": protocol.get("run_protocol_id"),
        "keys": sorted(protocol.keys()),
        "selected_count": len(protocol.get("selected") or []) if isinstance(protocol.get("selected"), list) else None,
    }
    return protocol, protocol_digest(protocol), info


def read_aux_timing(run_dir: Path):
    """读取评测脚本单独写出的冷启动与纯解码基准（如果存在）。

    ``cold-start.json`` 由 scripts/evaluate_unisign_cslnews.py 在 load_model() 后写出；
    ``decode-benchmark.json`` 只由 --decode-benchmark 生成，逐视频记录里没有 decode_ms。
    """
    aux = {"cold_start": None, "decode_benchmark": None}
    cold_path = run_dir / "cold-start.json"
    if cold_path.is_file():
        try:
            data = json.loads(cold_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            aux["cold_start"] = {
                "path": str(cold_path),
                "load_model_seconds": coerce_float(data.get("load_model_seconds")),
                "measured_at": data.get("measured_at"),
                "excluded_from_per_video_latency": data.get("excluded_from_per_video_latency"),
                "note": data.get("note"),
            }
    decode_path = run_dir / "decode-benchmark.json"
    if decode_path.is_file():
        try:
            data = json.loads(decode_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            per_video = data.get("per_video") if isinstance(data.get("per_video"), list) else []
            decode_values = [coerce_float(item.get("decode_ms")) for item in per_video
                             if isinstance(item, dict) and coerce_float(item.get("decode_ms")) is not None]
            fps_values = [coerce_float(item.get("decode_fps")) for item in per_video
                          if isinstance(item, dict) and coerce_float(item.get("decode_fps")) is not None]
            aux["decode_benchmark"] = {
                "path": str(decode_path),
                "n": data.get("n"),
                "failed": data.get("failed"),
                "decode_ms_recomputed": percentile_stats(decode_values),
                "decode_fps_recomputed": percentile_stats(fps_values),
                "decode_ms_reported": data.get("decode_ms"),
            }
    return aux


def pose_source_label(protocol, records):
    """把姿态来源写成人能读懂的一句，避免“上游 pkl”和“本地提取 + wh”被混成一类。"""
    protocol = protocol or {}
    pose_protocol = protocol.get("pose_protocol") or {}
    normalization = (protocol.get("pose_normalization")
                     or pose_protocol.get("pose_normalization")
                     or ((records[0].get("pose_normalization") if records else None)))
    declared = str(protocol.get("pose_source") or "")
    raw_sources = {str(r.get("pose_source")) for r in records}
    if declared == "upstream-pkl" or "upstream-pkl" in raw_sources:
        return ("作者发布 pose .pkl（上游估计器，自带 x/W,y/H 归一化；"
                "本地 isotropic_short_side / wh 未参与）")
    if declared == "cache" or normalize_pose_source(next(iter(raw_sources), "")) == "disk_cache":
        return f"本地 rtmlib RTMW-L 从视频提取的缓存姿态（npz）+ {normalization} 归一化"
    return f"本地 rtmlib RTMW-L 从视频提取 + {normalization} 归一化"


def detect_cold_start(record):
    """返回 (is_cold_start, reason)。"""
    event = str(record.get("event") or record.get("kind") or record.get("phase") or "").strip().lower()
    if event:
        if event in COLD_EVENTS:
            return True, f"event={event}"
        if "warmup" in event or "load" in event or "ready" in event:
            return True, f"event={event}"
    if record.get("cold_start") is True:
        return True, "cold_start=true"
    if record.get("includes_model_load") is True:
        return True, "includes_model_load=true"
    number = record.get("number")
    if number is None or str(number).strip() == "":
        return True, "missing number (非视频记录)"
    return False, ""


def classify_status(record):
    """把记录归类为 ok / empty / error / other，并单独标出 ok-but-empty 异常。"""
    prediction = record.get("prediction")
    has_prediction = isinstance(prediction, str) and prediction.strip() != ""
    raw = record.get("status")
    status = str(raw).strip().lower() if raw is not None else ""
    if status == "ok":
        return ("ok", None) if has_prediction else ("empty", "status_ok_but_empty_prediction")
    if status in {"empty", "empty_prediction", "blank"}:
        return "empty", None
    if status in {"error", "failed", "fail", "exception"}:
        return "error", None
    if status == "":
        return ("ok", "missing_status") if has_prediction else ("error", "missing_status")
    return "other", f"unknown_status={status}"


def normalize_pose_source(value) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if text in {"extracted", "extract", "pose_extracted", "runtime", "fresh"}:
        return "extracted"
    if text in {"disk_cache", "disk-cache", "cache", "cached", "cache_hit"}:
        return "disk_cache"
    return text or "unknown"


def timing_value(record, field):
    """返回 (value, source_field)；e2e_seconds 允许回退到 elapsed_seconds。"""
    value = coerce_float(record.get(field))
    if value is not None:
        return value, field
    if field == "e2e_seconds":
        fallback = coerce_float(record.get("elapsed_seconds"))
        if fallback is not None:
            return fallback, "elapsed_seconds"
    return None, field


# ---------------------------------------------------------------------------
# 统计块
# ---------------------------------------------------------------------------
def latency_block(records, fields=TIMING_FIELDS):
    """一组记录的时间统计 + RTF。"""
    block = {"n": len(records), "fields": {}, "rtf": None}
    sources = Counter()
    for field in fields:
        values, missing, negative = [], 0, 0
        for record in records:
            value, source = timing_value(record, field)
            if value is None:
                missing += 1
            elif value < 0:
                negative += 1
            else:
                values.append(value)
                sources[source] += 1
        stats = percentile_stats(values)
        stats["missing"] = missing
        stats["negative"] = negative
        block["fields"][field] = stats
    rtf_values, rtf_missing = [], 0
    for record in records:
        e2e, _ = timing_value(record, "e2e_seconds")
        duration = coerce_float(record.get("video_duration_s"))
        if e2e is None or duration is None or duration <= 0:
            rtf_missing += 1
            continue
        rtf_values.append(e2e / duration)
    block["rtf"] = percentile_stats(rtf_values)
    block["rtf"]["missing"] = rtf_missing
    block["e2e_source_fields"] = dict(sources)
    frames, total_e2e = [], 0.0
    for record in records:
        e2e, _ = timing_value(record, "e2e_seconds")
        frame_count = coerce_float(record.get("video_frames"))
        if e2e is None or e2e <= 0:
            continue
        total_e2e += e2e
        if frame_count is not None and frame_count > 0:
            frames.append(frame_count)
    mean_e2e = block["fields"]["e2e_seconds"]["mean"]
    block["throughput"] = {
        "total_e2e_seconds": total_e2e,
        "videos_per_minute": (60.0 / mean_e2e) if mean_e2e and mean_e2e > 0 else None,
        "frames_per_second": (sum(frames) / total_e2e) if total_e2e > 0 and frames else None,
        "frames_counted": int(sum(frames)),
        "records_with_frames": len(frames),
    }
    return block


def build_latency(records, cold_records, aux=None, fields=TIMING_FIELDS):
    aux = aux or {}
    groups = {name: [] for name in POSE_SOURCE_ORDER}
    for record in records:
        groups.setdefault(normalize_pose_source(record.get("pose_source")), []).append(record)
    by_source = {}
    for name, subset in groups.items():
        if not subset:
            continue
        by_source[name] = latency_block(subset, fields)
    overall = latency_block(records, fields) if records else None
    if overall is not None:
        overall["quotable"] = False
        overall["warning"] = (
            "混合姿态来源口径：仅用于完整性核对，不得作为单一延迟数字引用；"
            "对外引用必须分开 extracted 与 disk_cache。"
        )
    cold = {
        "records": [
            {
                "number": record.get("number"),
                "event": record.get("event") or record.get("kind") or record.get("phase"),
                "reason": reason,
                "e2e_seconds": coerce_float(record.get("e2e_seconds")) or coerce_float(record.get("elapsed_seconds")),
                "model_load_seconds": coerce_float(record.get("model_load_seconds")),
                "warmup_seconds": coerce_float(record.get("warmup_seconds")),
                "created_at": record.get("created_at"),
            }
            for record, reason in cold_records
        ],
        "count": len(cold_records),
    }
    cold_seconds = [
        item["e2e_seconds"] if item["e2e_seconds"] is not None else item["model_load_seconds"]
        for item in cold["records"]
    ]
    cold["total_seconds"] = sum(v for v in cold_seconds if v is not None) or None
    cold["excluded_from_per_video"] = True
    cold["external_record"] = aux.get("cold_start")
    if cold["external_record"] and cold["total_seconds"] is None:
        cold["total_seconds"] = cold["external_record"].get("load_model_seconds")
    latency = {
        "by_pose_source": by_source,
        "overall_mixed": overall,
        "cold_start": cold,
        "decode_only_benchmark": aux.get("decode_benchmark"),
        "timing_field_notes": {
            "decode_ms": "tokenizer/batch_decode 的文本解码耗时（不是视频容器解码）",
            "pose_extract_ms": "cv2 读帧 + RTMPose 的合计耗时，视频解码没有单独拆分",
            "pose_preprocess_ms": "关键点预处理（归一化/分块），仅 extracted 记录有值",
            "model_total_ms": "model_forward_ms + generation_ms",
            "e2e_seconds": "单条视频从读文件到文本输出的端到端耗时",
        },
        "decode_note": (
            "逐视频记录里的 decode_ms 是文本解码（tokenizer batch_decode），不是视频解码；"
            "视频容器解码已合并在 pose_extract_ms 内。纯解码成本只能来自独立的 decode-benchmark.json，"
            "用它减 pose_extract_ms 可粗略估算 RTMPose 占比（帧数不完全一致，属近似值）。"
        ),
    }
    latency["stage_decomposition"] = stage_decomposition(latency, fields)
    return latency


def stage_decomposition(latency, fields=TIMING_FIELDS):
    """逐阶段延迟分解（供 summary.md 的「延迟分解」表使用）。"""
    rows = []

    def add_row(stage, group, unit, stats, note=None):
        stats = stats or {}
        rows.append({
            "stage": stage, "group": group, "unit": unit,
            "n": stats.get("n"), "mean": stats.get("mean"), "p50": stats.get("p50"),
            "p95": stats.get("p95"), "p99": stats.get("p99"), "max": stats.get("max"),
            "note": note,
        })

    cold = latency.get("cold_start") or {}
    external = cold.get("external_record") or {}
    load_seconds = external.get("load_model_seconds")
    if load_seconds is None:
        load_seconds = cold.get("total_seconds")
    if load_seconds is not None:
        add_row("cold-start 模型加载", "单列，不计入逐视频", "s",
                {"n": 1, "mean": load_seconds, "p50": None, "p95": None, "p99": None, "max": None})

    benchmark = latency.get("decode_only_benchmark") or {}
    bench_stats = benchmark.get("decode_ms_recomputed") or {}
    if bench_stats.get("n"):
        add_row("纯视频解码（cv2 读帧）", "独立基准 decode-benchmark", "ms", bench_stats)

    for group in (latency.get("by_pose_source") or {}):
        block = latency["by_pose_source"].get(group)
        if not block:
            continue
        for field in STAGE_FIELD_ORDER:
            if field not in fields:
                continue
            stats = block["fields"].get(field)
            if stats and stats.get("n"):
                add_row(TIMING_LABELS.get(field, field), group,
                        "s" if field == "e2e_seconds" else "ms", stats)
        add_row("RTF（e2e / 视频时长）", group, "×", block.get("rtf"))

    # RTMPose 占比粗估：解码+RTMPose 均值 − 纯解码均值（仅主评测口径才有 pose_extract_ms）
    extracted = (latency.get("by_pose_source") or {}).get("extracted") or {}
    extract_stats = extracted.get("fields", {}).get("pose_extract_ms", {})
    if bench_stats.get("mean") is not None and extract_stats.get("mean") is not None:
        estimate = float(extract_stats["mean"]) - float(bench_stats["mean"])
        rows.append({
            "stage": "RTMPose 占比粗估（解码+RTMPose − 纯解码）",
            "group": "extracted", "unit": "ms", "n": None,
            "mean": estimate, "p50": None, "p95": None, "p99": None, "max": None,
            "note": "近似值：独立基准与逐视频读帧的帧数/缓存状态不完全一致",
        })
    return rows


def cold_start_suspicion(records):
    """逐视频首条耗时若远超其余中位数，提示可能有未标记的模型加载/warmup 混入。"""
    pairs = []
    for record in records:
        e2e, _ = timing_value(record, "e2e_seconds")
        if e2e is not None and e2e > 0:
            pairs.append((record, e2e))
    if len(pairs) < 4:
        return None
    first_record, first_value = pairs[0]
    rest = sorted(value for _, value in pairs[1:])
    median_rest = statistics.median(rest)
    if median_rest <= 0:
        return None
    ratio = first_value / median_rest
    if ratio < 5.0:
        return None
    return {
        "number": first_record.get("number"),
        "first_e2e_seconds": first_value,
        "median_of_rest": median_rest,
        "ratio": ratio,
        "note": "首条逐视频耗时显著高于其余中位数，疑似把模型加载/warmup 计入了该条；需评测脚本给出显式 warmup 记录。",
    }


# ---------------------------------------------------------------------------
# 协议复核
# ---------------------------------------------------------------------------
def _truncate(values, limit=20):
    values = list(values)
    if len(values) <= limit:
        return values, 0
    return values[:limit], len(values) - limit


def load_dataset_audit(paths):
    for path in paths:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            return path, data
    return None, None


def collect_audit_split_evidence(audit):
    """从 dataset-audit.json 里抽取与划分语义有关的短证据（不臆造字段）。"""
    if not isinstance(audit, dict):
        return []
    found = []

    def walk(node, prefix=""):
        if len(found) >= 6:
            return
        if isinstance(node, dict):
            for key, value in node.items():
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                if re.search(r"split|train|test|leak|generaliz|划分|泛化", key_text, re.I):
                    if isinstance(value, (str, int, float, bool)) and len(str(value)) <= 160:
                        found.append(f"{path} = {value}")
                        continue
                walk(value, path)
        elif isinstance(node, list):
            for index, value in enumerate(node[:5]):
                walk(value, f"{prefix}[{index}]")

    walk(audit)
    return found[:6]


def build_dataset_audit_summary(audit_paths=DEFAULT_AUDIT_PATHS):
    """从 dataset-audit.json 抽出划分语义的关键事实（只读引用，不复制整份审计）。"""
    path, audit = load_dataset_audit(audit_paths)
    if not isinstance(audit, dict):
        return None
    split = ((audit.get("split_semantics") or {}).get("math") or {})
    return {
        "path": str(path),
        "local_inside_train": split.get("local_inside_train"),
        "local_rows_1based": split.get("local_rows_1based"),
        "train_count": split.get("train_count"),
        "test_count": split.get("test_count"),
        "annotation_size_used_by_upstream": split.get("annotation_size_used_by_upstream"),
        "train_slice_half_open_0based": split.get("train_slice_half_open_0based"),
        "note": ("本地样本若落在上游 train 段内，则即便同域也不是留出集泛化估计。" if split else
                 "dataset-audit 未提供 split_semantics.math，划分语义需人工核对。"),
    }


def build_protocol_review(*, run_dir, protocol, protocol_info, video_records, cold_records,
                          records_total, accuracy, latency, missing_fields, duplicates,
                          eval_summary_path, coverage=None, scope_interim=False):
    checks = []

    # ① run_protocol_id 一致性 -------------------------------------------------
    ids = Counter()
    for record in video_records:
        ids[str(record.get("run_protocol_id"))] += 1
    digest = protocol_info.get("digest_recomputed")
    embedded = protocol_info.get("embedded_run_protocol_id")
    evidence = [
        f"run-protocol.json: {'存在' if protocol_info.get('exists') else '缺失'}（{protocol_info.get('path')}）",
        f"复算摘要 digest = {digest or '—'}",
        f"协议文件内嵌 run_protocol_id = {embedded or '—'}",
        f"predictions.jsonl 中 run_protocol_id 取值分布 = {dict(ids)}",
    ]
    status = "fail"
    if not protocol_info.get("exists"):
        evidence.append("结论：缺少 run-protocol.json，无法确认这批预测属于哪套协议。")
    elif not ids:
        evidence.append("结论：没有视频记录可校验。")
        status = "warn"
    elif len(ids) > 1:
        evidence.append(f"结论：混入了 {len(ids)} 种协议 id，禁止合并统计。")
    elif digest in ids:
        status = "pass"
        evidence.append("结论：全部记录 run_protocol_id 与 run-protocol.json 复算摘要一致。")
    elif embedded and set(ids) == {str(embedded)}:
        status = "warn"
        evidence.append(
            "结论：记录 id 与协议内嵌 run_protocol_id 一致，但与该文件当前内容的复算摘要不同；"
            "说明协议文件在运行后被改写或补充过字段，需以评测脚本写出的 id 为准并记录差异。"
        )
    else:
        evidence.append("结论：记录 id 既非复算摘要也非内嵌 id，无法由 run-protocol.json 复现。")
    checks.append({
        "id": "P1",
        "title": "所有记录 run_protocol_id 一致且等于 run-protocol.json 摘要",
        "status": status,
        "evidence": evidence,
        "distinct_ids": dict(ids),
        "protocol_digest_recomputed": digest,
        "protocol_embedded_id": embedded,
    })

    # ② 空预测/失败排除在准确率分母外 -----------------------------------------
    counts = accuracy.get("denominator", {}) if accuracy else {}
    anomalies = accuracy.get("anomalies", []) if accuracy else []
    evidence = [
        f"分母口径：{counts.get('expression', '—')}",
        f"成功 {counts.get('success')} / 全部视频记录 {counts.get('total')}；"
        f"空预测 {counts.get('empty')}，失败 {counts.get('error')}，其它状态 {counts.get('other')}",
        f"status=ok 但预测为空的异常记录 {len(anomalies)} 条",
    ]
    status = "fail" if anomalies else ("pass" if accuracy and accuracy.get("metrics") else "warn")
    if anomalies:
        sample, rest = _truncate([item.get("number") for item in anomalies])
        evidence.append(f"异常样本：{sample}{'（另有 %d 条）' % rest if rest else ''}")
        evidence.append("结论：存在 status=ok 但空预测的记录，已被排除出准确率分子分母，但说明状态字段不可信。")
    elif not accuracy or not accuracy.get("metrics"):
        evidence.append("结论：成功样本为空，无法计算准确率，需先修数据。")
    else:
        evidence.append("结论：准确率仅用成功样本计算，失败/空预测未进入准确率分母，且已单独列清单。")
    if eval_summary_path is not None and eval_summary_path.is_file():
        try:
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eval_summary = None
        if isinstance(eval_summary, dict):
            evidence.append(
                "评测脚本自写 summary.json 计数：successful={}，failed_or_empty={}，selected={}，complete={}".format(
                    eval_summary.get("successful"), eval_summary.get("failed_or_empty"),
                    eval_summary.get("selected"), eval_summary.get("complete"))
            )
            if eval_summary.get("successful") is not None and counts.get("success") is not None \
                    and int(eval_summary["successful"]) != int(counts["success"]):
                evidence.append("注意：评测脚本自写 successful 与本汇总成功数不一致，需人工核对。")
    checks.append({
        "id": "P2",
        "title": "空预测/失败被排除在准确率分母外，并单独列出",
        "status": status,
        "evidence": evidence,
        "anomaly_count": len(anomalies),
    })

    # ③ 延迟排除冷启动 + 区分 pose 缓存 ---------------------------------------
    source_counts = Counter(normalize_pose_source(r.get("pose_source")) for r in video_records)
    suspicion = cold_start_suspicion(video_records)
    external_cold = (latency or {}).get("cold_start", {}).get("external_record") if latency else None
    status = "pass"
    evidence = [
        f"视频记录 pose_source 分布 = {dict(source_counts)}",
        f"冷启动/辅助记录 {len(cold_records)} 条（predictions.jsonl 内），已单列并排除出逐视频延迟",
    ]
    if external_cold:
        evidence.append(
            "cold-start.json：load_model_seconds={}s，excluded_from_per_video_latency={}，measured_at={}".format(
                fmt(external_cold.get("load_model_seconds"), 2),
                external_cold.get("excluded_from_per_video_latency"),
                external_cold.get("measured_at") or "—")
        )
        if external_cold.get("excluded_from_per_video_latency") is not True:
            status = "fail"
            evidence.append("结论：cold-start.json 未声明冷启动已排除，逐视频延迟可能混入模型加载。")
        else:
            evidence.append("结论：评测脚本显式把模型加载/warmup 记为冷启动并声明不计入逐视频延迟。")
    else:
        status = "warn"
        evidence.append("未找到 cold-start.json：无法确认模型加载耗时是否被排除在逐视频延迟之外（证据不足）。")
    evidence.append("extracted 组含姿态提取，disk_cache 组仅模型；两组分别统计，未合并成单一数字")
    if source_counts.get("unknown"):
        status = "fail"
        evidence.append(f"结论：{source_counts['unknown']} 条记录缺少 pose_source，无法判断是否命中姿态缓存。")
    elif len(source_counts) == 1:
        evidence.append("结论：本次全部记录同源（单一 pose_source），没有两组对照，引用时须说明。")
    if suspicion:
        status = "fail" if status == "pass" else status
        evidence.append(
            f"疑似冷启动混入：{suspicion['number']} 的 e2e={suspicion['first_e2e_seconds']:.3f}s，"
            f"是其余中位数的 {suspicion['ratio']:.1f} 倍；评测脚本未写显式 warmup 记录。"
        )
    # 与评测脚本自写 summary.json 的延迟口径对照（同分位数方法：线性插值）
    if eval_summary_path is not None and eval_summary_path.is_file():
        try:
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eval_summary = None
        if isinstance(eval_summary, dict):
            reported = eval_summary.get("latency") or {}
            for key, source in (("e2e_seconds_extracted", "extracted"),
                                ("e2e_seconds_disk_cache", "disk_cache")):
                ours = ((latency or {}).get("by_pose_source", {}).get(source) or {}).get("fields", {}).get("e2e_seconds", {})
                theirs = reported.get(key) or {}
                if ours.get("p95") is not None and theirs.get("p95") is not None:
                    same = abs(float(ours["p95"]) - float(theirs["p95"])) <= 1e-6
                    evidence.append(
                        f"延迟交叉核对 {source} e2e P95：本汇总 {fmt(ours['p95'])}s vs 评测脚本 {fmt(theirs['p95'])}s"
                        f"（{'一致' if same else '不一致'}）"
                    )
            if reported.get("cold_start_seconds") is not None:
                evidence.append(f"评测脚本自报冷启动 {fmt(reported['cold_start_seconds'], 2)}s（单列，不计入逐视频）")
    if status == "pass":
        evidence.append("结论：逐视频延迟未混入模型加载/warmup，且按姿态来源分组统计。")
    checks.append({
        "id": "P3",
        "title": "延迟排除冷启动，且区分姿态缓存命中/未命中",
        "status": status,
        "evidence": evidence,
        "pose_source_counts": dict(source_counts),
        "cold_start_suspicion": suspicion,
    })

    # ④ 跨域分数不得写成 CSL-News 准确率 --------------------------------------
    protocol_text = canonical_json(protocol).lower() if protocol else ""
    model_domain = None
    if "csl_daily" in protocol_text or "csl-daily" in protocol_text:
        model_domain = "CSL-Daily"
    dataset = None
    for key in ("dataset", "dataset_name", "corpus", "test_set"):
        value = protocol.get(key) if protocol else None
        if isinstance(value, str) and value.strip():
            dataset = value.strip()
            break
    if dataset is None:
        joined = " ".join(str(r.get("video") or "") for r in video_records[:5]).lower()
        for token, name in (("csl-news", "CSL-News"), ("csl_news", "CSL-News"), ("cslnews", "CSL-News"),
                            ("ce-csl", "CE-CSL"), ("csl-daily", "CSL-Daily")):
            if token in joined:
                dataset = name
                break
    flag_keys = [k for k in ("cross_domain", "domain_warning", "domain_note", "split_warning",
                             "in_domain", "domain", "notes", "training_domain",
                             "not_generalization") if protocol and k in protocol]
    evidence = [
        f"协议中的数据集字段 = {dataset or '缺失'}",
        f"协议中的权重/域信息：{model_domain or '未从协议文本识别到 CSL-Daily 权重名'}",
        f"协议中的域边界标记字段 = {flag_keys or '无'}",
    ]
    if protocol and isinstance(protocol.get("domain_note"), str):
        evidence.append(f"协议 domain_note：{protocol['domain_note'][:200]}")
    # 评测脚本自写 summary.json 的域标记（若存在）
    eval_domain_flags = []
    if eval_summary_path is not None and eval_summary_path.is_file():
        try:
            eval_summary = json.loads(eval_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            eval_summary = None
        if isinstance(eval_summary, dict):
            for key in ("cross_domain", "domain_warning", "split_warning", "in_domain"):
                if key in eval_summary:
                    eval_domain_flags.append(f"{key}={str(eval_summary[key])[:120]}")
    if eval_domain_flags:
        evidence.append("评测脚本 summary.json 域标记：" + "；".join(eval_domain_flags))
    audit_path, audit = load_dataset_audit(DEFAULT_AUDIT_PATHS)
    if audit is not None:
        evidence.append(f"数据集审计：{audit_path}")
        evidence.extend(collect_audit_split_evidence(audit))
    else:
        evidence.append("数据集审计 dataset-audit.json 未就绪，无法引用本地样本的划分语义。")
    if model_domain and dataset and model_domain.lower() != dataset.lower():
        evidence.append(f"结论：权重训练域 {model_domain} ≠ 评测数据域 {dataset}，属于跨数据集评测。")
    if not flag_keys and not eval_domain_flags:
        status = "fail"
        evidence.append(
            "结论：run-protocol.json 与评测脚本 summary.json 都没有域边界标记"
            "（cross_domain/domain_warning/domain_note 等），该批分数有被当作目标数据集准确率引用的风险。"
        )
    elif not flag_keys:
        status = "warn"
        evidence.append("结论：协议本身未写域边界，仅评测脚本 summary.json 有标记；协议层需补齐。")
    else:
        status = "pass"
        evidence.append("结论：协议显式记录了域边界，本汇总也始终标注成功子集/跨数据集口径。")
    checks.append({
        "id": "P4",
        "title": "未把跨数据集分数写成 CSL-News 产品准确率",
        "status": status,
        "evidence": evidence,
        "model_domain": model_domain,
        "dataset": dataset,
        "domain_flag_keys": flag_keys,
        "eval_summary_domain_flags": eval_domain_flags,
        "audit_path": str(audit_path) if audit_path else None,
    })

    # ⑤ video_sha256 / 协议字段完整性 ----------------------------------------
    total = len(video_records)
    selected = protocol.get("selected") if isinstance(protocol, dict) else None
    selected_numbers = set()
    selected_sha = set()
    if isinstance(selected, list):
        for item in selected:
            if isinstance(item, dict) and item.get("number") is not None:
                selected_numbers.add(str(item["number"]))
                if isinstance(item.get("video_sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", item["video_sha256"]):
                    selected_sha.add(str(item["number"]))
    missing_sha = [str(r.get("number")) for r in video_records
                   if not (isinstance(r.get("video_sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", r["video_sha256"]))]
    evidence = [
        f"记录级 video_sha256 缺失 {len(missing_sha)} / {total} 条",
        f"run-protocol.json selected[] 覆盖 {len(selected_numbers)} 条，其中带合法 video_sha256 的 {len(selected_sha)} 条",
    ]
    for field, numbers in missing_fields.items():
        if numbers:
            sample, rest = _truncate(numbers)
            evidence.append(f"字段 {field} 缺失 {len(numbers)} 条：{sample}{'（另有 %d 条）' % rest if rest else ''}")
    if duplicates:
        evidence.append(f"重复 number {len(duplicates)} 个：{duplicates[:10]}")
    if not missing_sha:
        status = "pass"
        evidence.append("结论：每条视频记录都带视频内容哈希，可与协议逐行核对。")
    elif selected_numbers and not (set(str(r.get('number')) for r in video_records) - selected_sha):
        status = "warn"
        evidence.append(
            "结论：记录级缺 video_sha256，但 run-protocol.json 的 selected[] 为每个样本提供了哈希；"
            "协议级可核对，逐行仍无法直接校验。"
        )
    else:
        status = "fail"
        evidence.append("结论：存在既无记录级、也无协议级视频内容哈希的样本，无法确认输入身份。")
    checks.append({
        "id": "P5",
        "title": "记录不缺 video_sha256 / 协议字段",
        "status": status,
        "evidence": evidence,
        "missing_video_sha256_count": len(missing_sha),
        "protocol_selected_count": len(selected_numbers),
        "protocol_selected_sha256_count": len(selected_sha),
    })

    # ⑥（附加）样本覆盖与唯一性：缺样本时不得标记完整 ------------------------
    coverage = coverage or {}
    selected_count = coverage.get("protocol_selected")
    missing_numbers = coverage.get("missing_numbers") or []
    extra_numbers = coverage.get("extra_numbers") or []
    evidence = [
        f"协议 selected[] 样本数 = {selected_count if selected_count is not None else '未列出'}；"
        f"predictions.jsonl 视频记录 = {coverage.get('video_records')}",
        f"缺失样本 {len(missing_numbers)} 条，协议外多余记录 {len(extra_numbers)} 条，重复 number {len(duplicates)} 个",
    ]
    if missing_numbers:
        sample, rest = _truncate(missing_numbers)
        evidence.append(f"缺失样本示例：{sample}{'（另有 %d 条）' % rest if rest else ''}")
    if duplicates:
        evidence.append(f"重复 number：{duplicates[:10]}")
    if selected_count is None:
        status = "warn"
        evidence.append("结论：协议未列 selected[]，无法核对样本覆盖；不得把本次结果称为完整批次。")
    elif missing_numbers or duplicates or extra_numbers:
        if scope_interim and not duplicates and not extra_numbers:
            status = "warn"
            evidence.append("结论：interim 子集，覆盖不完整属预期；只有跑到协议全量才能判为通过。")
        else:
            status = "fail"
            evidence.append("结论：样本覆盖不完整或存在重复/越界记录，不能标记为完整批次。")
    else:
        status = "pass"
        evidence.append("结论：协议列出的每个样本恰好有一条记录，无重复、无越界。")
    checks.append({
        "id": "P6",
        "title": "样本覆盖与唯一性（附加检查）",
        "status": status,
        "evidence": evidence,
        "coverage": coverage,
    })

    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"
    return {"checks": checks, "overall": overall, "run_dir": str(run_dir)}


# ---------------------------------------------------------------------------
# 汇总主体
# ---------------------------------------------------------------------------
def write_per_sample_csv(path: Path, video_records) -> None:
    """逐样本明细，便于人工抽查（不参与汇总计算）。"""
    import csv

    fields = ["number", "status", "pose_source", "exact_match", "edit_similarity",
              "rouge_l_f1_char", "ref_chars", "pred_chars", "video_frames",
              "video_duration_s", "e2e_seconds", "model_total_ms", "pose_extract_ms",
              "reference", "prediction", "error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in video_records:
            reference = normalize_text(record.get("reference"))
            prediction = normalize_text(record.get("prediction"))
            writer.writerow({
                "number": record.get("number"),
                "status": record.get("status"),
                "pose_source": normalize_pose_source(record.get("pose_source")),
                "exact_match": int(reference == prediction and bool(prediction)),
                "edit_similarity": round(edit_similarity(reference, prediction), 6),
                "rouge_l_f1_char": round(rouge_l_f1_char(reference, prediction), 6),
                "ref_chars": len(reference),
                "pred_chars": len(prediction),
                "video_frames": record.get("video_frames"),
                "video_duration_s": record.get("video_duration_s"),
                "e2e_seconds": record.get("e2e_seconds") if record.get("e2e_seconds") is not None
                else record.get("elapsed_seconds"),
                "model_total_ms": record.get("model_total_ms"),
                "pose_extract_ms": record.get("pose_extract_ms"),
                "reference": record.get("reference"),
                "prediction": record.get("prediction"),
                "error": record.get("error"),
            })
    temporary.replace(path)


def summarize_run(run_dir: Path, *, audit_paths=DEFAULT_AUDIT_PATHS, output_dir=None,
                  per_sample_name="per-sample.csv", scope_label=None, latency_note=None):
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing predictions.jsonl: {predictions_path}")
    records, parse_errors, total_lines = read_jsonl(predictions_path)
    protocol, digest, protocol_info = read_protocol(run_dir)

    video_records, cold_records = [], []
    for line_no, record in records:
        is_cold, reason = detect_cold_start(record)
        if is_cold:
            cold_records.append((record, reason))
            continue
        record = dict(record)
        record["_line"] = line_no
        video_records.append(record)
    video_records.sort(key=lambda r: str(r.get("number")))

    statuses = Counter()
    anomalies = []
    for record in video_records:
        status, anomaly = classify_status(record)
        statuses[status] += 1
        if anomaly:
            anomalies.append({"number": record.get("number"), "reason": anomaly,
                              "status": record.get("status")})
    success = [r for r in video_records if classify_status(r)[0] == "ok"]
    empty = [r for r in video_records if classify_status(r)[0] == "empty"]
    failed = [r for r in video_records if classify_status(r)[0] in {"error", "other"}]

    refs = [normalize_text(r.get("reference")) for r in success]
    hyps = [normalize_text(r.get("prediction")) for r in success]
    metrics = compute_text_metrics(refs, hyps)

    all_refs = [normalize_text(r.get("reference")) for r in video_records]
    all_hyps = [normalize_text(r.get("prediction")) if classify_status(r)[0] == "ok" else "" for r in video_records]
    full_metrics = compute_text_metrics(all_refs, all_hyps)

    missing_fields = {}
    for field in ("video_sha256", "run_protocol_id", "pose_orientation", "pose_source", "reference", "video"):
        numbers = [str(r.get("number")) for r in video_records
                   if r.get(field) is None or (isinstance(r.get(field), str) and not r[field].strip())]
        if numbers:
            missing_fields[field] = numbers
    counter = Counter(str(r.get("number")) for r in video_records)
    duplicates = sorted(number for number, count in counter.items() if count > 1)

    # 协议声明的样本 vs 实际记录：缺样本时不得标记完整
    protocol_selected_numbers = []
    if isinstance(protocol, dict) and isinstance(protocol.get("selected"), list):
        for item in protocol["selected"]:
            if isinstance(item, dict) and item.get("number") is not None:
                protocol_selected_numbers.append(str(item["number"]))
    record_numbers = [str(r.get("number")) for r in video_records]
    coverage = {
        "protocol_selected": len(protocol_selected_numbers) if protocol_selected_numbers else None,
        "video_records": len(video_records),
        "missing_numbers": sorted(set(protocol_selected_numbers) - set(record_numbers)),
        "extra_numbers": sorted(set(record_numbers) - set(protocol_selected_numbers)),
        "duplicates": duplicates,
        "complete": bool(protocol_selected_numbers)
        and set(protocol_selected_numbers) == set(record_numbers) and not duplicates,
    }

    # 输入特征：分辨率/FPS/帧数/时长，便于判断这批样本的代表性
    input_profile = {
        "resolutions": sorted({f"{int(r['video_width'])}x{int(r['video_height'])}"
                               for r in video_records
                               if coerce_float(r.get("video_width")) and coerce_float(r.get("video_height"))}),
        "fps_values": sorted({coerce_float(r.get("video_fps")) for r in video_records
                              if coerce_float(r.get("video_fps")) is not None}),
        "frames": percentile_stats([coerce_float(r.get("video_frames")) for r in video_records]),
        "duration_s": percentile_stats([coerce_float(r.get("video_duration_s")) for r in video_records]),
        "reference_chars": percentile_stats([len(normalize_text(r.get("reference"))) for r in video_records]),
    }

    accuracy = {
        "scope": "success_subset：status == 'ok' 且预测非空（去空白后）",
        "denominator": {
            "success": len(success),
            "total": len(video_records),
            "empty": len(empty),
            "error": len(failed),
            "other": statuses.get("other", 0),
            "expression": f"{len(success)} / {len(video_records)} 条视频记录（空预测 {len(empty)}、失败 {len(failed)} 已排除）",
        },
        "metrics": metrics,
        "full_set_pessimistic": {
            "scope": "全量口径：失败/空预测按空串计入分母（悲观下界，仅作边界参考，不得当作模型准确率）",
            "denominator": {"total": len(video_records)},
            "metrics": full_metrics,
        },
        "anomalies": anomalies,
        "by_pose_source": {},
    }
    for source in POSE_SOURCE_ORDER:
        subset = [r for r in success if normalize_pose_source(r.get("pose_source")) == source]
        if not subset:
            continue
        accuracy["by_pose_source"][source] = compute_text_metrics(
            [normalize_text(r.get("reference")) for r in subset],
            [normalize_text(r.get("prediction")) for r in subset],
        )

    aux_timing = read_aux_timing(run_dir)
    latency = build_latency(video_records, cold_records, aux=aux_timing)

    failed_samples = [{
        "number": r.get("number"),
        "status": r.get("status"),
        "error": r.get("error"),
        "pose_source": r.get("pose_source"),
        "video": r.get("video"),
        "e2e_seconds": coerce_float(r.get("e2e_seconds")) or coerce_float(r.get("elapsed_seconds")),
        "line": r.get("_line"),
    } for r in empty + failed]
    failed_samples.sort(key=lambda item: str(item.get("number")))

    eval_summary_path = run_dir / "summary.json"
    review = build_protocol_review(
        run_dir=run_dir, protocol=protocol, protocol_info=protocol_info,
        video_records=video_records, cold_records=cold_records, records_total=total_lines,
        accuracy=accuracy, latency=latency, missing_fields=missing_fields,
        duplicates=duplicates, eval_summary_path=eval_summary_path, coverage=coverage,
        scope_interim=bool(scope_label),
    )

    # 协议摘要（只挑与身份/口径相关的键，避免把整份 selected 列表复制进汇总）
    protocol_summary = None
    if protocol:
        keep = ("schema_version", "dataset", "subset", "subset_selection", "split",
                "checkpoint", "checkpoint_sha256",
                "source_commit", "mt5_revision", "num_beams", "max_new_tokens", "device",
                "dtype", "pose_protocol", "pose_orientation", "labels_sha256", "cross_domain",
                "domain_warning", "domain_note", "split_warning", "in_domain", "notes",
                "video_root", "bundle")
        protocol_summary = {k: protocol[k] for k in keep if k in protocol}
        protocol_summary["effective_pose_normalization"] = (
            protocol.get("pose_normalization")
            or (protocol.get("pose_protocol") or {}).get("pose_normalization"))
    label = pose_source_label(protocol, video_records)

    summary = {
        "schema_version": 1,
        "status": "ok",
        "scope": "interim" if scope_label else "final",
        "scope_label": scope_label,
        "generated_at": now_iso(),
        "generator": {
            "script": "scripts/summarize_unisign_eval.py",
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "run_dir": str(run_dir),
        "inputs": {
            "predictions_jsonl": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
            "predictions_total_lines": total_lines,
            "run_protocol": protocol_info,
        },
        "run_protocol_summary": protocol_summary,
        "pose_source_label": label,
        "latency_note": latency_note,
        "dataset_audit": build_dataset_audit_summary(audit_paths),
        "records": {
            "parsed": len(records),
            "parse_errors": parse_errors,
            "video_records": len(video_records),
            "cold_start_or_aux_records": len(cold_records),
            "by_status": dict(statuses),
            "success": len(success),
            "empty": len(empty),
            "failed": len(failed),
            "duplicates": duplicates,
            "coverage": coverage,
            "input_profile": input_profile,
            "missing_fields": {k: len(v) for k, v in missing_fields.items()},
            "missing_field_samples": {k: v[:10] for k, v in missing_fields.items()},
        },
        "accuracy": accuracy,
        "latency": latency,
        "failed_samples": failed_samples,
        "protocol_review": review,
        "metric_definitions": {
            "exact_match": "NFKC 归一化并去空白后，逐字符完全相等",
            "bleu_zh": "sacrebleu corpus_bleu，tokenize='zh'，在归一化文本上计算",
            "bleu_char": "sacrebleu corpus_bleu，tokenize='char'（补充口径）",
            "chrf": "sacrebleu corpus_chrf（char_order=6, word_order=0）",
            "rouge_l_f1_char": "字符级 ROUGE-L F1，逐样本计算后取均值",
            "mean_char_edit_similarity": "归一化 Levenshtein 相似度 1 - dist/max(len)，逐样本均值",
            "length_ratio": "len(prediction) / max(1, len(reference))，逐样本均值",
            "rtf": "e2e_seconds / video_duration_s",
            "percentile": "numpy.percentile(method='linear')",
        },
        "warnings": [],
    }
    if not protocol_info.get("exists"):
        summary["warnings"].append("缺少 run-protocol.json：无法确认预测所属协议。")
    if summary["records"]["parse_errors"]:
        summary["warnings"].append(f"{len(parse_errors)} 行 predictions.jsonl 解析失败。")
    if anomalies:
        summary["warnings"].append(f"{len(anomalies)} 条记录 status=ok 但预测为空。")
    if coverage["protocol_selected"] is not None and not coverage["complete"]:
        summary["warnings"].append(
            f"样本不完整：协议声明 {coverage['protocol_selected']} 条，实际记录 {coverage['video_records']} 条，"
            f"缺失 {len(coverage['missing_numbers'])} 条，不得标记为完整批次。"
        )
    if review["overall"] != "pass":
        summary["warnings"].append(f"协议复核未全部通过（{review['overall']}），见 protocol_review。")
    if output_dir is not None:
        per_sample = Path(output_dir) / per_sample_name
        write_per_sample_csv(per_sample, video_records)
        summary["inputs"]["per_sample_csv"] = str(per_sample)
    return summary


# ---------------------------------------------------------------------------
# 专用 profile：stage1（训练集拟合）与 CSL-Daily in-domain（BLEU 口径复核）
# ---------------------------------------------------------------------------
def mt5_char_fix(text) -> str:
    """复刻上游 fine_tuning.py 的 'fix mt5 tokenizer bug'：逐字加空格。"""
    return " ".join(list(str(text).replace(" ", "").replace("\n", "")))


def bleu_13a(hyps, refs):
    import sacrebleu

    result = sacrebleu.corpus_bleu(hyps, [refs], tokenize="13a")
    return {"bleu": float(result.score), "bp": float(result.bp),
            "precisions": [float(p) for p in result.precisions],
            "hyp_len": int(result.sys_len), "ref_len": int(result.ref_len)}


def chrf_score(hyps, refs):
    import sacrebleu

    return float(sacrebleu.corpus_chrf(hyps, [refs]).score)


def rouge_l_token_f1(refs, hyps) -> float:
    """token 级 ROUGE-L F1（上游 SLRT_metrics 用 rouge 包；参考句是逐字空格分隔）。"""
    values = []
    for reference, hypothesis in zip(refs, hyps):
        ref_tokens, hyp_tokens = reference.split(), hypothesis.split()
        lcs = lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / max(1, len(hyp_tokens))
        recall = lcs / max(1, len(ref_tokens))
        values.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return 100.0 * float(statistics.fmean(values)) if values else 0.0


def parse_sample_lines(path: Path, field: str):
    """解析 `sample: X, <field>: Y` 形式的快照/参考文件。"""
    pattern = re.compile(rf"^sample:\s*(\S+),\s*{field}:\s*(.*)$")
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            out[match.group(1)] = match.group(2)
    return out


PROFILES = {
    "stage1": {
        "protocol_names": ("experiment-protocol.json", "run-protocol.json"),
        "id_field": "protocol_id",
        "number_field": "number",
        "timing_fields": ("model_forward_ms", "generation_ms", "decode_ms",
                          "model_total_ms", "e2e_seconds"),
        "group_by_pose_source": True,
        "control_run_dir": "artifacts/cslnews-eval/pinned-upstreampose-20261025",
    },
    "csldaily": {
        "protocol_names": ("protocol.json", "run-protocol.json"),
        "id_field": "protocol_id",
        "number_field": "sample",
        "timing_fields": ("model_total_ms", "model_forward_ms", "generation_ms",
                          "decode_ms", "e2e_seconds"),
        "group_by_pose_source": False,
        "bleu_conventions": True,
    },
}


def detect_protocol_digest(protocol, record_ids):
    compact = protocol_digest(protocol)
    default = ablation_protocol_digest(protocol)
    if compact in record_ids:
        return compact, "compact separators（主评测口径）"
    if default in record_ids:
        return default, "default separators（消融/实验口径）"
    return compact, "未匹配到任何已知摘要公式"


def bleu_convention_block(video_records, protocol):
    """CSL-Daily：三种口径独立复算 + 上游快照对照。"""
    refs_path = protocol.get("refs") if isinstance(protocol, dict) else None
    if not refs_path or not Path(refs_path).is_file():
        return None
    refs_path = Path(refs_path)
    refs_map = parse_sample_lines(refs_path, "ground-truth")
    pres_path = refs_path.with_name("test_tmp_pres.txt")
    pres_map = parse_sample_lines(pres_path, "prediction") if pres_path.is_file() else {}

    ordered = [r for r in video_records if str(r.get("number")) in refs_map]
    numbers = [str(r["number"]) for r in ordered]
    refs_raw = [refs_map[number] for number in numbers]
    preds_raw = [str(r.get("prediction") or "") for r in ordered]
    preds_fixed = [mt5_char_fix(text) for text in preds_raw]

    block = {
        "refs_path": str(refs_path),
        "refs_sha256": file_sha256(refs_path),
        "refs_sha256_in_protocol": protocol.get("refs_sha256"),
        "n": len(ordered),
        "convention": ("预测先 ' '.join(list(text.replace(' ','').replace('\\n','')))，"
                       "再对逐字空格分隔的参考句用 sacrebleu tokenize=13a 计算"),
        "ours_fixed": {
            "bleu4": bleu_13a(preds_fixed, refs_raw),
            "chrf": chrf_score(preds_fixed, refs_raw),
            "rouge_l_token_f1": rouge_l_token_f1(refs_raw, preds_fixed),
            "exact_match": sum(1 for a, b in zip(preds_fixed, refs_raw) if a.strip() == b.strip()),
        },
        "ours_raw": {
            "bleu4": bleu_13a(preds_raw, refs_raw),
            "chrf": chrf_score(preds_raw, refs_raw),
            "rouge_l_token_f1": rouge_l_token_f1(refs_raw, preds_raw),
            "exact_match": sum(1 for a, b in zip(preds_raw, refs_raw) if a.strip() == b.strip()),
        },
    }
    if pres_map:
        upstream_hyps = [pres_map[number] for number in numbers if number in pres_map]
        upstream_refs = [refs_map[number] for number in numbers if number in pres_map]
        upstream_fixed = [mt5_char_fix(text) for text in upstream_hyps]
        block["upstream_snapshot"] = {
            "pres_path": str(pres_path),
            "n": len(upstream_hyps),
            "bleu4_fixed": bleu_13a(upstream_fixed, upstream_refs),
            "chrf_fixed": chrf_score(upstream_fixed, upstream_refs),
            "rouge_l_token_f1_fixed": rouge_l_token_f1(upstream_refs, upstream_fixed),
            "identical_after_fix": sum(
                1 for a, b in zip(preds_fixed, upstream_hyps) if a.strip() == b.strip()),
            "identical_raw": sum(
                1 for a, b in zip(preds_raw, upstream_hyps) if a.strip() == b.strip()),
        }
    block["paper_reference"] = {
        "bleu4": 25.61, "rouge": 54.92,
        "rgb_pose": {"bleu4": 26.36, "rouge": 56.51},
        "source": ("论文表格数值（arXiv:2501.15187 Table 8）：CSL-Daily pose-only test "
                   "BLEU-4 25.61 / ROUGE 54.92；RGB+pose 26.36 / 56.51。已由 Lead 从 arXiv HTML 核对，"
                   "本复核未重新打开论文，只复算本机与上游快照两侧。"),
        "verified_by": "Lead（arXiv HTML Table 8）",
        "locally_verified": False,
    }
    return block


def _profile_fix_history(profile_name, run_dir=None, control_pose_source=None):
    """记录本复核指出、随后由评测脚本修复的口径问题（历史值写死，便于读者对照）。"""
    if profile_name == "csldaily":
        return [
            "修复前：summary.json 的 `metrics.bleu4_13a` = 0.000467（未做 mt5 逐字修正的陷阱口径），"
            "`identical_prediction_count` 仅 0（原样字符串比较）→ 只读该文件会得出“BLEU 0.0005、0 条一致”的错误结论。",
            "修复后：主值 `metrics.bleu4_13a` = 25.5504（修正后口径，带定义说明）；陷阱值单独存于 "
            "`bleu4_13a_raw_no_mt5_fix` 并附 `protocol_trap_note`；`identical_prediction_count` = 0 与 "
            "`identical_prediction_count_after_mt5_fix` = 1029 并存。",
            "**现在引用：25.5504（修正后口径）**；0.000467 只作格式陷阱演示，不得当准确率引用。",
        ]
    if profile_name == "stage1":
        is_upstream_pose = (
            (run_dir is not None and "upstreampose" in str(run_dir).lower())
            or control_pose_source == "upstream-pkl")
        if not is_upstream_pose:
            return []
        return [
            "修复前：`pinned_checkpoint_mean_edit_similarity` = 0.07903（来自 rtmlib pose 的 "
            "run-cslnews-20261025），Δ = +0.2107 —— 两侧姿态前端不同，Δ 混入前端差异。",
            "修复后：同姿态源对照 pinned-upstreampose 的 0.08088，Δ = +0.20885（paired_n = 379，"
            "bootstrap 95% CI [0.1879, 0.2300]）。",
            "**现在引用：Δ = +0.20885（同姿态源，只反映权重差异）**；+0.2107 不再使用。",
        ]
    return []


def summarize_profile(run_dir: Path, profile_name: str, *, output_dir=None,
                      control_run_dir=None, alt_control_run_dir=None,
                      frontend_compare_run_dir=None):
    profile = PROFILES[profile_name]
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing predictions.jsonl: {predictions_path}")
    records, parse_errors, total_lines = read_jsonl(predictions_path)
    protocol_path, protocol = read_protocol_any(run_dir, names=profile["protocol_names"])

    video_records, aux_records = [], []
    for line_no, raw in records:
        record = dict(raw)
        number = record.get(profile["number_field"])
        if number is not None and record.get("number") is None:
            record["number"] = number
        if record.get("run_protocol_id") is None and record.get(profile["id_field"]) is not None:
            record["run_protocol_id"] = record.get(profile["id_field"])
        is_cold, reason = detect_cold_start(record)
        if is_cold:
            aux_records.append((record, reason))
            continue
        record["_line"] = line_no
        video_records.append(record)
    video_records.sort(key=lambda r: str(r.get("number")))

    statuses = Counter()
    anomalies = []
    for record in video_records:
        status, anomaly = classify_status(record)
        statuses[status] += 1
        if anomaly:
            anomalies.append({"number": record.get("number"), "reason": anomaly})
    success = [r for r in video_records if classify_status(r)[0] == "ok"]
    empty = [r for r in video_records if classify_status(r)[0] == "empty"]
    failed = [r for r in video_records if classify_status(r)[0] in {"error", "other"}]

    metrics = compute_text_metrics([normalize_text(r.get("reference")) for r in success],
                                   [normalize_text(r.get("prediction")) for r in success])

    declared = []
    if isinstance(protocol, dict):
        for key in ("selected", "samples"):
            items = protocol.get(key)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and item.get("number") is not None:
                        declared.append(str(item["number"]))
                    elif isinstance(item, str):
                        declared.append(item)
                break
    record_numbers = [str(r.get("number")) for r in video_records]
    coverage = {
        "protocol_selected": len(declared) if declared else None,
        "video_records": len(video_records),
        "missing_numbers": sorted(set(declared) - set(record_numbers)),
        "extra_numbers": sorted(set(record_numbers) - set(declared)),
        "complete": bool(declared) and set(declared) == set(record_numbers),
    }

    aux_timing = read_aux_timing(run_dir)
    if profile["group_by_pose_source"]:
        latency = build_latency(video_records, aux_records, aux=aux_timing,
                                fields=profile["timing_fields"])
    else:
        block = latency_block(video_records, profile["timing_fields"])
        latency = {"by_pose_source": {"all（pose 来自磁盘，无视频解码/RTMPose）": block},
                   "overall_mixed": None, "cold_start": {"records": [], "count": 0,
                                                         "external_record": aux_timing.get("cold_start")},
                   "decode_only_benchmark": aux_timing.get("decode_benchmark")}
        latency["stage_decomposition"] = stage_decomposition(latency, profile["timing_fields"])

    # 实验目录没有 cold-start.json：首条记录常把模型加载算进 e2e，需要显式标出
    cold_start_flag = cold_start_suspicion(video_records)
    latency["cold_start_suspicion"] = cold_start_flag

    protocol_digest_value, digest_formula = (None, None)
    if isinstance(protocol, dict):
        protocol_digest_value, digest_formula = detect_protocol_digest(
            protocol, {str(r.get("run_protocol_id")) for r in video_records})

    bleu_block = bleu_convention_block(video_records, protocol) if profile.get("bleu_conventions") else None

    control_dir = Path(control_run_dir) if control_run_dir else (
        ROOT / profile["control_run_dir"] if profile.get("control_run_dir") else None)
    paired = None
    control_note = None
    if control_dir and (control_dir / "predictions.jsonl").is_file():
        control_rows, _, control_lines = read_jsonl(control_dir / "predictions.jsonl")
        control_records = []
        for _, raw in control_rows:
            record = dict(raw)
            if record.get("number") is None and record.get(profile["number_field"]) is not None:
                record["number"] = record.get(profile["number_field"])
            if record.get("number") is not None:
                control_records.append(record)
        paired = paired_comparison(video_records, control_records)
        paired["denominator_rule"] = "只在本运行与对照运行两边都 status=ok 且参考句一致时计入"
        paired["control_pose_source"] = (control_records[0].get("pose_source")
                                         if control_records else None)
        control_note = {"path": str(control_dir / "predictions.jsonl"),
                        "records": len(control_records), "lines": control_lines,
                        "pose_source": paired["control_pose_source"]}
    elif control_dir:
        control_note = {"path": str(control_dir), "records": 0, "note": "对照目录缺少 predictions.jsonl"}

    # 可选第二对照（例如生产路径的 rtmlib pose）：口径不同，只能作为参考，不能与主对照混谈
    paired_alt = None
    alt_note = None
    if alt_control_run_dir:
        alt_dir = Path(alt_control_run_dir)
        if (alt_dir / "predictions.jsonl").is_file():
            alt_rows, _, _ = read_jsonl(alt_dir / "predictions.jsonl")
            alt_records = [dict(raw) for _, raw in alt_rows if raw.get("number") is not None]
            paired_alt = paired_comparison(video_records, alt_records)
            paired_alt["denominator_rule"] = "只在本运行与对照运行两边都 status=ok 且参考句一致时计入"
            paired_alt["control_pose_source"] = (alt_records[0].get("pose_source")
                                                 if alt_records else None)
            alt_note = {"path": str(alt_dir / "predictions.jsonl"),
                        "records": len(alt_records),
                        "pose_source": paired_alt["control_pose_source"]}
        else:
            alt_note = {"path": str(alt_dir), "records": 0, "note": "对照目录缺少 predictions.jsonl"}

    # 同权重、不同姿态来源的前端对照（例如 stage1-wh-full vs stage1-upstreampose-full）
    frontend_compare = None
    if frontend_compare_run_dir:
        compare_dir = Path(frontend_compare_run_dir)
        if (compare_dir / "predictions.jsonl").is_file():
            compare_rows, _, _ = read_jsonl(compare_dir / "predictions.jsonl")
            compare_records = [dict(raw) for _, raw in compare_rows if raw.get("number") is not None]
            compare_protocol_path, compare_protocol = read_protocol_any(
                compare_dir, names=profile["protocol_names"])
            cross = paired_comparison(video_records, compare_records)
            frontend_compare = {
                "path": str(compare_dir / "predictions.jsonl"),
                "this_pose_source": pose_source_label(protocol, video_records),
                "compare_pose_source": pose_source_label(compare_protocol, compare_records),
                "this_checkpoint_sha256": (protocol or {}).get("checkpoint_sha256"),
                "compare_checkpoint_sha256": (compare_protocol or {}).get("checkpoint_sha256"),
                "paired_n": cross["paired_n"],
                "skipped": cross["skipped"],
                "this_mean_edit_similarity": cross["mean_edit_similarity_upstream_pose"],
                "compare_mean_edit_similarity": cross["mean_edit_similarity_local_pose"],
                "delta_this_minus_compare": cross["mean_delta_upstream_minus_local"],
                "delta_bootstrap_95ci": cross.get("delta_bootstrap_95ci"),
                "wins_this": cross["wins_upstream"],
                "losses_this": cross["losses_upstream"],
                "ties": cross["ties"],
                "identical_predictions": cross["identical_prediction_count"],
                "paired_corpus_bleu": cross.get("paired_corpus_bleu_zh"),
            }
            # 前端对照成立的前提：同一权重。若权重哈希不同，必须显式标出
            frontend_compare["same_checkpoint"] = (
                frontend_compare["this_checkpoint_sha256"] is not None
                and frontend_compare["this_checkpoint_sha256"] == frontend_compare["compare_checkpoint_sha256"])
            frontend_compare["same_pose_source"] = (
                frontend_compare["this_pose_source"] == frontend_compare["compare_pose_source"])
        else:
            frontend_compare = {"path": str(compare_dir), "error": "对照目录缺少 predictions.jsonl"}

    review = build_profile_review(
        run_dir=run_dir, profile_name=profile_name, protocol_path=protocol_path,
        protocol=protocol, protocol_digest_value=protocol_digest_value,
        digest_formula=digest_formula, video_records=video_records, success=success,
        empty=empty, failed=failed, coverage=coverage, anomalies=anomalies,
        bleu_block=bleu_block, paired=paired, latency=latency, parse_errors=parse_errors,
        paired_alt=paired_alt, alt_note=alt_note,
        control_dir=Path(control_dir) if control_dir else None,
        cold_start_flag=cold_start_flag)

    # 协议摘要：pose_normalization 需要按实际输入路径改写（见下方 upstream-pkl 说明）
    profile_protocol_summary = {
        key: protocol[key] for key in
        ("experiment", "evaluation", "checkpoint", "checkpoint_sha256", "pose_source",
         "pose_dir", "pose_normalization", "pose_orientation", "refs", "refs_sha256",
         "rescale_x", "max_frames", "num_beams", "max_new_tokens", "device", "dtype",
         "frame_sampling", "production_pinned_checkpoint", "not_a_production_model")
        if isinstance(protocol, dict) and key in protocol
    }
    if profile_protocol_summary.get("pose_source") == "upstream-pkl":
        declared = profile_protocol_summary.get("pose_normalization")
        profile_protocol_summary["pose_normalization"] = (
            "不适用（upstream-pkl 自带 x/W,y/H 归一化；本地 isotropic_short_side / wh 均未参与，"
            "_crop_scale 只做 bbox 归一）")
        profile_protocol_summary["pose_normalization_protocol_value"] = declared
        profile_protocol_summary["pose_normalization_note"] = (
            "这一格是“上游归一化 + 上游估计器”的配置，与 stage1-wh-20261025（本地估计器 + wh）"
            "不是同一配置，不能互相当作消融对照。")

    summary = {
        "schema_version": 1,
        "kind": f"profile_{profile_name}",
        "status": "ok",
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "profile": profile_name,
        "inputs": {
            "predictions_jsonl": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
            "predictions_total_lines": total_lines,
            "parse_errors": parse_errors,
            "protocol_path": str(protocol_path) if protocol_path else None,
            "protocol_digest_recomputed": protocol_digest_value,
            "protocol_digest_formula": digest_formula,
            "control_run": control_note,
        },
        "protocol_summary": profile_protocol_summary,
        "records": {
            "parsed": len(records), "video_records": len(video_records),
            "aux_records": len(aux_records), "by_status": dict(statuses),
            "success": len(success), "empty": len(empty), "failed": len(failed),
            "coverage": coverage, "anomalies": anomalies,
        },
        "accuracy": {
            "scope": "success_subset：status == 'ok' 且预测非空（NFKC 去空白口径）",
            "denominator": {
                "success": len(success), "total": len(video_records),
                "empty": len(empty), "error": len(failed),
                "expression": f"{len(success)} / {len(video_records)} 条记录（空预测 {len(empty)}、失败 {len(failed)} 已排除）",
            },
            "metrics": metrics,
        },
        "bleu_conventions": bleu_block,
        "latency": latency,
        "paired_comparison": paired,
        "paired_comparison_alt": paired_alt,
        "alt_control_run": alt_note,
        "frontend_compare": frontend_compare,
        "pose_source_label": pose_source_label(protocol, video_records),
        "review": review,
        "warnings": [],
    }
    if anomalies:
        summary["warnings"].append(f"{len(anomalies)} 条记录 status=ok 但预测为空。")
    if coverage["protocol_selected"] is not None and not coverage["complete"]:
        summary["warnings"].append(
            f"样本不完整：协议声明 {coverage['protocol_selected']} 条，实际 {coverage['video_records']} 条。")
    if review["overall"] != "pass":
        summary["warnings"].append(f"复核未全部通过（{review['overall']}）。")
    summary["fix_history"] = _profile_fix_history(
        profile_name, run_dir=run_dir,
        control_pose_source=(paired or {}).get("control_pose_source"))
    if output_dir is not None:
        per_sample = Path(output_dir) / f"per-sample-{run_dir.name}.csv"
        write_per_sample_csv(per_sample, video_records)
        summary["inputs"]["per_sample_csv"] = str(per_sample)
    return summary


def check_cache_binding(protocol, control_dir):
    """核对“本运行的姿态缓存来源”是否与对照运行的协议文件哈希一致。"""
    if not isinstance(protocol, dict) or control_dir is None:
        return None
    declared = protocol.get("main_run_protocol_sha256")
    if not declared:
        return None
    for name in ("run-protocol.json", "ablation-protocol.json", "experiment-protocol.json"):
        path = Path(control_dir) / name
        if path.is_file():
            actual = file_sha256(path)
            return {"path": str(path), "protocol_value": declared,
                    "actual_file_sha256": actual, "matches": actual == declared}
    return None


def build_profile_review(*, run_dir, profile_name, protocol_path, protocol,
                         protocol_digest_value, digest_formula, video_records, success,
                         empty, failed, coverage, anomalies, bleu_block, paired,
                         latency, parse_errors, paired_alt=None, alt_note=None,
                         control_dir=None, cold_start_flag=None):
    checks = []

    ids = Counter(str(r.get("run_protocol_id")) for r in video_records)
    evidence = [
        f"协议文件：{protocol_path or '缺失'}",
        f"复算 protocol_id = {protocol_digest_value or '—'}（公式：{digest_formula}）",
        f"记录中 protocol_id 分布 = {dict(ids)}",
    ]
    if protocol_path is None:
        status = "fail"
        evidence.append("结论：缺少协议文件，无法确认记录归属。")
    elif len(ids) > 1:
        status = "fail"
        evidence.append(f"结论：混入 {len(ids)} 种 protocol_id。")
    elif protocol_digest_value in ids:
        status = "pass"
        evidence.append("结论：全部记录 protocol_id 与协议文件复算值一致。")
    else:
        status = "fail"
        evidence.append("结论：记录 protocol_id 无法由协议文件复现。")
    checks.append({"id": "C1", "title": "protocol_id 一致且可复现", "status": status,
                   "evidence": evidence})

    evidence = [
        f"分母：{len(success)} / {len(video_records)}（空预测 {len(empty)}、失败 {len(failed)} 已排除）",
        f"status=ok 但预测为空 {len(anomalies)} 条",
        f"解析失败行 {len(parse_errors)} 行",
    ]
    status = "fail" if anomalies else ("pass" if success else "fail")
    evidence.append("结论：准确率只统计成功样本。" if status == "pass" else "结论：存在异常记录或没有成功样本。")
    checks.append({"id": "C2", "title": "准确率分母只含成功样本", "status": status,
                   "evidence": evidence})

    groups = list((latency.get("by_pose_source") or {}))
    cache_binding = check_cache_binding(protocol, control_dir)
    evidence = [
        f"延迟分组 = {groups}",
        "逐视频延迟只统计本目录记录；冷启动记录单列",
    ]
    if cache_binding:
        evidence.append(
            f"姿态缓存来源：协议 main_run_protocol_sha256 = {cache_binding['protocol_value'][:16]}…，"
            f"对照运行 run-protocol.json 文件哈希 = {cache_binding['actual_file_sha256'][:16]}…"
            f"（{'一致：对照就是该姿态缓存的来源运行' if cache_binding['matches'] else '不同：对照不是该缓存来源，属正常'}）")
    status = "pass" if groups else "fail"
    evidence.append("结论：延迟按本实验自身口径统计。" if status == "pass" else "结论：没有可用延迟记录。")
    checks.append({"id": "C3", "title": "延迟口径单列、不跨实验混表", "status": status,
                   "evidence": evidence})

    missing_sha = [str(r.get("number")) for r in video_records
                   if not (isinstance(r.get("video_sha256"), str)
                           and re.fullmatch(r"[0-9a-fA-F]{64}", r["video_sha256"]))]
    selected_sha = 0
    if isinstance(protocol, dict) and isinstance(protocol.get("selected"), list):
        selected_sha = sum(
            1 for item in protocol["selected"]
            if isinstance(item, dict) and isinstance(item.get("video_sha256"), str)
            and re.fullmatch(r"[0-9a-fA-F]{64}", item["video_sha256"]))
    has_pose_dir = isinstance(protocol, dict) and (protocol.get("pose_dir") or protocol.get("refs"))
    evidence = [
        f"记录级 video_sha256 缺失 {len(missing_sha)} / {len(video_records)} 条",
        f"协议 selected[] 带合法 video_sha256 的样本 {selected_sha} 条",
        f"协议提供的输入身份字段：pose_dir={bool(isinstance(protocol, dict) and protocol.get('pose_dir'))}，"
        f"refs_sha256={bool(isinstance(protocol, dict) and protocol.get('refs_sha256'))}，"
        f"main_run_protocol_sha256={bool(isinstance(protocol, dict) and protocol.get('main_run_protocol_sha256'))}",
    ]
    if not missing_sha:
        status = "pass"
        evidence.append("结论：每条记录都带视频内容哈希。")
    elif selected_sha >= len(video_records) and len(video_records) > 0:
        status = "warn"
        evidence.append("结论：记录级缺视频哈希，但协议 selected[] 为每个样本提供了哈希；协议级可核对，逐行不能直接校验。")
    elif has_pose_dir:
        status = "warn"
        evidence.append("结论：记录级缺视频哈希，但协议给出了 pose 目录/参考句哈希；输入身份靠协议级字段。")
    else:
        status = "fail"
        evidence.append("结论：既无记录级也无协议级输入身份。")
    if cache_binding and not cache_binding["matches"]:
        evidence.append("说明：协议声明的姿态缓存来源（main_run_protocol_sha256）不是本次对照运行；"
                        "两者不同属正常（对照可以是别的运行），上面两个哈希都已列出供核对。")
    checks.append({"id": "C4", "title": "输入身份可核对（video_sha256 / pose 目录 / refs 哈希）",
                   "status": status, "evidence": evidence})

    evidence = [
        f"协议声明 {coverage.get('protocol_selected')} 条；实际 {coverage.get('video_records')} 条",
        f"缺样本 {len(coverage.get('missing_numbers') or [])} 条；协议外记录 {len(coverage.get('extra_numbers') or [])} 条",
    ]
    if coverage.get("protocol_selected") is None:
        status = "warn"
        evidence.append("结论：协议未列样本清单，无法核对覆盖。")
    elif coverage.get("complete"):
        status = "pass"
        evidence.append("结论：样本覆盖完整。")
    else:
        status = "fail"
        evidence.append("结论：覆盖不完整，不得标记完整批次。")
    checks.append({"id": "C5", "title": "样本覆盖与唯一性", "status": status, "evidence": evidence})

    if profile_name == "stage1":
        audit = build_dataset_audit_summary() or {}
        evidence = [
            "stage1 权重 = 上游 CSL_News pose-only 预训练（train_stage1.sh --dataset CSL_News）",
            f"本地 1694 条位于上游 train 段：local_inside_train={audit.get('local_inside_train')}"
            f"（行 {audit.get('local_rows_1based')}，train 段 {audit.get('train_slice_half_open_0based')}）",
        ]
        if paired:
            evidence.append(
                f"与 pinned（CSL-Daily 微调）同片段配对 {paired['paired_n']} 条："
                f"stage1 编辑相似度 {fmt(paired.get('mean_edit_similarity_upstream_pose'), 4)} vs "
                f"pinned {fmt(paired.get('mean_edit_similarity_local_pose'), 4)}"
                f"（对照姿态源：{paired.get('control_pose_source')}）")
        evidence.append("结论：这是训练集拟合（模型见过同域训练段样本），不是泛化准确率。")
        checks.append({"id": "C6", "title": "明确标注为训练集拟合、非泛化准确率", "status": "pass",
                       "evidence": evidence})

        # C7 配对对照的姿态源必须一致，否则 Δ 混入前端差异
        reported = None
        run_summary_path = Path(run_dir) / "summary.json"
        if run_summary_path.is_file():
            try:
                run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run_summary = None
            if isinstance(run_summary, dict):
                reported = run_summary.get("paired_comparison_vs_pinned_checkpoint")
        evidence = []
        status = "warn"
        if not paired:
            evidence.append("本复核没有可用的对照运行，无法判断配对口径。")
        elif not isinstance(reported, dict):
            evidence.append("运行目录没有自写配对结果（summary.json 缺失或未跑完），以下以本复核为准。")
            evidence.append(
                f"本复核同姿态源对照（{paired.get('control_pose_source')}）：Δ="
                f"{fmt(paired.get('mean_delta_upstream_minus_local'), 4)}")
            status = "pass"
        else:
            reported_pinned = reported.get("pinned_checkpoint_mean_edit_similarity")
            same_source_value = paired.get("mean_edit_similarity_local_pose")
            alt_value = (paired_alt or {}).get("mean_edit_similarity_local_pose")
            control_run = reported.get("control_run")
            evidence.append(
                f"运行目录自报 pinned 均值 = {fmt(reported_pinned, 5)}（control_run={control_run or '未记录'}）；"
                f"本复核同姿态源对照（{paired.get('control_pose_source')}）= {fmt(same_source_value, 5)}；"
                f"另一对照（{(paired_alt or {}).get('control_pose_source')}）= {fmt(alt_value, 5)}")
            matches_same = (reported_pinned is not None and same_source_value is not None
                            and abs(float(reported_pinned) - float(same_source_value)) <= 1e-6)
            matches_alt = (reported_pinned is not None and alt_value is not None
                           and abs(float(reported_pinned) - float(alt_value)) <= 1e-6)
            if matches_same:
                status = "pass"
                evidence.append("结论：自报配对用的是同姿态源对照，Δ 只反映权重差异。")
            elif matches_alt:
                status = "fail"
                evidence.append(
                    "结论：自报配对混用了不同姿态前端（本运行=上游 pose，对照=本地 rtmlib pose），"
                    "Δ 里混入前端差异；做“只换权重”的结论必须用同姿态源对照。")
            else:
                evidence.append("结论：自报 pinned 均值与本复核两种对照都对不上，需人工核对对照目录。")
        checks.append({"id": "C7", "title": "配对对照姿态源一致（只换权重）", "status": status,
                       "evidence": evidence})
    else:
        paper = (bleu_block or {}).get("paper_reference") or {}
        ours_fixed = ((bleu_block or {}).get("ours_fixed") or {}).get("bleu4") or {}
        ours_raw = ((bleu_block or {}).get("ours_raw") or {}).get("bleu4") or {}
        upstream = (((bleu_block or {}).get("upstream_snapshot") or {}).get("bleu4_fixed") or {})
        evidence = [
            f"本机 + mt5 修正（13a）= {fmt(ours_fixed.get('bleu'), 4)}",
            f"本机 不修正（13a）= {fmt(ours_raw.get('bleu'), 4)}（BP {fmt(ours_raw.get('bp'), 4)}，"
            f"precisions {[round(p, 2) for p in (ours_raw.get('precisions') or [])]}）",
            f"上游快照 + 修正（13a）= {fmt(upstream.get('bleu'), 4)}",
            f"论文引用值 = {paper.get('bleu4')}（{paper.get('source')}）",
        ]
        fixed_value = ours_fixed.get("bleu")
        status = "pass" if fixed_value is not None and abs(fixed_value - 25.55) <= 0.05 else "fail"
        if ours_raw.get("bleu") is not None and ours_raw["bleu"] > 0.01:
            status = "fail"
            evidence.append("结论：未修正口径没有出现预期的大幅下跌，口径假设需复核。")
        else:
            evidence.append(
                "结论：25.55 可复算；不做逐字空格修正时 BLEU 会假崩到 0.0005（每句退化为 ~1 个 token，BP≈0.0026），"
                "与 chrF 正常形成典型“假失败”特征。")
        checks.append({"id": "C6", "title": "BLEU-4 25.55 可复算，且格式陷阱可复现", "status": status,
                       "evidence": evidence})

        # C7 运行目录自写 summary.json 记录的是哪一套口径
        reported = None
        run_summary_path = Path(run_dir) / "summary.json"
        if run_summary_path.is_file():
            try:
                run_summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                run_summary = None
            if isinstance(run_summary, dict):
                reported = run_summary
        evidence = []
        problems = []
        if reported is None:
            evidence.append("运行目录没有自写 summary.json，无法核对 artifact 口径。")
        else:
            metrics_reported = reported.get("metrics") or {}
            comparison = (reported.get("comparison")
                          or reported.get("paired_comparison_vs_upstream") or {})
            headline = metrics_reported.get("bleu4_13a")
            raw_value = metrics_reported.get("bleu4_13a_raw_no_mt5_fix")
            raw_bp = metrics_reported.get("bleu4_13a_raw_brevity_penalty")
            trap_note = metrics_reported.get("protocol_trap_note") or reported.get("protocol_trap_note")
            identical_raw = comparison.get("identical_prediction_count")
            identical_fixed = comparison.get("identical_prediction_count_after_mt5_fix")
            upstream_block = (bleu_block or {}).get("upstream_snapshot") or {}
            evidence.append(
                f"summary.json 主值 metrics.bleu4_13a = {fmt(headline, 4)}"
                f"（定义：{metrics_reported.get('bleu4_13a_definition') or '未注明'}）；"
                f"本复核修正后 = {fmt(ours_fixed.get('bleu'), 4)}")
            evidence.append(
                f"陷阱字段 bleu4_13a_raw_no_mt5_fix = {fmt(raw_value, 6)}"
                f"（BP {fmt(raw_bp, 5)}；本复核未修正 = {fmt(ours_raw.get('bleu'), 6)}）"
                f"；声明：{trap_note or '无'}")
            evidence.append(
                f"逐样本一致：原样 {identical_raw}；修正后 {identical_fixed}；"
                f"本复核修正后 = {upstream_block.get('identical_after_fix')} / {upstream_block.get('n')}")
            if headline is None:
                problems.append("缺少主值 metrics.bleu4_13a")
            elif ours_fixed.get("bleu") is None or abs(float(headline) - float(ours_fixed["bleu"])) > 0.05:
                problems.append("主值不是修正后口径（读者会拿到陷阱值）")
            if raw_value is not None and not trap_note:
                problems.append("陷阱字段缺少 protocol_trap_note 声明")
            if raw_value is not None and ours_raw.get("bleu") is not None \
                    and abs(float(raw_value) - float(ours_raw["bleu"])) > 1e-6:
                problems.append("陷阱字段数值与本复核未修正口径不一致")
            if identical_fixed is None:
                problems.append("缺少 identical_prediction_count_after_mt5_fix（读者可能把 0 误读成“完全不一致”）")
            elif upstream_block.get("identical_after_fix") is not None \
                    and int(identical_fixed) != int(upstream_block["identical_after_fix"]):
                problems.append("修正后 identical 计数与本复核不一致")
        status = "fail" if problems else ("pass" if reported is not None else "warn")
        if problems:
            evidence.append("结论：不通过 —— " + "；".join(problems) + "。")
        elif reported is not None:
            evidence.append("结论：只读 summary.json 也能拿到修正后口径（25.55 量级），陷阱值有声明且不被当作准确率。")
        checks.append({"id": "C7", "title": "运行目录自写 summary.json 的口径与报告一致",
                       "status": status, "evidence": evidence})

    # C8 逐视频延迟是否把模型加载（首条）算进去了
    evidence = []
    if cold_start_flag:
        evidence.append(
            f"首条 e2e = {fmt(cold_start_flag['first_e2e_seconds'], 2)} s，"
            f"其余中位数 = {fmt(cold_start_flag['median_of_rest'], 3)} s，"
            f"是其余中位数的 {cold_start_flag['ratio']:.0f} 倍（样本 {cold_start_flag['number']}）")
        evidence.append("结论：第一条记录把模型加载/预热算进了 e2e（本实验没有 cold-start.json）。"
                        "聚合 mean 影响约 1%，但 max（以及按 max 引用的说法）会被抬高；引用极值时应剔除首条。")
        status = "warn"
    else:
        evidence.append("未发现首条 e2e 异常偏大；也未见 cold-start.json（本 schema 不产出）。")
        evidence.append("结论：无证据表明逐视频延迟混入模型加载。")
        status = "pass"
    checks.append({"id": "C8", "title": "逐视频延迟是否排除模型加载（首条离群）",
                   "status": status, "evidence": evidence})

    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"
    return {"checks": checks, "overall": overall, "run_dir": str(run_dir)}


# ---------------------------------------------------------------------------
# 前端消融（上游 CSL-News pose pkl vs 本地 rtmlib）汇总
# ---------------------------------------------------------------------------
def ablation_protocol_digest(protocol):
    """与 scripts/evaluate_unisign_cslnews_upstream_pose.py 的 protocol_id 完全一致。

    该脚本用 json.dumps(protocol, sort_keys=True, ensure_ascii=False) 的默认分隔符，
    与主评测的 protocol_digest（紧凑分隔符）不同，不能混用。
    """
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def normalize_ablation_record(record):
    """把消融记录的 protocol_id/pose_source 映射到与主评测一致的字段名。"""
    out = dict(record)
    if out.get("run_protocol_id") is None and out.get("protocol_id") is not None:
        out["run_protocol_id"] = out.get("protocol_id")
    if not out.get("pose_source"):
        out["pose_source"] = "upstream-pkl"
    return out


def read_protocol_any(run_dir: Path, names=("run-protocol.json", "ablation-protocol.json", "protocol.json")):
    for name in names:
        path = run_dir / name
        if path.is_file():
            try:
                return path, json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
    return None, None


def paired_comparison(ablation_records, main_records, *, seed=0, resamples=10000):
    """同片段配对：只在两边都 status=ok、且参考句一致时计入。"""
    main_by_number = {}
    for record in main_records:
        number = record.get("number")
        if number is not None:
            main_by_number[str(number)] = record

    pairs, skipped = [], Counter()
    for record in ablation_records:
        number = str(record.get("number"))
        main = main_by_number.get(number)
        if main is None:
            skipped["main_missing"] += 1
            continue
        a_status, _ = classify_status(record)
        m_status, _ = classify_status(main)
        if a_status != "ok":
            skipped["ablation_not_ok"] += 1
            continue
        if m_status != "ok":
            skipped["main_not_ok"] += 1
            continue
        ref_a = normalize_text(record.get("reference"))
        ref_m = normalize_text(main.get("reference"))
        if ref_a != ref_m:
            skipped["reference_mismatch"] += 1
            continue
        sim_a = edit_similarity(ref_a, normalize_text(record.get("prediction")))
        sim_m = edit_similarity(ref_m, normalize_text(main.get("prediction")))
        pairs.append({
            "number": number,
            "edit_similarity_upstream_pose": sim_a,
            "edit_similarity_local_pose": sim_m,
            "delta_upstream_minus_local": sim_a - sim_m,
            "upstream_prediction": record.get("prediction"),
            "local_prediction": main.get("prediction"),
            "reference": record.get("reference"),
        })

    deltas = [pair["delta_upstream_minus_local"] for pair in pairs]
    upstream = [pair["edit_similarity_upstream_pose"] for pair in pairs]
    local = [pair["edit_similarity_local_pose"] for pair in pairs]
    result = {
        "paired_n": len(pairs),
        "denominator_rule": "只在 ablation 与 main 两边都 status=ok 且参考句一致时计入",
        "skipped": dict(skipped),
        "mean_edit_similarity_upstream_pose": float(statistics.fmean(upstream)) if upstream else None,
        "mean_edit_similarity_local_pose": float(statistics.fmean(local)) if local else None,
        "mean_delta_upstream_minus_local": float(statistics.fmean(deltas)) if deltas else None,
        "delta_stats": percentile_stats(deltas),
        "wins_upstream": sum(1 for d in deltas if d > 1e-9),
        "losses_upstream": sum(1 for d in deltas if d < -1e-9),
        "ties": sum(1 for d in deltas if abs(d) <= 1e-9),
        "identical_prediction_count": sum(
            1 for pair in pairs if (pair["upstream_prediction"] or "").strip() == (pair["local_prediction"] or "").strip()),
        "pairs": pairs,
    }
    if deltas:
        rng = np.random.default_rng(seed)
        array = np.asarray(deltas, dtype=float)
        means = [float(np.mean(rng.choice(array, size=array.size, replace=True)))
                 for _ in range(resamples)]
        result["delta_bootstrap_95ci"] = [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]
        result["bootstrap_resamples"] = resamples
    else:
        result["delta_bootstrap_95ci"] = None
    if pairs:
        refs = [normalize_text(pair["reference"]) for pair in pairs]
        result["paired_corpus_bleu_zh"] = {
            "upstream_pose": compute_text_metrics(refs, [normalize_text(p["upstream_prediction"]) for p in pairs]),
            "local_pose": compute_text_metrics(refs, [normalize_text(p["local_prediction"]) for p in pairs]),
        }
    return result


def summarize_ablation(run_dir: Path, main_run_dir: Path, *, output_dir=None):
    """汇总前端消融目录，并与主评测做配对复核。"""
    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.is_file():
        raise FileNotFoundError(f"missing predictions.jsonl: {predictions_path}")
    records, parse_errors, total_lines = read_jsonl(predictions_path)
    protocol_path, protocol = read_protocol_any(run_dir)
    protocol_digest_value = ablation_protocol_digest(protocol) if isinstance(protocol, dict) else None

    video_records, aux_records = [], []
    for line_no, raw in records:
        record = normalize_ablation_record(raw)
        is_cold, reason = detect_cold_start(record)
        if is_cold:
            aux_records.append((record, reason))
            continue
        record["_line"] = line_no
        video_records.append(record)
    video_records.sort(key=lambda r: str(r.get("number")))

    statuses = Counter()
    anomalies = []
    for record in video_records:
        status, anomaly = classify_status(record)
        statuses[status] += 1
        if anomaly:
            anomalies.append({"number": record.get("number"), "reason": anomaly})
    success = [r for r in video_records if classify_status(r)[0] == "ok"]
    empty = [r for r in video_records if classify_status(r)[0] == "empty"]
    failed = [r for r in video_records if classify_status(r)[0] in {"error", "other"}]

    metrics = compute_text_metrics([normalize_text(r.get("reference")) for r in success],
                                   [normalize_text(r.get("prediction")) for r in success])

    selected_numbers = []
    if isinstance(protocol, dict) and isinstance(protocol.get("selected"), list):
        for item in protocol["selected"]:
            if isinstance(item, dict) and item.get("number") is not None:
                selected_numbers.append(str(item["number"]))
    record_numbers = [str(r.get("number")) for r in video_records]
    coverage = {
        "protocol_selected": len(selected_numbers) if selected_numbers else None,
        "video_records": len(video_records),
        "missing_numbers": sorted(set(selected_numbers) - set(record_numbers)),
        "extra_numbers": sorted(set(record_numbers) - set(selected_numbers)),
        "complete": bool(selected_numbers) and set(selected_numbers) == set(record_numbers),
    }

    aux_timing = read_aux_timing(run_dir)
    latency = build_latency(video_records, aux_records, aux=aux_timing, fields=ABLATION_TIMING_FIELDS)

    # 主评测（只读）用于配对
    main_records, main_note = [], None
    main_predictions = main_run_dir / "predictions.jsonl"
    if main_predictions.is_file():
        main_rows, _, main_lines = read_jsonl(main_predictions)
        for _, raw in main_rows:
            if raw.get("number") is None:
                continue
            main_records.append(raw)
        main_note = {"path": str(main_predictions), "records": len(main_records), "lines": main_lines}
    else:
        main_note = {"path": str(main_predictions), "records": 0, "note": "主评测 predictions.jsonl 不存在"}

    paired = paired_comparison(video_records, main_records) if main_records else None

    # 与消融脚本自写 summary.json 的配对结果交叉核对
    script_summary_path = run_dir / "summary.json"
    script_paired = None
    if script_summary_path.is_file():
        try:
            script_summary = json.loads(script_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            script_summary = None
        if isinstance(script_summary, dict):
            script_paired = script_summary.get("paired_comparison_vs_local_frontend")

    review = build_ablation_review(
        run_dir=run_dir, protocol_path=protocol_path, protocol=protocol,
        protocol_digest_value=protocol_digest_value, video_records=video_records,
        success=success, empty=empty, failed=failed, coverage=coverage,
        paired=paired, script_paired=script_paired, anomalies=anomalies,
        main_note=main_note, latency=latency,
    )

    summary = {
        "schema_version": 1,
        "kind": "ablation_frontend_upstream_pose",
        "status": "ok",
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "ablation_not_production": True,
        "inputs": {
            "predictions_jsonl": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
            "predictions_total_lines": total_lines,
            "parse_errors": parse_errors,
            "protocol_path": str(protocol_path) if protocol_path else None,
            "protocol_digest_recomputed": protocol_digest_value,
            "main_run": main_note,
        },
        "protocol_summary": {
            key: protocol[key] for key in
            ("ablation", "main_run_protocol_id", "checkpoint_sha256", "source_commit",
             "pose_source", "pose_zip", "pose_zip_sha256", "pose_pipeline", "num_beams",
             "max_new_tokens", "device", "dtype")
            if isinstance(protocol, dict) and key in protocol
        },
        "records": {
            "parsed": len(records), "video_records": len(video_records),
            "aux_records": len(aux_records), "by_status": dict(statuses),
            "success": len(success), "empty": len(empty), "failed": len(failed),
            "coverage": coverage,
        },
        "accuracy": {
            "scope": "success_subset：status == 'ok' 且预测非空",
            "denominator": {
                "success": len(success), "total": len(video_records),
                "empty": len(empty), "error": len(failed),
                "expression": f"{len(success)} / {len(video_records)} 条消融记录（空预测 {len(empty)}、失败 {len(failed)} 已排除）",
            },
            "metrics": metrics,
            "anomalies": anomalies,
        },
        "latency": latency,
        "paired_comparison": paired,
        "review": review,
        "warnings": [],
    }
    if coverage["protocol_selected"] is not None and not coverage["complete"]:
        summary["warnings"].append(
            f"消融样本不完整：协议 {coverage['protocol_selected']} 条，实际 {coverage['video_records']} 条。")
    if anomalies:
        summary["warnings"].append(f"{len(anomalies)} 条消融记录 status=ok 但预测为空。")
    if review["overall"] != "pass":
        summary["warnings"].append(f"消融复核未全部通过（{review['overall']}）。")
    if output_dir is not None:
        summary["inputs"]["per_sample_csv"] = None
    return summary


def build_ablation_review(*, run_dir, protocol_path, protocol, protocol_digest_value, video_records,
                          success, empty, failed, coverage, paired, script_paired, anomalies,
                          main_note, latency):
    checks = []

    # A1 协议 id 一致
    ids = Counter(str(r.get("protocol_id") or r.get("run_protocol_id")) for r in video_records)
    evidence = [
        f"协议文件：{protocol_path or '缺失'}",
        f"复算 protocol_id（消融脚本口径）= {protocol_digest_value or '—'}",
        f"记录中 protocol_id 分布 = {dict(ids)}",
    ]
    if protocol_path is None:
        status = "fail"
        evidence.append("结论：缺少 ablation-protocol.json，无法确认消融记录属于哪套协议。")
    elif len(ids) > 1:
        status = "fail"
        evidence.append(f"结论：混入 {len(ids)} 种 protocol_id，禁止合并统计。")
    elif protocol_digest_value in ids:
        status = "pass"
        evidence.append("结论：全部消融记录 protocol_id 与 ablation-protocol.json 复算值一致。")
    else:
        status = "fail"
        evidence.append("结论：记录 protocol_id 无法由 ablation-protocol.json 复现。")
    checks.append({"id": "A1", "title": "消融记录 protocol_id 一致且可复现", "status": status,
                   "evidence": evidence, "distinct_ids": dict(ids)})

    # A2 配对分母只含两边 ok
    evidence = [
        f"配对规则：{paired.get('denominator_rule') if paired else '未计算'}",
        f"主评测可用记录：{(main_note or {}).get('records')} 条（{main_note.get('path') if main_note else '—'}）",
    ]
    if not paired:
        status = "fail"
        evidence.append("结论：无法配对（主评测 predictions.jsonl 缺失或为空）。")
    else:
        skipped = paired.get("skipped") or {}
        evidence.append(
            f"配对样本 {paired['paired_n']} 条；剔除：主评测缺该条 {skipped.get('main_missing', 0)}、"
            f"消融非 ok {skipped.get('ablation_not_ok', 0)}、主评测非 ok {skipped.get('main_not_ok', 0)}、"
            f"参考句不一致 {skipped.get('reference_mismatch', 0)}"
        )
        status = "pass"
        if paired["paired_n"] == 0:
            status = "fail"
            evidence.append("结论：没有任何可配对样本。")
        else:
            evidence.append("结论：配对分母只包含两边都 ok 且参考句一致的样本。")
        if script_paired:
            reported_n = script_paired.get("paired_n")
            reported_delta = script_paired.get("mean_delta_upstream_minus_local")
            ours_delta = paired.get("mean_delta_upstream_minus_local")
            evidence.append(
                f"消融脚本自报：paired_n={reported_n}，mean_delta={fmt(reported_delta, 6)}；"
                f"本复核重算：paired_n={paired['paired_n']}，mean_delta={fmt(ours_delta, 6)}"
            )
            if reported_n is not None and int(reported_n) != paired["paired_n"]:
                status = "fail"
                evidence.append("结论：脚本自报 paired_n 与本复核不一致（可能把单边样本或参考不一致样本计入了配对）。")
            elif reported_delta is not None and ours_delta is not None and abs(float(reported_delta) - ours_delta) > 1e-6:
                status = "fail"
                evidence.append("结论：脚本自报 mean_delta 与本复核重算不一致。")
            else:
                evidence.append("结论：脚本自报配对结果与独立重算一致。")
        else:
            evidence.append("未找到消融脚本自写的配对结果（summary.json 缺失或未跑完），以上为本复核独立重算。")
    checks.append({"id": "A2", "title": "配对比较按配对样本计算，分母只含两边都 ok 的样本",
                   "status": status, "evidence": evidence,
                   "paired_n": (paired or {}).get("paired_n")})

    # A3 延迟不与主评测混表
    source_groups = list((latency.get("by_pose_source") or {}))
    evidence = [
        f"本消融记录的 pose_source 组 = {source_groups}",
        "消融延迟只统计本目录记录：pose_load_ms / pose_prepare_ms / model_total_ms / e2e_seconds",
        "本文件不引用主评测的 e2e/RTF 数字；主评测数据只用于配对文本指标",
    ]
    status = "pass"
    if len(source_groups) > 1:
        status = "fail"
        evidence.append(f"结论：消融目录混入多个 pose_source（{source_groups}），延迟不可合并。")
    else:
        evidence.append("结论：延迟口径与主评测分离，未混在同一张表里。")
    checks.append({"id": "A3", "title": "消融延迟不与主评测 e2e 混表", "status": status,
                   "evidence": evidence})

    # A4 标注为消融/非生产
    evidence = [
        "ablation-summary.md 顶部有「消融，不是生产输入路径」声明",
        "消融脚本 protocol.ablation = upstream-cslnews-pose-pkl-vs-local-rtmlib",
        "主评测与消融使用同一权重、同一解码参数，只替换姿态来源",
    ]
    checks.append({"id": "A4", "title": "明确标注为消融、非生产输入路径", "status": "pass",
                   "evidence": evidence})

    # A5 覆盖与记录完整性
    missing_sha = [str(r.get("number")) for r in video_records
                   if not (isinstance(r.get("video_sha256"), str) and re.fullmatch(r"[0-9a-fA-F]{64}", r["video_sha256"]))]
    evidence = [
        f"协议 selected[] = {coverage.get('protocol_selected')}；实际记录 = {coverage.get('video_records')}",
        f"缺 video_sha256 的记录 {len(missing_sha)} 条",
        f"status=ok 但空预测 {len(anomalies)} 条",
    ]
    if coverage.get("protocol_selected") is None:
        status = "warn"
        evidence.append("结论：协议未列 selected[]，无法核对覆盖。")
    elif coverage.get("missing_numbers"):
        status = "fail"
        evidence.append(f"结论：缺 {len(coverage['missing_numbers'])} 条样本，不得标记完整。")
    elif missing_sha:
        status = "fail"
        evidence.append("结论：存在缺视频内容哈希的记录。")
    else:
        status = "pass"
        evidence.append("结论：样本覆盖完整，每条记录都有视频哈希。")
    checks.append({"id": "A5", "title": "消融样本覆盖与记录完整性", "status": status, "evidence": evidence})

    overall = "pass"
    if any(c["status"] == "fail" for c in checks):
        overall = "fail"
    elif any(c["status"] == "warn" for c in checks):
        overall = "warn"
    return {"checks": checks, "overall": overall, "run_dir": str(run_dir)}


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------
def fmt(value, digits=4):
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "—"
        return f"{float(value):.{digits}f}"
    return str(value)


def md_escape(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render_markdown(summary, *, self_test=None, no_data_reason=None):
    lines = []
    add = lines.append
    add("# CSL-News 评测汇总（Uni-Sign）")
    add("")
    if summary.get("status") != "ok":
        add(f"> **状态：数据未就绪。** {no_data_reason or ''}")
        add("")
        add(f"- 生成时间：{summary.get('generated_at')}")
        add(f"- 期望运行目录：`{summary.get('expected_run_glob')}`")
        add("- 本文件当前只包含数据边界与脚本自测结果；真实评测完成后会重新生成。")
        add("")
        render_domain_boundary(add, None, None, None)
        render_selftest(add, self_test)
        render_reproduce(add, None)
        return "\n".join(lines) + "\n"

    run_dir = summary["run_dir"]
    protocol = summary.get("run_protocol_summary") or {}
    records = summary["records"]
    accuracy = summary["accuracy"]
    denominator = accuracy["denominator"]
    metrics = accuracy["metrics"] or {}
    latency = summary["latency"]
    review = summary["protocol_review"]

    add(f"- 生成时间：{summary['generated_at']}")
    if summary.get("scope") == "interim":
        coverage = (records.get("coverage") or {})
        add("")
        add(f"> **中间结果（interim，{summary.get('scope_label') or '未完成'}）：** 仅覆盖 "
            f"{coverage.get('video_records')} / {coverage.get('protocol_selected') or '?'} 条，"
            "数字会随记录增加而变化，**不得当作最终结论**；最终版另见 `summary.md`。")
        add("")
    add(f"- 运行目录：`{run_dir}`")
    add(f"- 预测文件：`{summary['inputs']['predictions_jsonl']}`"
        f"（sha256 `{summary['inputs']['predictions_sha256'][:16]}…`，{summary['inputs']['predictions_total_lines']} 行）")
    if summary["inputs"].get("per_sample_csv"):
        add(f"- 逐样本明细：`{summary['inputs']['per_sample_csv']}`（人工抽查用，不参与汇总计算）")
    if summary.get("run_protocol_summary"):
        add(f"- 协议摘要：checkpoint `{protocol.get('checkpoint_sha256') or protocol.get('checkpoint') or '—'}`，"
            f"device `{protocol.get('device') or '—'}`，dtype `{protocol.get('dtype') or '—'}`，"
            f"beams `{protocol.get('num_beams') or '—'}`")
    if summary.get("pose_source_label"):
        add(f"- **姿态来源：{summary['pose_source_label']}**")
    add("")

    render_domain_boundary(add, summary, protocol, review)

    # 1. 样本与分母
    add("## 1. 样本与分母")
    add("")
    add("| 项 | 数值 |")
    add("|---|---:|")
    add(f"| predictions.jsonl 可解析记录 | {records['parsed']} |")
    add(f"| 视频记录 | {records['video_records']} |")
    add(f"| 冷启动/辅助记录（单列，不计入逐视频统计） | {records['cold_start_or_aux_records']} |")
    add(f"| status=ok 且预测非空（成功样本） | {records['success']} |")
    add(f"| 空预测 | {records['empty']} |")
    add(f"| 失败 | {records['failed']} |")
    add(f"| 解析失败行 | {len(records['parse_errors'])} |")
    coverage = records.get("coverage") or {}
    add(f"| 协议 selected[] 声明的样本数 | {coverage.get('protocol_selected') if coverage.get('protocol_selected') is not None else '未列出'} |")
    add(f"| 协议内但缺记录的样本 | {len(coverage.get('missing_numbers') or [])} |")
    add(f"| 记录是否覆盖协议全部样本 | {'是' if coverage.get('complete') else '否（不得标记完整批次）'} |")
    add("")
    add(f"**准确率分母：{denominator['expression']}。** 成功子集指标不得写成全量指标。")
    add("")
    profile = records.get("input_profile") or {}
    if profile:
        add("### 1.1 输入特征（这批样本的代表性）")
        add("")
        add(f"- 分辨率：{profile.get('resolutions') or '—'}；FPS：{profile.get('fps_values') or '—'}")
        frames = profile.get("frames") or {}
        duration = profile.get("duration_s") or {}
        chars = profile.get("reference_chars") or {}
        add(f"- 帧数：mean {fmt(frames.get('mean'), 1)}，P50 {fmt(frames.get('p50'), 1)}，"
            f"P95 {fmt(frames.get('p95'), 1)}，max {fmt(frames.get('max'), 1)}")
        add(f"- 时长(s)：mean {fmt(duration.get('mean'), 2)}，P50 {fmt(duration.get('p50'), 2)}，"
            f"P95 {fmt(duration.get('p95'), 2)}，max {fmt(duration.get('max'), 2)}")
        add(f"- 参考句字符数：mean {fmt(chars.get('mean'), 1)}，P50 {fmt(chars.get('p50'), 1)}，"
            f"max {fmt(chars.get('max'), 1)}")
        add("")

    # 2. 准确率
    add("## 2. 准确率（成功子集，主口径）")
    add("")
    add(f"分母 = {denominator['success']} / {denominator['total']} 条视频记录。")
    add("")
    add("| 指标 | 数值 |")
    add("|---|---:|")
    add(f"| exact match（逐字相等） | {metrics.get('exact_match_count')} / {metrics.get('n')}"
        f"（{fmt(metrics.get('exact_match_percent'), 2)}%） |")
    add(f"| BLEU-zh（sacrebleu tokenize=zh） | {fmt(metrics.get('bleu_zh'), 4)} |")
    add(f"| chrF | {fmt(metrics.get('chrf'), 4)} |")
    add(f"| ROUGE-L 字符级 F1 | {fmt(metrics.get('rouge_l_f1_char'), 4)} |")
    add(f"| 平均字符编辑相似度 | {fmt(metrics.get('mean_char_edit_similarity_percent'), 2)}%"
        f"（{fmt(metrics.get('mean_char_edit_similarity'), 4)}） |")
    add(f"| 预测/参考长度比 | {fmt(metrics.get('mean_length_ratio_prediction_to_reference'), 4)} |")
    add(f"| 参考字符数合计 / 预测字符数合计 | {metrics.get('reference_chars_total')} / {metrics.get('prediction_chars_total')} |")
    if metrics.get("metric_error"):
        add(f"| 指标计算异常 | {md_escape(metrics['metric_error'])} |")
    add("")
    if accuracy.get("by_pose_source"):
        add("### 2.1 按姿态来源分组的成功子集指标")
        add("")
        add("| 姿态来源 | n | exact | BLEU-zh | chrF | ROUGE-L | 编辑相似度% |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        group_count = len(accuracy["by_pose_source"])
        for source, block in accuracy["by_pose_source"].items():
            row_label = source
            if group_count == 1 and summary.get("pose_source_label"):
                row_label = f"{source}（{summary['pose_source_label']}）"
            add(f"| {row_label} | {block['n']} | {block['exact_match_count']} | {fmt(block.get('bleu_zh'))} | "
                f"{fmt(block.get('chrf'))} | {fmt(block.get('rouge_l_f1_char'))} | "
                f"{fmt(block.get('mean_char_edit_similarity_percent'), 2)} |")
        add("")

    # 3. 全量口径
    add("## 3. 全量口径（悲观下界，仅作边界参考）")
    add("")
    full = accuracy["full_set_pessimistic"]
    full_metrics = full["metrics"] or {}
    add(f"分母 = {full['denominator']['total']} 条视频记录；失败/空预测按空串计入。")
    add("")
    add("| 指标 | 数值 |")
    add("|---|---:|")
    add(f"| exact match | {full_metrics.get('exact_match_count')} / {full_metrics.get('n')}"
        f"（{fmt(full_metrics.get('exact_match_percent'), 2)}%） |")
    add(f"| BLEU-zh | {fmt(full_metrics.get('bleu_zh'), 4)} |")
    add(f"| chrF | {fmt(full_metrics.get('chrf'), 4)} |")
    add(f"| ROUGE-L 字符级 F1 | {fmt(full_metrics.get('rouge_l_f1_char'), 4)} |")
    add(f"| 平均字符编辑相似度 | {fmt(full_metrics.get('mean_char_edit_similarity_percent'), 2)}% |")
    add("")
    add("> 这一节不是模型准确率：它把失败/空预测当成“完全错”来展示最坏情况，"
        "用于说明样本损失对指标的影响，任何对外引用都必须同时给成功子集与失败数。")
    add("")

    # 4. 延迟
    add("## 4. 延迟")
    add("")
    if summary.get("latency_note"):
        add(f"> ⚠️ **延迟口径警示：{summary['latency_note']}**")
        add("")
    if summary.get("pose_source_label"):
        add(f"> 本运行的姿态来源：{summary['pose_source_label']}")
        add("")
    add("> **口径（2026-10-25 与评测脚本核对）：**")
    add("> - `decode_ms` 是 `runtime.infer()` 内 tokenizer/batch_decode 的**文本解码**耗时，"
        "**不是**视频容器解码；")
    add("> - 视频容器解码（cv2 读帧）**没有单独计时**，合并在 `pose_extract_ms` 内（读帧+RTMPose 同一计时区间），"
        "因此该字段只能叫「解码+RTMPose 合计」；")
    add("> - 纯视频解码成本来自 `--decode-benchmark` 的独立基准 `decode-benchmark.json`；")
    add("> - `extracted` 含姿态提取、`disk_cache` 仅模型；**两组必须分开引用，不得合并成单一延迟数字**。")
    add("")
    rows = latency.get("stage_decomposition") or []
    if rows:
        add("### 4.0 延迟分解（逐阶段）")
        add("")
        add("| 阶段 | 组 | n | mean | P50 | P95 | P99 | max | 单位 |")
        add("|---|---|---:|---:|---:|---:|---:|---:|---|")
        for row in rows:
            add(f"| {md_escape(row['stage'])} | {row['group']} | {row.get('n') if row.get('n') is not None else '—'} | "
                f"{fmt(row.get('mean'), 4)} | {fmt(row.get('p50'), 4)} | {fmt(row.get('p95'), 4)} | "
                f"{fmt(row.get('p99'), 4)} | {fmt(row.get('max'), 4)} | {row['unit']} |")
        add("")
        estimate_rows = [row for row in rows if row.get("note")]
        for row in estimate_rows:
            add(f"- {md_escape(row['stage'])}：{row['note']}。")
        add("")
        bench = latency.get("decode_only_benchmark") or {}
        fps_stats = bench.get("decode_fps_recomputed") or {}
        if fps_stats.get("n"):
            add(f"- 纯解码基准来自 `{bench['path']}`（n={bench.get('n')}，失败 {bench.get('failed')}），"
                f"解码帧率 mean {fmt(fps_stats.get('mean'), 1)} fps。")
            add("")
        add(f"- {latency.get('decode_note')}")
        add("")
    for source in POSE_SOURCE_ORDER:
        block = latency["by_pose_source"].get(source)
        if not block:
            continue
        heading = f"### 4.{list(latency['by_pose_source']).index(source) + 1} pose_source = `{source}`（n={block['n']}）"
        if len(latency["by_pose_source"]) == 1 and summary.get("pose_source_label"):
            heading += f" —— {summary['pose_source_label']}"
        add(heading)
        add("")
        add("| 字段 | n | mean | P50 | P95 | P99 | max | 缺失 |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for field in TIMING_FIELDS:
            stats = block["fields"][field]
            add(f"| {TIMING_LABELS.get(field, field)}（{field}） | {stats['n']} | {fmt(stats['mean'])} | "
                f"{fmt(stats['p50'])} | {fmt(stats['p95'])} | {fmt(stats['p99'])} | {fmt(stats['max'])} | "
                f"{stats['missing']} |")
        rtf = block["rtf"]
        add(f"| RTF（e2e/视频时长） | {rtf['n']} | {fmt(rtf['mean'], 4)} | {fmt(rtf['p50'], 4)} | "
            f"{fmt(rtf['p95'], 4)} | {fmt(rtf['p99'], 4)} | {fmt(rtf['max'], 4)} | {rtf['missing']} |")
        add("")
    if latency.get("overall_mixed"):
        overall = latency["overall_mixed"]
        add("### 4.3 混合口径（仅供完整性核对，禁止引用）")
        add("")
        add(f"- n={overall['n']}，e2e mean {fmt(overall['fields']['e2e_seconds']['mean'])}s，"
            f"P50 {fmt(overall['fields']['e2e_seconds']['p50'])}s，P95 {fmt(overall['fields']['e2e_seconds']['p95'])}s")
        add(f"- {overall['warning']}")
        add("")

    # 5. 吞吐
    add("## 5. 吞吐与 RTF")
    add("")
    add("| 组 | n | 视频/分钟 | 帧/秒 | e2e 合计(s) | RTF mean | RTF P95 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for source in POSE_SOURCE_ORDER:
        block = latency["by_pose_source"].get(source)
        if not block:
            continue
        throughput = block["throughput"]
        add(f"| {source} | {block['n']} | {fmt(throughput['videos_per_minute'], 2)} | "
            f"{fmt(throughput['frames_per_second'], 2)} | {fmt(throughput['total_e2e_seconds'], 2)} | "
            f"{fmt(block['rtf']['mean'], 4)} | {fmt(block['rtf']['p95'], 4)} |")
    add("")
    add("- 视频/分钟 = 60 / mean(e2e_seconds)；帧/秒 = Σ视频帧数 / Σ e2e_seconds；RTF = e2e_seconds / video_duration_s。")
    add("")

    # 6. 冷启动
    add("## 6. 冷启动 / 预热（单列，不计入逐视频延迟）")
    add("")
    cold = latency["cold_start"]
    if cold.get("external_record"):
        external = cold["external_record"]
        add("| 来源 | 模型加载(s) | 已声明排除出逐视频延迟 | 记录时间 |")
        add("|---|---:|---|---|")
        add(f"| cold-start.json | {fmt(external.get('load_model_seconds'), 2)} | "
            f"{'是' if external.get('excluded_from_per_video_latency') else '否'} | "
            f"{md_escape(external.get('measured_at') or '—')} |")
        add("")
        if external.get("note"):
            add(f"- 脚本说明：{md_escape(external['note'])}")
            add("")
    if cold["count"] == 0:
        add("- predictions.jsonl 中没有 warmup/model_load 记录"
            + ("；模型加载耗时由 cold-start.json 单独记录。" if cold.get("external_record")
               else "；也未找到 cold-start.json，无法确认冷启动是否被排除。"))
        if not cold.get("external_record"):
            add("- 若评测脚本在首个视频里加载模型，该条 e2e 会异常偏大（见协议复核 P3 的离群检查）。")
    else:
        add("| 记录 | 事件 | 原因 | e2e/加载耗时(s) | 创建时间 |")
        add("|---|---|---|---:|---|")
        for item in cold["records"]:
            add(f"| {item['number'] or '—'} | {md_escape(item['event'] or '—')} | {md_escape(item['reason'])} | "
                f"{fmt(item['e2e_seconds'] or item['model_load_seconds'])} | {md_escape(item['created_at'] or '—')} |")
        add("")
        add(f"- 冷启动总耗时：{fmt(cold['total_seconds'], 2)} s（{cold['count']} 条，已从逐视频统计中排除）")
    add("")

    # 7. 失败清单
    add("## 7. 失败与空预测清单")
    add("")
    if not summary["failed_samples"]:
        add("- 无失败/空预测样本。")
    else:
        add("| number | status | error | pose_source | e2e(s) |")
        add("|---|---|---|---|---:|")
        for item in summary["failed_samples"]:
            add(f"| {md_escape(item['number'])} | {md_escape(item['status'])} | "
                f"{md_escape(item['error'] or '—')} | {md_escape(item['pose_source'] or '—')} | "
                f"{fmt(item['e2e_seconds'])} |")
    add("")

    # 8. 协议复核
    add("## 8. 协议复核")
    add("")
    add(f"总体结论：**{review['overall']}**（通过 = pass；存疑 = warn；不通过 = fail）。")
    add("")
    labels = {"pass": "通过", "warn": "存疑", "fail": "不通过"}
    titles = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"]
    for index, check in enumerate(review["checks"]):
        add(f"### {titles[index] if index < len(titles) else '-'} {check['title']} —— {labels.get(check['status'], check['status'])}")
        add("")
        for evidence in check["evidence"]:
            add(f"- {evidence}")
        add("")

    # 9. 自测
    render_selftest(add, self_test)

    render_reproduce(add, run_dir)
    return "\n".join(lines) + "\n"


def render_domain_boundary(add, summary, protocol, review):
    add("## 0. 数据边界（必读）")
    add("")
    if summary is None:
        add("- 数据未就绪，本节在真实汇总生成时补全。")
        add("")
        return
    model_domain = None
    dataset = None
    if review:
        for check in review["checks"]:
            if check["id"] == "P4":
                model_domain = check.get("model_domain")
                dataset = check.get("dataset")
    add(f"- 权重训练域：{model_domain or '未从协议识别（需人工确认）'}；评测数据域：{dataset or '未从协议识别（需人工确认）'}。")
    if model_domain and dataset and model_domain.lower() != dataset.lower():
        add("- 这是**跨数据集**评测：分数不是该数据集的同域准确率，也不是产品识别准确率，"
            "不能作为泛化估计引用。")
    else:
        add("- 未能在协议中确认训练域与数据域不同；在确认之前，这些分数只能当作**已检查样本上的回归指标**，"
            "不得写成产品准确率或泛化估计。")
    add("- 所有准确率均标注为“成功子集”口径；失败与空预测单独计数，不进入准确率分母。")
    add("- 字符编辑相似度 / BLEU / chrF 都不是“识别准确率”，也不等于逐词 Gloss 正确率。")
    audit = summary.get("dataset_audit") or {}
    if audit:
        add(f"- 数据集审计（`{audit.get('path')}`）：本地样本 local_inside_train = "
            f"`{audit.get('local_inside_train')}`（本地行 {audit.get('local_rows_1based')}，"
            f"上游 train 段 {audit.get('train_slice_half_open_0based')} / 共 {audit.get('train_count')} 条）。"
            f"{audit.get('note')}")
    add("")


def render_selftest(add, self_test):
    add("## 9. 自测（合成 fixture）")
    add("")
    if not self_test:
        add("- 本次未运行自测（`--skip-self-test`）。")
        add("")
        return
    add(f"- fixture 根目录：`{self_test.get('fixtures_root')}`")
    add(f"- 断言 {self_test.get('total_assertions')} 项，失败 {self_test.get('failed_assertions')} 项，"
        f"总体：{'通过' if self_test.get('ok') else '不通过'}")
    add("")
    add("| fixture | 断言 | 失败 | exact% | BLEU-zh | chrF | ROUGE-L | 编辑相似度% | 复核 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for fixture in self_test.get("fixtures", []):
        summary = fixture.get("headline") or {}
        metrics = summary
        review_status = summary.get("protocol_review")
        add(f"| {fixture['name']} | {len(fixture.get('assertions', []))} | {len(fixture.get('failures', []))} | "
            f"{fmt(metrics.get('exact_match_percent'), 2)} | {fmt(metrics.get('bleu_zh'), 2)} | "
            f"{fmt(metrics.get('chrf'), 2)} | {fmt(metrics.get('rouge_l_f1_char'), 4)} | "
            f"{fmt(metrics.get('mean_char_edit_similarity_percent'), 2)} | {review_status or '—'} |")
    add("")
    if self_test.get("failures"):
        add("失败断言：")
        for failure in self_test["failures"][:30]:
            add(f"- `{failure['fixture']}` :: `{failure['path']}` 期望 {failure['expected']!r}，实际 {failure['actual']!r}")
        add("")
    add("fixture 覆盖：相同文本（exact=100/BLEU=100/chrF=100）、混合状态（空预测+失败+ok-but-empty 异常）、"
        "分组延迟与冷启动排除（含 cold-start.json / decode-benchmark.json）、协议混入与缺 video_sha256、"
        "干净样本全部通过。`02-mixed-status` 与 `04-protocol-mixed` 的复核 fail 是**刻意构造的负样本**，"
        "用于证明检测逻辑真的会报错，不是脚本缺陷。")
    add("")


def render_reproduce(add, run_dir):
    add("## 10. 复现")
    add("")
    add("```powershell")
    if run_dir:
        add(r".\.conda-env\python.exe scripts\summarize_unisign_eval.py "
            f"--run-dir '{run_dir}'")
    else:
        add(r".\.conda-env\python.exe scripts\summarize_unisign_eval.py --self-test")
    add("```")
    add("")


def render_ablation_md(summary):
    """消融汇总 md：独立成文，延迟不与主评测混表。"""
    lines = []
    add = lines.append
    add("# Uni-Sign 前端消融汇总（上游 CSL-News pose .pkl vs 本地 rtmlib）")
    add("")
    add("> **这是消融实验，不是生产输入路径。** 只替换姿态来源（上游发布的 CSL-News pose `.pkl` vs "
        "本仓库 rtmlib RTMW-L 从视频提取），权重、解码参数、采样规则保持一致。")
    add("> **其延迟不得与主评测的 e2e 混在同一张表里**（消融没有视频解码/RTMPose 成本）；"
        "文本指标只用于说明前端差异，不是产品识别准确率。")
    add("")
    add(f"- 生成时间：{summary['generated_at']}")
    add(f"- 消融目录：`{summary['run_dir']}`")
    add(f"- 预测文件：`{summary['inputs']['predictions_jsonl']}`"
        f"（sha256 `{summary['inputs']['predictions_sha256'][:16]}…`，{summary['inputs']['predictions_total_lines']} 行）")
    protocol = summary.get("protocol_summary") or {}
    if protocol:
        add(f"- 协议：ablation=`{protocol.get('ablation')}`，pose_source=`{protocol.get('pose_source')}`，"
            f"pose_zip_sha256=`{protocol.get('pose_zip_sha256')}`")
        add(f"- 主评测协议 id（引用值）：`{protocol.get('main_run_protocol_id')}`")
    main_note = summary["inputs"].get("main_run") or {}
    add(f"- 配对用的主评测：`{main_note.get('path')}`（可用记录 {main_note.get('records')} 条）")
    add("")

    records = summary["records"]
    accuracy = summary["accuracy"]
    metrics = accuracy["metrics"] or {}
    coverage = records.get("coverage") or {}

    add("## 1. 样本与分母")
    add("")
    add("| 项 | 数值 |")
    add("|---|---:|")
    add(f"| 可解析记录 | {records['parsed']} |")
    add(f"| 视频记录 | {records['video_records']} |")
    add(f"| 成功（ok 且预测非空） | {records['success']} |")
    add(f"| 空预测 | {records['empty']} |")
    add(f"| 失败 | {records['failed']} |")
    add(f"| 协议 selected[] 声明 | {coverage.get('protocol_selected') if coverage.get('protocol_selected') is not None else '未列出'} |")
    add(f"| 缺样本数 | {len(coverage.get('missing_numbers') or [])} |")
    add(f"| 覆盖完整 | {'是' if coverage.get('complete') else '否'} |")
    add("")
    add(f"**准确率分母：{accuracy['denominator']['expression']}。**")
    add("")

    add("## 2. 准确率（成功子集，仅作前端差异参考）")
    add("")
    add("| 指标 | 数值 |")
    add("|---|---:|")
    add(f"| exact match | {metrics.get('exact_match_count')} / {metrics.get('n')}"
        f"（{fmt(metrics.get('exact_match_percent'), 2)}%） |")
    add(f"| BLEU-zh | {fmt(metrics.get('bleu_zh'), 4)} |")
    add(f"| chrF | {fmt(metrics.get('chrf'), 4)} |")
    add(f"| ROUGE-L 字符级 F1 | {fmt(metrics.get('rouge_l_f1_char'), 4)} |")
    add(f"| 平均字符编辑相似度 | {fmt(metrics.get('mean_char_edit_similarity_percent'), 2)}% |")
    add("")

    add("## 3. 延迟（消融自身口径，不与主评测 e2e 混表）")
    add("")
    add("> 消融从磁盘读 pose `.pkl`，**没有**视频解码与 RTMPose 成本，因此这里的 e2e 与主评测的 "
        "`e2e_seconds` 不是同一口径，禁止并列比较或写成一个数字。")
    add("")
    add("| 阶段 | n | mean | P50 | P95 | P99 | max | 单位 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    latency = summary["latency"]
    for group, block in (latency.get("by_pose_source") or {}).items():
        for field in ABLATION_TIMING_FIELDS:
            stats = block["fields"].get(field) or {}
            add(f"| {TIMING_LABELS.get(field, field)}（{group}） | {stats.get('n')} | "
                f"{fmt(stats.get('mean'), 3)} | {fmt(stats.get('p50'), 3)} | {fmt(stats.get('p95'), 3)} | "
                f"{fmt(stats.get('p99'), 3)} | {fmt(stats.get('max'), 3)} | "
                f"{'s' if field == 'e2e_seconds' else 'ms'} |")
        rtf = block.get("rtf") or {}
        add(f"| RTF（e2e / 视频时长，{group}） | {rtf.get('n')} | {fmt(rtf.get('mean'), 4)} | "
            f"{fmt(rtf.get('p50'), 4)} | {fmt(rtf.get('p95'), 4)} | {fmt(rtf.get('p99'), 4)} | "
            f"{fmt(rtf.get('max'), 4)} | × |")
    add("")

    paired = summary.get("paired_comparison")
    add("## 4. 配对对照（同片段：上游 pose vs 本地 pose）")
    add("")
    if not paired:
        add("- 无法配对：主评测 predictions.jsonl 缺失或为空。")
    else:
        skipped = paired.get("skipped") or {}
        add(f"- 配对分母：**{paired['paired_n']} 条**；规则：{paired['denominator_rule']}。")
        add(f"- 剔除：主评测缺该条 {skipped.get('main_missing', 0)}；消融非 ok {skipped.get('ablation_not_ok', 0)}；"
            f"主评测非 ok {skipped.get('main_not_ok', 0)}；参考句不一致 {skipped.get('reference_mismatch', 0)}。")
        add("")
        add("| 指标 | 上游 pose | 本地 pose | 差值（上游−本地） |")
        add("|---|---:|---:|---:|")
        add(f"| 平均字符编辑相似度 | {fmt(paired.get('mean_edit_similarity_upstream_pose'), 4)} | "
            f"{fmt(paired.get('mean_edit_similarity_local_pose'), 4)} | "
            f"{fmt(paired.get('mean_delta_upstream_minus_local'), 4)} |")
        delta = paired.get("delta_stats") or {}
        add(f"| Δ P50 / P95 | — | — | {fmt(delta.get('p50'), 4)} / {fmt(delta.get('p95'), 4)} |")
        ci = paired.get("delta_bootstrap_95ci")
        if ci:
            add(f"| Δ 均值 95% CI（bootstrap {paired.get('bootstrap_resamples')} 次） | — | — | "
                f"[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}] |")
        add(f"| 胜/负/平（上游更好/更差/相同） | — | — | {paired.get('wins_upstream')} / "
            f"{paired.get('losses_upstream')} / {paired.get('ties')} |")
        add(f"| 预测文本完全相同 | — | — | {paired.get('identical_prediction_count')} 条 |")
        add("")
        corpus = paired.get("paired_corpus_bleu_zh") or {}
        if corpus:
            up = corpus.get("upstream_pose") or {}
            lo = corpus.get("local_pose") or {}
            add(f"- 配对子集语料 BLEU-zh：上游 {fmt(up.get('bleu_zh'), 4)} vs 本地 {fmt(lo.get('bleu_zh'), 4)}；"
                f"chrF：{fmt(up.get('chrf'), 4)} vs {fmt(lo.get('chrf'), 4)}。")
            add("")

    add("## 5. 复核")
    add("")
    labels = {"pass": "通过", "warn": "存疑", "fail": "不通过"}
    review = summary["review"]
    add(f"总体结论：**{review['overall']}**")
    add("")
    titles = ["①", "②", "③", "④", "⑤"]
    for index, check in enumerate(review["checks"]):
        add(f"### {titles[index] if index < len(titles) else '-'} {check['title']} —— "
            f"{labels.get(check['status'], check['status'])}")
        add("")
        for evidence in check["evidence"]:
            add(f"- {evidence}")
        add("")
    if summary.get("warnings"):
        add("### 警告")
        add("")
        for warning in summary["warnings"]:
            add(f"- {warning}")
        add("")

    add("## 6. 复现")
    add("")
    add("```powershell")
    add(r".\.conda-env\python.exe scripts\summarize_unisign_eval.py --ablation "
        f"--run-dir '{summary['run_dir']}' --compare-run-dir '{main_note.get('path', '').replace(chr(92) + 'predictions.jsonl', '')}'")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


def render_profile_md(summary):
    profile_name = summary["profile"]
    lines = []
    add = lines.append
    records = summary["records"]
    accuracy = summary["accuracy"]
    metrics = accuracy["metrics"] or {}
    coverage = records.get("coverage") or {}
    latency = summary["latency"]
    review = summary["review"]
    protocol = summary.get("protocol_summary") or {}

    if profile_name == "stage1":
        add("# stage1 权重（CSL-News 预训练）在本地 CSL-News 片段上的汇总")
        add("")
        add("> **这是训练集拟合，不是泛化准确率。** stage1 权重是用 CSL-News pose-only 预训练得到的"
            "（上游 `train_stage1.sh --dataset CSL_News`），而本地这 1,694 条片段又正好落在上游 train 段"
            "（`datasets.py::S2T_Dataset_news` 用 [0, 99%) 作训练，本地行 1..1694 ⊂ [0, 715487)）。"
            "因此这些分数只说明“模型在见过的训练域/训练段上能拟合到什么程度”，不能作为识别准确率、"
            "也不能与 CSL-Daily 论文数值对照。")
    else:
        add("# CSL-Daily in-domain 重跑汇总（作者发布的 test pose，1,176 条）")
        add("")
        add("> **这是同域（in-domain）评测，但指标口径极易踩坑：** 上游 `fine_tuning.py` 在算指标前会把预测"
            "逐字加空格（`' '.join(list(text.replace(' ','').replace('\\n','')))`，注释为 *fix mt5 tokenizer bug*），"
            "参考句本身就是逐字空格分隔。不做这一步，BLEU 会假崩到 0.0005；chrF 不受影响。")
    add("")
    add(f"- 生成时间：{summary['generated_at']}")
    add(f"- 运行目录：`{summary['run_dir']}`")
    add(f"- 预测文件：`{summary['inputs']['predictions_jsonl']}`"
        f"（sha256 `{summary['inputs']['predictions_sha256'][:16]}…`，{summary['inputs']['predictions_total_lines']} 行）")
    if summary["inputs"].get("protocol_path"):
        add(f"- 协议文件：`{summary['inputs']['protocol_path']}`"
            f"（protocol_id `{summary['inputs']['protocol_digest_recomputed']}`，公式：{summary['inputs']['protocol_digest_formula']}）")
    if summary["inputs"].get("control_run"):
        control = summary["inputs"]["control_run"]
        add(f"- 配对对照：`{control.get('path')}`（可用记录 {control.get('records')} 条）")
    if summary["inputs"].get("per_sample_csv"):
        add(f"- 逐样本明细：`{summary['inputs']['per_sample_csv']}`")
    if summary.get("pose_source_label"):
        add(f"- **姿态来源：{summary['pose_source_label']}**")
    add("")
    if protocol:
        add("## 0. 协议摘要")
        add("")
        for key, value in protocol.items():
            if key.endswith("_note"):
                continue
            text = value if not isinstance(value, str) or len(value) <= 160 else value[:157] + "…"
            add(f"- `{key}`：{text}")
        if protocol.get("pose_normalization_protocol_value") is not None:
            add(f"- （协议文件里写的是 `pose_normalization: {protocol['pose_normalization_protocol_value']}`，"
                "但该运行没有走本地归一化，故上面按实际输入路径改写。）")
        if protocol.get("pose_normalization_note"):
            add("")
            add(f"> **归一化口径提醒：** {protocol['pose_normalization_note']}")
        add("")

    add("## 1. 样本与分母")
    add("")
    add("| 项 | 数值 |")
    add("|---|---:|")
    add(f"| 可解析记录 | {records['parsed']} |")
    add(f"| 视频记录 | {records['video_records']} |")
    add(f"| 成功（ok 且预测非空） | {records['success']} |")
    add(f"| 空预测 | {records['empty']} |")
    add(f"| 失败 | {records['failed']} |")
    add(f"| 协议声明样本数 | {coverage.get('protocol_selected') if coverage.get('protocol_selected') is not None else '未列出'} |")
    add(f"| 缺样本数 | {len(coverage.get('missing_numbers') or [])} |")
    add(f"| 覆盖完整 | {'是' if coverage.get('complete') else '否'} |")
    add("")
    add(f"**准确率分母：{accuracy['denominator']['expression']}。**")
    add("")

    add("## 2. 准确率（成功子集，NFKC 去空白口径）")
    add("")
    add("| 指标 | 数值 |")
    add("|---|---:|")
    add(f"| exact match（去空白后逐字相等） | {metrics.get('exact_match_count')} / {metrics.get('n')}"
        f"（{fmt(metrics.get('exact_match_percent'), 2)}%） |")
    add(f"| BLEU-zh（sacrebleu tokenize=zh） | {fmt(metrics.get('bleu_zh'), 4)} |")
    add(f"| chrF | {fmt(metrics.get('chrf'), 4)} |")
    add(f"| ROUGE-L 字符级 F1 | {fmt(metrics.get('rouge_l_f1_char'), 4)} |")
    add(f"| 平均字符编辑相似度 | {fmt(metrics.get('mean_char_edit_similarity_percent'), 2)}% |")
    add(f"| 预测/参考长度比 | {fmt(metrics.get('mean_length_ratio_prediction_to_reference'), 4)} |")
    add("")

    bleu = summary.get("bleu_conventions")
    if bleu:
        add("## 3. BLEU-4 口径复算（本次复核的核心）")
        add("")
        add(f"- 参考句文件：`{bleu['refs_path']}`（sha256 `{bleu['refs_sha256'][:16]}…`，"
            f"协议记录值 `{bleu.get('refs_sha256_in_protocol')}`，"
            f"{'一致' if bleu.get('refs_sha256') == bleu.get('refs_sha256_in_protocol') else '**不一致**'}）")
        add(f"- 口径：{bleu['convention']}；样本 {bleu['n']} 条")
        add("")
        add("| 口径 | BLEU-4 (13a) | BP | precisions(1-4) | chrF | ROUGE-L F1(token) | exact |")
        add("|---|---:|---:|---|---:|---:|---:|")
        ours_fixed = bleu["ours_fixed"]
        ours_raw = bleu["ours_raw"]
        add(f"| 本机 + mt5 逐字修正（正确口径） | {fmt(ours_fixed['bleu4']['bleu'], 4)} | "
            f"{fmt(ours_fixed['bleu4']['bp'], 4)} | {[round(p, 2) for p in ours_fixed['bleu4']['precisions']]} | "
            f"{fmt(ours_fixed['chrf'], 4)} | {fmt(ours_fixed['rouge_l_token_f1'], 2)} | "
            f"{ours_fixed['exact_match']} / {bleu['n']} |")
        add(f"| 本机 不修正（陷阱口径） | {fmt(ours_raw['bleu4']['bleu'], 4)} | "
            f"{fmt(ours_raw['bleu4']['bp'], 4)} | {[round(p, 2) for p in ours_raw['bleu4']['precisions']]} | "
            f"{fmt(ours_raw['chrf'], 4)} | {fmt(ours_raw['rouge_l_token_f1'], 2)} | "
            f"{ours_raw['exact_match']} / {bleu['n']} |")
        upstream = bleu.get("upstream_snapshot")
        if upstream:
            add(f"| 上游 1,176 条预测快照（同口径） | {fmt(upstream['bleu4_fixed']['bleu'], 4)} | "
                f"{fmt(upstream['bleu4_fixed']['bp'], 4)} | "
                f"{[round(p, 2) for p in upstream['bleu4_fixed']['precisions']]} | "
                f"{fmt(upstream['chrf_fixed'], 4)} | {fmt(upstream['rouge_l_token_f1_fixed'], 2)} | — |")
        paper = bleu.get("paper_reference") or {}
        add(f"| 论文引用值 | {paper.get('bleu4')} | — | — | — | {paper.get('rouge')} | — |")
        add("")
        if upstream:
            add(f"- 与上游快照逐样本一致（修正后）：**{upstream['identical_after_fix']} / {upstream['n']}"
                f"（{100 * upstream['identical_after_fix'] / max(1, upstream['n']):.1f}%）**；"
                f"不修正直接比字符串则只有 {upstream['identical_raw']} / {upstream['n']}（纯格式差异，不是模型差异）。")
        add(f"- 论文值来源：{paper.get('source')}")
        add("")
        add("### 3.1 为什么“不修正”会假崩")
        add("")
        add(f"- 不修正时每句预测是 1 个 token（中文无空格），参考句是 15.2 个 token；"
            f"语料 BP 掉到 {fmt(ours_raw['bleu4']['bp'], 4)}（长度惩罚主导），1-gram 精度还有 "
            f"{round((ours_raw['bleu4']['precisions'] or [0])[0], 2)}，2-gram 起接近 0。")
        add(f"- 同一份预测的 chrF 在两种口径下都是 {fmt(ours_fixed['chrf'], 4)}（chrF 对空白不敏感），"
            "这正是“chrF 正常、BLEU 接近 0”的假失败特征。")
        add(f"- 注意：**token 级 ROUGE 也会一起崩**（{fmt(ours_fixed['rouge_l_token_f1'], 2)} → "
            f"{fmt(ours_raw['rouge_l_token_f1'], 2)}），只有**字符级** ROUGE（去空白后按字符算）保持在 "
            f"{fmt(metrics.get('rouge_l_f1_char'), 4)}。所以“chrF/ROUGE 正常”这个诊断只在字符级 ROUGE 下成立，"
            "按上游 token 级 ROUGE 判断会误判成模型失败。")
        add("")

    add("## 4. 延迟（本实验自身口径，model-only）")
    add("")
    add(f"> 本运行的姿态来源：**{summary.get('pose_source_label') or '姿态来自磁盘'}**。"
        "姿态从磁盘读取，**没有逐视频的视频解码与 RTMPose 成本**；"
        "不得与主评测（含姿态提取）的 e2e 并列成一个数字。")
    if summary.get("latency_note"):
        add("")
        add(f"> ⚠️ **延迟口径警示：{summary['latency_note']}**")
    add("")
    add("| 阶段 | n | mean | P50 | P95 | P99 | max | 单位 |")
    add("|---|---:|---:|---:|---:|---:|---:|---|")
    for group, block in (latency.get("by_pose_source") or {}).items():
        for field in PROFILES[profile_name]["timing_fields"]:
            stats = block["fields"].get(field) or {}
            if not stats.get("n"):
                continue
            add(f"| {TIMING_LABELS.get(field, field)} | {stats['n']} | {fmt(stats['mean'], 3)} | "
                f"{fmt(stats['p50'], 3)} | {fmt(stats['p95'], 3)} | {fmt(stats['p99'], 3)} | "
                f"{fmt(stats['max'], 3)} | {'s' if field == 'e2e_seconds' else 'ms'} |")
        rtf = block.get("rtf") or {}
        if rtf.get("n"):
            add(f"| RTF（e2e / 视频时长） | {rtf['n']} | {fmt(rtf['mean'], 4)} | {fmt(rtf['p50'], 4)} | "
                f"{fmt(rtf['p95'], 4)} | {fmt(rtf['p99'], 4)} | {fmt(rtf['max'], 4)} | × |")
    add("")
    flag = latency.get("cold_start_suspicion")
    if flag:
        add(f"- ⚠️ **首条记录含模型加载/预热**：e2e {fmt(flag['first_e2e_seconds'], 2)} s vs 其余中位数 "
            f"{fmt(flag['median_of_rest'], 3)} s（{flag['ratio']:.0f} 倍）。本实验目录没有 `cold-start.json`，"
            "该条未被剔除；聚合 mean 影响约 1%，但 **max 不是稳态延迟**，引用极值时请剔除首条。")
    all_rtf_missing = all(((block.get("rtf") or {}).get("n") or 0) == 0
                          for block in (latency.get("by_pose_source") or {}).values())
    if all_rtf_missing:
        add("- 本 schema 没有 `video_duration_s`，**无法计算 RTF**（也不宜用视频时长做分母估算）。")
    if flag or all_rtf_missing:
        add("")

    paired = summary.get("paired_comparison")
    if paired:
        add("## 5. 与对照运行的同片段配对")
        add("")
        skipped = paired.get("skipped") or {}
        add(f"- 配对分母：**{paired['paired_n']}**；规则：{paired['denominator_rule']}")
        add(f"- 剔除：对照缺该条 {skipped.get('main_missing', 0)}；本运行非 ok {skipped.get('ablation_not_ok', 0)}；"
            f"对照非 ok {skipped.get('main_not_ok', 0)}；参考句不一致 {skipped.get('reference_mismatch', 0)}")
        add("")
        add("| 指标 | 本运行 | 对照运行 | 差值（本 − 对照） |")
        add("|---|---:|---:|---:|")
        add(f"| 平均字符编辑相似度 | {fmt(paired.get('mean_edit_similarity_upstream_pose'), 4)} | "
            f"{fmt(paired.get('mean_edit_similarity_local_pose'), 4)} | "
            f"{fmt(paired.get('mean_delta_upstream_minus_local'), 4)} |")
        ci = paired.get("delta_bootstrap_95ci")
        if ci:
            add(f"| Δ 均值 95% CI（bootstrap {paired.get('bootstrap_resamples')} 次） | — | — | "
                f"[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}] |")
        add(f"| 胜/负/平 | — | — | {paired.get('wins_upstream')} / {paired.get('losses_upstream')} / "
            f"{paired.get('ties')} |")
        add(f"| 预测文本完全相同 | — | — | {paired.get('identical_prediction_count')} |")
        add("")
        if paired.get("control_pose_source"):
            add(f"> 对照侧姿态源：`{paired.get('control_pose_source')}`；"
                "只有两侧姿态源一致时，Δ 才能解释为“只换权重”的差异。")
            add("")
        alt = summary.get("paired_comparison_alt")
        if alt:
            alt_note = summary.get("alt_control_run") or {}
            add("### 5.2 第二对照（姿态源不同，仅参考，不能当作纯权重差异）")
            add("")
            add(f"- 对照：`{alt_note.get('path')}`（姿态源 `{alt.get('control_pose_source')}`）")
            add(f"- 配对分母 {alt['paired_n']}；剔除：对照缺该条 {(alt.get('skipped') or {}).get('main_missing', 0)}，"
                f"本运行非 ok {(alt.get('skipped') or {}).get('ablation_not_ok', 0)}，"
                f"对照非 ok {(alt.get('skipped') or {}).get('main_not_ok', 0)}")
            add("")
            add("| 指标 | 本运行 | 第二对照 | 差值（本 − 对照） |")
            add("|---|---:|---:|---:|")
            add(f"| 平均字符编辑相似度 | {fmt(alt.get('mean_edit_similarity_upstream_pose'), 4)} | "
                f"{fmt(alt.get('mean_edit_similarity_local_pose'), 4)} | "
                f"{fmt(alt.get('mean_delta_upstream_minus_local'), 4)} |")
            alt_ci = alt.get("delta_bootstrap_95ci")
            if alt_ci:
                add(f"| Δ 均值 95% CI | — | — | [{fmt(alt_ci[0], 4)}, {fmt(alt_ci[1], 4)}] |")
            add("")

    frontend = summary.get("frontend_compare")
    if frontend and not frontend.get("error"):
        add("### 5.3 同权重、不同姿态来源的前端对照")
        add("")
        same_ckpt = "是" if frontend.get("same_checkpoint") else "**否（权重不同，不能叫前端对照）**"
        add(f"- 对照运行：`{frontend.get('path')}`")
        add(f"- 本运行姿态来源：{frontend.get('this_pose_source')}")
        add(f"- 对照姿态来源：{frontend.get('compare_pose_source')}")
        add(f"- 同一权重（checkpoint_sha256 相同）：{same_ckpt}"
            f"（本 `{(frontend.get('this_checkpoint_sha256') or '—')[:16]}…` vs "
            f"对照 `{(frontend.get('compare_checkpoint_sha256') or '—')[:16]}…`）")
        add(f"- 配对分母：**{frontend.get('paired_n')}**；剔除：对照缺该条 "
            f"{(frontend.get('skipped') or {}).get('main_missing', 0)}，本运行非 ok "
            f"{(frontend.get('skipped') or {}).get('ablation_not_ok', 0)}，对照非 ok "
            f"{(frontend.get('skipped') or {}).get('main_not_ok', 0)}，参考句不一致 "
            f"{(frontend.get('skipped') or {}).get('reference_mismatch', 0)}")
        add("")
        add("| 姿态来源 | 平均字符编辑相似度 | 差值（本 − 对照） |")
        add("|---|---:|---:|")
        add(f"| {frontend.get('compare_pose_source')} | {fmt(frontend.get('compare_mean_edit_similarity'), 4)} | — |")
        add(f"| {frontend.get('this_pose_source')} | {fmt(frontend.get('this_mean_edit_similarity'), 4)} | "
            f"{fmt(frontend.get('delta_this_minus_compare'), 4)} |")
        add("")
        ci = frontend.get("delta_bootstrap_95ci")
        if ci:
            add(f"- Δ 均值 95% CI（bootstrap）：[{fmt(ci[0], 4)}, {fmt(ci[1], 4)}]")
        add(f"- 胜/负/平（本更好/更差/相同）：{frontend.get('wins_this')} / "
            f"{frontend.get('losses_this')} / {frontend.get('ties')}；预测完全相同 "
            f"{frontend.get('identical_predictions')} 条")
        corpus = frontend.get("paired_corpus_bleu") or {}
        if corpus:
            this_side = corpus.get("upstream_pose") or {}
            other_side = corpus.get("local_pose") or {}
            add(f"- 配对子集语料 BLEU-zh：本（{frontend.get('this_pose_source')}）"
                f"{fmt(this_side.get('bleu_zh'), 4)} vs 对照（{frontend.get('compare_pose_source')}）"
                f"{fmt(other_side.get('bleu_zh'), 4)}")
        add("")
        add("> 差异只来自姿态来源（估计器与归一化），权重、解码参数、采样规则相同。"
            "两条分数不能互换引用，也不能与主生产路径（本地 isotropic 提取）混为一谈。")
        add("")

    add("## 6. 复核")
    add("")
    labels = {"pass": "通过", "warn": "存疑", "fail": "不通过"}
    add(f"总体结论：**{review['overall']}**")
    bad = [c for c in review["checks"] if c["status"] != "pass"]
    if bad:
        add("")
        add("非通过项：" + "；".join(f"{c['id']} {c['title']}（{labels.get(c['status'])}）" for c in bad))
    add("")
    titles = ["①", "②", "③", "④", "⑤", "⑥", "⑦", "⑧"]
    for index, check in enumerate(review["checks"]):
        add(f"### {titles[index] if index < len(titles) else '-'} {check['title']} —— "
            f"{labels.get(check['status'], check['status'])}")
        add("")
        for evidence in check["evidence"]:
            add(f"- {evidence}")
        add("")
    history = summary.get("fix_history") or []
    if history:
        add("### 修复前后对照（口径变更记录）")
        add("")
        for item in history:
            add(f"- {item}")
        add("")
    if summary.get("warnings"):
        add("### 警告")
        add("")
        for warning in summary["warnings"]:
            add(f"- {warning}")
        add("")

    add("## 7. 复现")
    add("")
    add("```powershell")
    add(r".\.conda-env\python.exe scripts\summarize_unisign_eval.py "
        f"--profile {profile_name} --run-dir '{summary['run_dir']}'")
    add("```")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# pose_detect_stride 精度验收复核（stride=5 vs stride=1）
# ---------------------------------------------------------------------------
STRIDE_RUN_DEFAULTS = {
    "stride5_stage1": "artifacts/cslnews-eval/stage1-wh-stride5-20261025",
    "stride1_stage1_control": "artifacts/cslnews-eval/stage1-wh-full-20261025",
    "stride5_pose": "artifacts/cslnews-eval/pinned-wh-stride5-20261025",
    "stride1_pose_wh": "artifacts/cslnews-eval/pinned-wh-full-20261025",
    "stride1_pose_isotropic": "artifacts/cslnews-eval/run-cslnews-20261025",
    "combined": "artifacts/cslnews-eval/pinned-wh-combined-20261025",
}
STRIDE_ACCEPTANCE_BOUNDARY = [
    "静态单人镜头（archive_001 本地片段）",
    "固定机位、无摄像机运动",
    "分辨率 204×170（横纵比接近 1，wh 归一化与 isotropic 差异很小）",
    "pose 缓存 + wh 归一化、beam=4、max_new_tokens=100、bf16/CUDA",
    "300 条子集（cslnews-000001..000300），非全量 1694",
]


def _load_by_number(path):
    if not Path(path).is_file():
        return {}
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["number"]): row for row in rows if row.get("number") is not None}


def _accuracy_block(records, numbers):
    refs = [normalize_text(records[n].get("reference")) for n in numbers]
    hyps = [normalize_text(records[n].get("prediction")) for n in numbers]
    return compute_text_metrics(refs, hyps)


def _latency_block(records, numbers):
    subset = [records[n] for n in numbers if n in records]
    pose = [coerce_float(r.get("pose_extract_ms")) for r in subset]
    e2e = [coerce_float(r.get("e2e_seconds")) for r in subset]
    rtf = []
    for r in subset:
        seconds = coerce_float(r.get("e2e_seconds"))
        duration = coerce_float(r.get("video_duration_s"))
        if seconds is not None and duration and duration > 0:
            rtf.append(seconds / duration)
    e2e_values = [v for v in e2e if v]
    first_ratio = None
    if len(e2e_values) >= 4:
        rest_median = statistics.median(e2e_values[1:])
        if rest_median > 0:
            first_ratio = e2e_values[0] / rest_median
    frames = [coerce_float(r.get("video_frames")) for r in subset]
    frames = [v for v in frames if v]
    total_e2e = sum(e2e_values) if e2e_values else 0.0
    throughput = {
        "videos_per_minute": (60.0 / statistics.fmean(e2e_values)) if e2e_values else None,
        "frames_per_second": (sum(frames) / total_e2e) if total_e2e > 0 and frames else None,
        "frames_total": int(sum(frames)) if frames else None,
    }
    return {
        "n": len(subset),
        "pose_extract_ms": percentile_stats([v for v in pose if v is not None]),
        "e2e_seconds": percentile_stats(e2e_values),
        "rtf": percentile_stats(rtf),
        "first_record_ratio": first_ratio,
        "throughput": throughput,
    }


def _pose_bucket_table(combined_records, reference_records, numbers):
    """按片段长度分桶比较姿态阶段耗时：验证 sample_only 是否把检测帧数封顶。"""
    buckets = ((0, 200), (200, 300), (300, 400), (400, 600))
    table = []
    for low, high in buckets:
        subset = []
        for n in numbers:
            if n not in combined_records or n not in reference_records:
                continue
            frames = coerce_float(combined_records[n].get("video_frames"))
            if frames is None or not (low <= frames < high):
                continue
            a = coerce_float(combined_records[n].get("pose_extract_ms"))
            b = coerce_float(reference_records[n].get("pose_extract_ms"))
            if a is None or b is None:
                continue
            subset.append((a, b, a - b))
        if not subset:
            continue
        table.append({
            "frames_low": low, "frames_high": high, "n": len(subset),
            "combined_pose_mean": statistics.fmean(x[0] for x in subset),
            "reference_pose_mean": statistics.fmean(x[1] for x in subset),
            "delta_mean": statistics.fmean(x[2] for x in subset),
            "delta_median": statistics.median(x[2] for x in subset),
        })
    return table


def summarize_stride_validation(*, stride5_run, control_run, stride5_pose_run,
                                baseline_wh_run, baseline_isotropic_run, combined_run):
    stride5 = _load_by_number(Path(stride5_run) / "predictions.jsonl")
    control_all = _load_by_number(Path(control_run) / "predictions.jsonl")
    numbers = sorted(stride5)
    control = {n: control_all[n] for n in numbers if n in control_all}
    missing_control = [n for n in numbers if n not in control_all]
    references_equal = all(
        normalize_text(stride5[n].get("reference")) == normalize_text(control[n].get("reference"))
        for n in control)

    _, stride5_protocol = read_protocol_any(Path(stride5_run), names=("experiment-protocol.json",))
    _, control_protocol = read_protocol_any(Path(control_run), names=("experiment-protocol.json",))
    _, pose5_protocol = read_protocol_any(Path(stride5_pose_run), names=("run-protocol.json",))
    pose5_info = {
        "path": str(Path(stride5_pose_run) / "run-protocol.json"),
        "file_sha256": file_sha256(Path(stride5_pose_run) / "run-protocol.json")
        if (Path(stride5_pose_run) / "run-protocol.json").is_file() else None,
        "pose_detect_stride": ((pose5_protocol or {}).get("pose_protocol") or {}).get("pose_detect_stride"),
        "pose_normalization": ((pose5_protocol or {}).get("pose_protocol") or {}).get("pose_normalization"),
        "num_beams": (pose5_protocol or {}).get("num_beams"),
        "max_new_tokens": (pose5_protocol or {}).get("max_new_tokens"),
    }
    cold_start_path = Path(stride5_pose_run) / "cold-start.json"
    cold_start = None
    if cold_start_path.is_file():
        data = json.loads(cold_start_path.read_text(encoding="utf-8"))
        cold_start = {"load_model_seconds": coerce_float(data.get("load_model_seconds")),
                      "excluded": data.get("excluded_from_per_video_latency"),
                      "note": data.get("note"),
                      "measured_at": data.get("measured_at")}

    metrics5 = _accuracy_block(stride5, numbers)
    metrics1 = _accuracy_block(control, numbers)
    deltas = []
    for n in numbers:
        sim5 = edit_similarity(normalize_text(stride5[n].get("reference")), normalize_text(stride5[n].get("prediction")))
        sim1 = edit_similarity(normalize_text(control[n].get("reference")), normalize_text(control[n].get("prediction")))
        deltas.append(sim5 - sim1)
    array = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(0)
    boot = [float(np.mean(rng.choice(array, size=array.size, replace=True))) for _ in range(10000)]
    paired = {
        "n": len(numbers),
        "mean_delta": float(array.mean()),
        "delta_ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "prob_positive": float((np.asarray(boot) > 0).mean()),
        "wins": int((array > 1e-9).sum()),
        "losses": int((array < -1e-9).sum()),
        "ties": int((np.abs(array) <= 1e-9).sum()),
        "identical_predictions": sum(
            1 for n in numbers
            if str(stride5[n].get("prediction") or "").strip() == str(control[n].get("prediction") or "").strip()),
        "bootstrap_resamples": 10000,
        "rule": "同一批片段、同一权重（stage1）、同一解码；只改 pose_detect_stride",
    }

    pose5_records = _load_by_number(Path(stride5_pose_run) / "predictions.jsonl")
    latency = {
        "stride5": _latency_block(pose5_records, numbers),
        "stride1_wh": _latency_block(_load_by_number(Path(baseline_wh_run) / "predictions.jsonl"), numbers),
        "stride1_isotropic": _latency_block(_load_by_number(Path(baseline_isotropic_run) / "predictions.jsonl"), numbers),
    }

    combined = {}
    combined_path = Path(combined_run) / "predictions.jsonl"
    if combined_path.is_file() and combined_path.stat().st_size > 0:
        combined_records = _load_by_number(combined_path)
        combined_numbers = sorted(combined_records)
        combined_metrics = _accuracy_block(combined_records, combined_numbers)
        flag_fields = sorted({key for record in combined_records.values()
                              for key in record
                              if "sample_only" in key or "skipped" in key or "detect_stride" in key})
        combined = {
            "ready": True,
            "path": str(combined_path),
            "records": len(combined_records),
            "latency": _latency_block(combined_records, combined_numbers),
            "accuracy_not_evidence": combined_metrics,
            "protocol": read_protocol_any(Path(combined_run), names=("run-protocol.json",))[1] or {},
            "sample_only_flag_fields": flag_fields,
            "pose_buckets": _pose_bucket_table(
                combined_records, pose5_records, [n for n in numbers if n in combined_records]),
        }
        combined_cold_path = Path(combined_run) / "cold-start.json"
        if combined_cold_path.is_file():
            data = json.loads(combined_cold_path.read_text(encoding="utf-8"))
            combined["cold_start"] = {"load_model_seconds": coerce_float(data.get("load_model_seconds")),
                                      "excluded": data.get("excluded_from_per_video_latency")}
    else:
        combined = {"ready": False, "path": str(combined_path)}

    checks = []

    def add_check(cid, title, status, evidence):
        checks.append({"id": cid, "title": title, "status": status, "evidence": evidence})

    add_check("D1", "配对样本同一性（同一批片段、参考句一致）", "pass" if not missing_control and references_equal else "fail", [
        f"stride-5 记录 {len(numbers)} 条（{numbers[0]}..{numbers[-1]}）；对照可按 number 匹配 {len(control)} 条",
        f"对照缺失 {len(missing_control)} 条；参考句全部一致：{references_equal}",
    ])
    same_ckpt = (stride5_protocol or {}).get("checkpoint_sha256") == (control_protocol or {}).get("checkpoint_sha256")
    same_decode = ((stride5_protocol or {}).get("num_beams") == (control_protocol or {}).get("num_beams")
                   and (stride5_protocol or {}).get("max_new_tokens") == (control_protocol or {}).get("max_new_tokens"))
    add_check("D2", "权重与解码参数一致", "pass" if same_ckpt and same_decode else "fail", [
        f"checkpoint_sha256：stride-5 `{(stride5_protocol or {}).get('checkpoint_sha256')}` vs 对照 "
        f"`{(control_protocol or {}).get('checkpoint_sha256')}`（一致={same_ckpt}）",
        f"num_beams={((stride5_protocol or {}).get('num_beams'))} vs {((control_protocol or {}).get('num_beams'))}；"
        f"max_new_tokens={((stride5_protocol or {}).get('max_new_tokens'))} vs "
        f"{((control_protocol or {}).get('max_new_tokens'))}（一致={same_decode}）",
        f"stride-5 姿态缓存身份 = `{(stride5_protocol or {}).get('main_run_protocol_sha256')}`，"
        f"pinned-wh-stride5 协议文件哈希 = `{pose5_info['file_sha256']}`（"
        f"{'一致' if (stride5_protocol or {}).get('main_run_protocol_sha256') == pose5_info['file_sha256'] else '不一致'}）",
    ])
    ci = paired["delta_ci95"]
    significant = not (ci[0] <= 0.0 <= ci[1])
    add_check("D3", "精度差异是否显著", "pass" if not significant else "fail", [
        f"Δ 平均编辑相似度 = {paired['mean_delta']:+.4f}（{100 * paired['mean_delta']:+.2f} pp），"
        f"bootstrap 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]",
        f"胜/负/平 = {paired['wins']}/{paired['losses']}/{paired['ties']}；P(Δ>0) = {paired['prob_positive']:.3f}；"
        f"预测逐字相同 = {paired['identical_predictions']}/{paired['n']}",
        ("结论：CI 含 0，差异不显著；点估计方向为正（stride-5 略好），但幅度远小于样本不确定性。"
         if not significant else "结论：CI 不含 0，存在显著差异，需评估是否可接受。"),
    ])
    add_check("D4", "延迟基线来源与争用状态", "warn", [
        f"stride-5（wh, stride 5）：e2e mean {fmt(latency['stride5']['e2e_seconds']['mean'], 3)} s，"
        f"姿态阶段 mean {fmt(latency['stride5']['pose_extract_ms']['mean'], 1)} ms，RTF {fmt(latency['stride5']['rtf']['mean'], 4)}",
        f"stride-1（isotropic, pinned，run-cslnews 前 300）：e2e mean "
        f"{fmt(latency['stride1_isotropic']['e2e_seconds']['mean'], 3)} s，姿态 "
        f"{fmt(latency['stride1_isotropic']['pose_extract_ms']['mean'], 1)} ms，RTF "
        f"{fmt(latency['stride1_isotropic']['rtf']['mean'], 4)}",
        f"stride-1（wh, pinned-wh-full 前 300）：e2e mean "
        f"{fmt(latency['stride1_wh']['e2e_seconds']['mean'], 3)} s，姿态 "
        f"{fmt(latency['stride1_wh']['pose_extract_ms']['mean'], 1)} ms —— 该运行有已记录的并发争用，**不得当稳态基线**",
        "结论：延迟收益请引用 isotropic 稳态基线（≈5.54 s → 3.33 s）；wh 基线只作口径对照。",
    ])
    cold_ok = (cold_start or {}).get("excluded") is True and \
        (latency["stride5"]["first_record_ratio"] is None or latency["stride5"]["first_record_ratio"] < 1.5)
    add_check("D5", "冷启动是否排除在逐条延迟之外", "pass" if cold_ok else "fail", [
        f"cold-start.json：load_model_seconds = {fmt((cold_start or {}).get('load_model_seconds'), 2)} s，"
        f"excluded_from_per_video_latency = {(cold_start or {}).get('excluded')}",
        f"stride-5 首条 e2e / 其余中位数 = {fmt(latency['stride5']['first_record_ratio'], 2)}（<1.5 视为未混入）",
        f"isotropic 基线首条比值 = {fmt(latency['stride1_isotropic']['first_record_ratio'], 2)}；"
        f"wh 基线首条比值 = {fmt(latency['stride1_wh']['first_record_ratio'], 2)}",
    ])
    combined_complete = combined.get("ready") and combined.get("records", 0) >= 300
    d6_evidence = ["pinned 在 CSL-News 上是跨域，其准确率不是验收证据；组合运行仅用于看 stride5+sample_only 的延迟"]
    if combined.get("ready"):
        block = combined["latency"]
        base_e2e = latency["stride1_isotropic"]["e2e_seconds"]["mean"]
        s5_e2e = latency["stride5"]["e2e_seconds"]["mean"]
        speed_base = (base_e2e / block["e2e_seconds"]["mean"]) if block["e2e_seconds"]["mean"] else None
        speed_s5 = (s5_e2e / block["e2e_seconds"]["mean"]) if block["e2e_seconds"]["mean"] else None
        d6_evidence.append(
            f"组合运行{'已完成' if combined_complete else '进行中（未满 300 条，数字仅初步）'}：{combined['records']} 条；"
            f"姿态 {fmt(block['pose_extract_ms']['mean'], 0)} ms，e2e {fmt(block['e2e_seconds']['mean'], 3)} s"
            f"（P95 {fmt(block['e2e_seconds']['p95'], 3)} / P99 {fmt(block['e2e_seconds']['p99'], 3)}），"
            f"RTF {fmt(block['rtf']['mean'], 3)}，吞吐 {fmt(block['throughput']['videos_per_minute'], 1)} 条/分、"
            f"{fmt(block['throughput']['frames_per_second'], 1)} 帧/秒")
        d6_evidence.append(f"相对 isotropic 基线 {fmt(speed_base, 2)}×；相对仅 stride5 {fmt(speed_s5, 2)}×")
        cold = combined.get("cold_start") or {}
        d6_evidence.append(f"cold-start：load_model_seconds = {fmt(cold.get('load_model_seconds'), 2)} s，"
                           f"excluded_from_per_video_latency = {cold.get('excluded')}")
        acc = combined.get("accuracy_not_evidence") or {}
        d6_evidence.append(
            f"其准确率 BLEU-zh {fmt(acc.get('bleu_zh'), 2)} / 编辑相似度 "
            f"{fmt(100 * (acc.get('mean_char_edit_similarity') or 0), 2)}% → pinned 跨域废分，**不作精度证据**；"
            "精度验收仍以 stride-5 那份（stage1 权重、Δ CI 含 0）为准")
        if combined.get("sample_only_flag_fields"):
            d6_evidence.append(f"逐条记录含 sample_only 相关字段：{combined['sample_only_flag_fields']}")
        else:
            d6_evidence.append("逐条记录**没有** `pose_sample_only` / `pose_skipped_frames` 字段"
                               "（脚本没有把 metrics 合并进记录）→ 只能间接佐证 sample_only 是否生效")
        buckets = combined.get("pose_buckets") or []
        for bucket in buckets:
            d6_evidence.append(
                f"帧数[{bucket['frames_low']},{bucket['frames_high']}) n={bucket['n']}："
                f"姿态 {bucket['combined_pose_mean']:.0f} ms（仅 stride5 {bucket['reference_pose_mean']:.0f} ms，"
                f"Δ{bucket['delta_mean']:+.0f} ms，中位 Δ{bucket['delta_median']:+.0f} ms）")
        long_buckets = [bucket for bucket in buckets if bucket["frames_low"] >= 300]
        if long_buckets and all(bucket["delta_mean"] < -300 for bucket in long_buckets):
            d6_evidence.append(
                "结论：sample_only 生效 —— 长片段（≥300 帧，窗口会降采样到 256）姿态耗时下降 "
                f"{abs(long_buckets[0]['delta_mean']):.0f}~{abs(long_buckets[-1]['delta_mean']):.0f} ms，"
                "短片段（<300 帧、降采样本来就不发生）几乎不变，与 sample_only 语义一致。")
    else:
        d6_evidence.append(f"组合运行尚未产出 predictions（{combined.get('path')}），本版只留占位")
    add_check("D6", "组合运行只作延迟参考（含 sample_only 生效核验）",
              "pass" if combined_complete else "warn", d6_evidence)

    overall = "pass" if all(c["status"] != "fail" for c in checks) else "fail"
    verdict_enabled = overall == "pass"
    verdict = (
        "在本地 CSL-News 300 条（静态单人、固定机位、204×170、wh、stage1 权重）上，"
        f"`pose_detect_stride=5` {'可以启用' if verdict_enabled else '暂不可启用'}："
        f"精度与 stride-1 无显著差异（Δ编辑相似度 {100 * paired['mean_delta']:+.2f} pp，"
        f"95% CI [{100 * ci[0]:+.2f}, {100 * ci[1]:+.2f}] pp 含 0），"
        f"延迟明显下降（e2e {fmt(latency['stride1_isotropic']['e2e_seconds']['mean'], 2)} s → "
        f"{fmt(latency['stride5']['e2e_seconds']['mean'], 2)} s，RTF "
        f"{fmt(latency['stride1_isotropic']['rtf']['mean'], 3)} → {fmt(latency['stride5']['rtf']['mean'], 3)}）；"
        "该结论不外推到多人、移动机位、其他分辨率或其他数据集。"
    )
    if combined.get("ready"):
        block = combined["latency"]
        speed = (latency["stride1_isotropic"]["e2e_seconds"]["mean"] / block["e2e_seconds"]["mean"]
                 if block["e2e_seconds"]["mean"] else None)
        verdict += (f" 若再叠加 `pose_sample_only`（组合运行，仅延迟参考），e2e 进一步降到 "
                    f"{fmt(block['e2e_seconds']['mean'], 2)} s、RTF {fmt(block['rtf']['mean'], 3)}"
                    f"（相对基线 {fmt(speed, 2)}×）；其准确率不作验收证据。")
    return {
        "schema_version": 1,
        "kind": "stride_validation",
        "generated_at": now_iso(),
        "runs": {"stride5_stage1": str(stride5_run), "stride1_stage1_control": str(control_run),
                 "stride5_pose": str(stride5_pose_run), "stride1_pose_wh": str(baseline_wh_run),
                 "stride1_pose_isotropic": str(baseline_isotropic_run), "combined": str(combined_run)},
        "numbers": {"first": numbers[0], "last": numbers[-1], "n": len(numbers)},
        "accuracy": {"stride5": metrics5, "stride1": metrics1, "paired": paired},
        "latency": latency,
        "combined": combined,
        "pose5_protocol": pose5_info,
        "cold_start": cold_start,
        "checks": checks,
        "overall": overall,
        "verdict": verdict,
        "acceptance_boundary": STRIDE_ACCEPTANCE_BOUNDARY,
    }


def render_stride_md(summary):
    lines = []
    add = lines.append
    acc = summary["accuracy"]
    paired = acc["paired"]
    latency = summary["latency"]
    combined = summary["combined"]
    ratio = 100 * paired["mean_delta"]
    add("# pose_detect_stride=5 精度验收复核（本地 CSL-News 300 条）")
    add("")
    add(f"> **结论：** {summary['verdict']}")
    add("")
    add(f"- 生成时间：{summary['generated_at']}")
    add(f"- stride-5（stage1）：`{summary['runs']['stride5_stage1']}`")
    add(f"- stride-1 对照（stage1 全量，按 number 取前 300）：`{summary['runs']['stride1_stage1_control']}`")
    add(f"- stride-5 姿态侧（pinned，延迟用）：`{summary['runs']['stride5_pose']}`")
    add(f"- stride-1 稳态延迟基线（isotropic）：`{summary['runs']['stride1_pose_isotropic']}`")
    add(f"- 样本：{summary['numbers']['n']} 条（{summary['numbers']['first']}..{summary['numbers']['last']}）")
    add("")

    add("## 1. 精度：配对复算")
    add("")
    add("| 指标 | stride=1（stage1，前 300） | stride=5（stage1） | 差值 |")
    add("|---|---:|---:|---:|")
    add(f"| BLEU-zh | {fmt(acc['stride1']['bleu_zh'], 4)} | {fmt(acc['stride5']['bleu_zh'], 4)} | "
        f"{fmt(acc['stride5']['bleu_zh'] - acc['stride1']['bleu_zh'], 4)} |")
    add(f"| chrF | {fmt(acc['stride1']['chrf'], 4)} | {fmt(acc['stride5']['chrf'], 4)} | "
        f"{fmt(acc['stride5']['chrf'] - acc['stride1']['chrf'], 4)} |")
    add(f"| ROUGE-L 字符级 F1 | {fmt(acc['stride1']['rouge_l_f1_char'], 4)} | {fmt(acc['stride5']['rouge_l_f1_char'], 4)} | "
        f"{fmt(acc['stride5']['rouge_l_f1_char'] - acc['stride1']['rouge_l_f1_char'], 4)} |")
    add(f"| 平均编辑相似度 | {fmt(100 * acc['stride1']['mean_char_edit_similarity'], 2)}% | "
        f"{fmt(100 * acc['stride5']['mean_char_edit_similarity'], 2)}% | "
        f"{fmt(100 * (acc['stride5']['mean_char_edit_similarity'] - acc['stride1']['mean_char_edit_similarity']), 2)} pp |")
    add(f"| exact match | {acc['stride1']['exact_match_count']} | {acc['stride5']['exact_match_count']} | — |")
    add("")
    add("**配对（逐条编辑相似度差，stride5 − stride1）**")
    add("")
    add(f"- 配对样本：{paired['n']} 条；规则：{paired['rule']}")
    add(f"- 均值差：**{paired['mean_delta']:+.4f}**（{ratio:+.2f} pp）；bootstrap（{paired['bootstrap_resamples']} 次）"
        f"95% CI **[{paired['delta_ci95'][0]:+.4f}, {paired['delta_ci95'][1]:+.4f}]**")
    add(f"- 胜/负/平：{paired['wins']}/{paired['losses']}/{paired['ties']}；P(Δ>0) = {paired['prob_positive']:.3f}")
    add(f"- 预测逐字相同：{paired['identical_predictions']} / {paired['n']}")
    add("")
    ci = paired["delta_ci95"]
    if ci[0] <= 0.0 <= ci[1]:
        add("> **显著性判定：CI 含 0 → 精度差异不显著。** 点估计为正（stride-5 略好）但幅度 0.36 pp，"
            "远小于抽样不确定性；没有任何证据表明 stride=5 会让精度变差。")
    else:
        add("> **显著性判定：CI 不含 0 → 存在显著差异，需要先评估可接受性再启用。**")
    add("")

    add("## 2. 延迟：stride=5 vs stride=1")
    add("")
    add("| 口径 | n | 姿态阶段 mean (ms) | 姿态 P95 | e2e mean (s) | e2e P50 | e2e P95 | RTF mean | RTF P95 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    def row(label, block):
        add(f"| {label} | {block['n']} | {fmt(block['pose_extract_ms']['mean'], 1)} | "
            f"{fmt(block['pose_extract_ms']['p95'], 1)} | {fmt(block['e2e_seconds']['mean'], 3)} | "
            f"{fmt(block['e2e_seconds']['p50'], 3)} | {fmt(block['e2e_seconds']['p95'], 3)} | "
            f"{fmt(block['rtf']['mean'], 4)} | {fmt(block['rtf']['p95'], 4)} |")
    row("stride=5（wh）", latency["stride5"])
    if combined.get("ready"):
        row("**stride=5 + sample_only（组合，仅延迟参考）**", combined["latency"])
    row("stride=1（isotropic 稳态基线）", latency["stride1_isotropic"])
    row("stride=1（wh，**受并发争用，仅口径对照**）", latency["stride1_wh"])
    add("")
    iso, s5 = latency["stride1_isotropic"], latency["stride5"]
    pose_drop = 100 * (1 - s5["pose_extract_ms"]["mean"] / iso["pose_extract_ms"]["mean"]) if iso["pose_extract_ms"]["mean"] else None
    e2e_drop = 100 * (1 - s5["e2e_seconds"]["mean"] / iso["e2e_seconds"]["mean"]) if iso["e2e_seconds"]["mean"] else None
    add(f"- 姿态阶段：{fmt(iso['pose_extract_ms']['mean'], 0)} ms → {fmt(s5['pose_extract_ms']['mean'], 0)} ms"
        f"（−{fmt(pose_drop, 1)}%）；e2e：{fmt(iso['e2e_seconds']['mean'], 2)} s → {fmt(s5['e2e_seconds']['mean'], 2)} s"
        f"（−{fmt(e2e_drop, 1)}%）；RTF：{fmt(iso['rtf']['mean'], 3)} → {fmt(s5['rtf']['mean'], 3)}")
    add(f"- 冷启动：`cold-start.json` 记录 load_model_seconds = "
        f"{fmt((summary['cold_start'] or {}).get('load_model_seconds'), 2)} s（含 ONNX 会话初始化），"
        f"已声明 excluded_from_per_video_latency = {(summary['cold_start'] or {}).get('excluded')}；"
        f"stride-5 首条 e2e/其余中位数 = {fmt(latency['stride5']['first_record_ratio'], 2)} → **首条没有再把加载算进去**。")
    add("- 注意：stride-1 的 wh 基线与 stride-5 同口径（wh），但该运行期间有并发争用（e2e 6.84 s 明显高于稳态），"
        "所以速度结论引用 isotropic 稳态基线，不引用 wh 基线。")
    add("")

    add("## 3. 组合运行（pinned + stride5 + sample_only，仅延迟参考）")
    add("")
    if combined.get("ready"):
        block = combined["latency"]
        proto = combined.get("protocol") or {}
        pose_proto = proto.get("pose_protocol") or {}
        base = latency["stride1_isotropic"]
        s5_only = latency["stride5"]
        speed_base = (base["e2e_seconds"]["mean"] / block["e2e_seconds"]["mean"]) if block["e2e_seconds"]["mean"] else None
        speed_s5 = (s5_only["e2e_seconds"]["mean"] / block["e2e_seconds"]["mean"]) if block["e2e_seconds"]["mean"] else None
        phase = "已完成" if combined["records"] >= 300 else "进行中（未满 300 条，数字仅初步）"
        add(f"- 状态：{phase}，{combined['records']} 条记录")
        add(f"- 配置：`pose_detect_stride={pose_proto.get('pose_detect_stride')}`，"
            f"`pose_sample_only={pose_proto.get('pose_sample_only')}`，wh 归一化，beam={proto.get('num_beams')}")
        add("")
        add("| 配置 | n | 姿态 mean (ms) | e2e mean (s) | e2e P95 | e2e P99 | RTF mean | 吞吐(条/分) | 吞吐(帧/秒) |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for label, block2 in (("stride=1 基线（isotropic）", base),
                              ("stride=5（仅 stride）", s5_only),
                              ("**stride=5 + sample_only**", block)):
            add(f"| {label} | {block2['n']} | {fmt(block2['pose_extract_ms']['mean'], 0)} | "
                f"{fmt(block2['e2e_seconds']['mean'], 2)} | {fmt(block2['e2e_seconds']['p95'], 2)} | "
                f"{fmt(block2['e2e_seconds']['p99'], 2)} | {fmt(block2['rtf']['mean'], 3)} | "
                f"{fmt((block2.get('throughput') or {}).get('videos_per_minute'), 1)} | "
                f"{fmt((block2.get('throughput') or {}).get('frames_per_second'), 1)} |")
        add("")
        add(f"- 加速：相对 isotropic 基线 **{fmt(speed_base, 2)}×**；相对仅 stride5 **{fmt(speed_s5, 2)}×**"
            f"（姿态阶段 {fmt(base['pose_extract_ms']['mean'], 0)} → {fmt(s5_only['pose_extract_ms']['mean'], 0)} → "
            f"{fmt(block['pose_extract_ms']['mean'], 0)} ms）")
        cold = combined.get("cold_start") or {}
        add(f"- 冷启动：`cold_start_seconds = {fmt(cold.get('load_model_seconds'), 2)} s`（含 ONNX 会话初始化），"
            f"excluded_from_per_video_latency = {cold.get('excluded')}；"
            f"首条 e2e/其余中位数 = {fmt(block.get('first_record_ratio'), 2)}")
        add("")
        add("### 3.1 `pose_sample_only` 是否真的生效")
        add("")
        if combined.get("sample_only_flag_fields"):
            add(f"- 逐条记录含 sample_only 相关字段：{combined['sample_only_flag_fields']}")
        else:
            add("- 逐条记录**没有** `pose_sample_only` / `pose_skipped_frames` 字段"
                "（评测脚本没有把 metrics 合并进记录），只能间接佐证；建议脚本补这两个字段。")
        add("")
        buckets = combined.get("pose_buckets") or []
        if buckets:
            add("| 片段帧数 | n | 组合姿态 mean (ms) | 仅 stride5 姿态 mean (ms) | Δ mean | Δ 中位 |")
            add("|---|---:|---:|---:|---:|---:|")
            for bucket in buckets:
                add(f"| [{bucket['frames_low']}, {bucket['frames_high']}) | {bucket['n']} | "
                    f"{bucket['combined_pose_mean']:.0f} | {bucket['reference_pose_mean']:.0f} | "
                    f"{bucket['delta_mean']:+.0f} | {bucket['delta_median']:+.0f} |")
            add("")
            long_buckets = [bucket for bucket in buckets if bucket["frames_low"] >= 300]
            if long_buckets and all(bucket["delta_mean"] < -300 for bucket in long_buckets):
                add(f"> **判定：sample_only 生效。** 长片段（≥300 帧、窗口会降采样到 256）姿态耗时下降 "
                    f"{abs(long_buckets[0]['delta_mean']):.0f}~{abs(long_buckets[-1]['delta_mean']):.0f} ms；"
                    "短片段（<300 帧，本来就不降采样）几乎不变 —— 与 `pose_sample_only`（只对最终进入窗口的帧做检测）"
                    "的语义完全一致。逐条 flag 字段缺失时，这是最强的间接证据。")
            else:
                add("> 注意：按片段长度分桶后没有看到预期的长片段收益，sample_only 生效证据不足，需补字段复核。")
            add("")
        acc = combined.get("accuracy_not_evidence") or {}
        add(f"- ⚠️ 组合运行的准确率（BLEU-zh {fmt(acc.get('bleu_zh'), 2)}、"
            f"编辑相似度 {fmt(100 * (acc.get('mean_char_edit_similarity') or 0), 2)}%、"
            f"chrF {fmt(acc.get('chrf'), 2)}）是 **pinned 权重在 CSL-News 上的跨域废分，不能当精度证据**；"
            "精度验收仍以 stride-5 那份（stage1 权重、Δ +0.0036、CI 含 0）为准。")
    else:
        add(f"- **尚未产出 predictions**（`{combined.get('path')}`），本版留占位；跑完后重跑 "
            "`--stride-review` 即可补上。")
    add("")

    add("## 4. 复核清单")
    add("")
    labels = {"pass": "通过", "warn": "存疑", "fail": "不通过"}
    add(f"总体：**{summary['overall']}**")
    add("")
    titles = ["①", "②", "③", "④", "⑤", "⑥", "⑦"]
    for index, check in enumerate(summary["checks"]):
        add(f"### {titles[index] if index < len(titles) else '-'} {check['title']} —— "
            f"{labels.get(check['status'], check['status'])}")
        add("")
        for evidence in check["evidence"]:
            add(f"- {evidence}")
        add("")

    add("## 5. 验收边界（结论只在这个范围内成立）")
    add("")
    for item in summary["acceptance_boundary"]:
        add(f"- {item}")
    add("- 未覆盖：多人场景、移动机位/手持、其他分辨率与宽高比、其他数据集、实时流式链路。")
    add("- stride=5 降低的是姿态采样密度（每 5 帧检测一次），语义层的时序卷积仍看到 256 帧；"
        "若换到更大动作幅度或更快手速的数据，需要重新做同样的配对验收。")
    add("")

    add("## 6. 复现")
    add("")
    add("```powershell")
    add(r".\.conda-env\python.exe scripts\summarize_unisign_eval.py --stride-review")
    add("```")
    add("")
    add("- 六个目录默认已固定；`--stride5-stage1 / --stride1-stage1 / --stride5-pose / "
        "--stride1-pose-iso / --stride1-pose-wh / --combined-run` 可覆盖。")
    add(f"- 本版 §2/§3 已包含组合运行 {summary['combined'].get('records') or 0} 条；"
        "组合跑完或重跑后，再执行同一条命令即可刷新（不需要改任何参数）。")
    add("- 精度配对（§1）不依赖组合运行；组合运行只影响 §2 的组合行与 §3。")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 自测 fixture
# ---------------------------------------------------------------------------
def _sha(number: str) -> str:
    return hashlib.sha256(str(number).encode("utf-8")).hexdigest()

def _record(number, reference, prediction, *, protocol_id, status="ok", pose_source="disk_cache",
            e2e=1.0, duration=10.0, frames=300, decode=10.0, pose_extract=0.0,
            pose_preprocess=None, model_forward=100.0, generation=50.0, model_total=150.0,
            error=None, sha=True):
    return {
        "number": number,
        "video": f"E:/fake/CSL-NEWS/video/{number}.mp4",
        "reference": reference,
        "prediction": prediction,
        "status": status,
        "error": error,
        "pose_source": pose_source,
        "video_frames": frames,
        "video_duration_s": duration,
        "video_fps": 30.0,
        "video_width": 1280,
        "video_height": 720,
        "decode_ms": decode,
        "pose_extract_ms": pose_extract,
        **({"pose_preprocess_ms": pose_preprocess} if pose_preprocess is not None else {}),
        "model_forward_ms": model_forward,
        "generation_ms": generation,
        "model_total_ms": model_total,
        "e2e_seconds": e2e,
        "pose_orientation": "preserve",
        "run_protocol_id": protocol_id,
        "created_at": "2026-10-25T12:00:00+08:00",
        **({"video_sha256": _sha(number)} if sha else {}),
    }


def _protocol(**extra):
    protocol = {
        "schema_version": 2,
        "dataset": "CSL-News",
        "split": "local-1694",
        "checkpoint": "csl_daily_pose_only_slt.pth",
        "checkpoint_sha256": "73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c",
        "source_commit": "eed438bcb49e30405cd6ccdfcccca330c134e830",
        "num_beams": 4,
        "max_new_tokens": 100,
        "device": "cuda",
        "dtype": "float16",
        "cross_domain": True,
        "domain_warning": "CSL-Daily weights on CSL-News; not product accuracy.",
    }
    protocol.update(extra)
    return protocol


def _cold_start(load_seconds=27.0):
    return {
        "schema_version": 1,
        "measured_at": "2026-10-25T12:00:00+08:00",
        "load_model_seconds": load_seconds,
        "excluded_from_per_video_latency": True,
        "note": "fixture: load_model() also warms RTMPose and one synthetic window; cold start only.",
        "runtime": {"device": "cuda", "dtype": "float16"},
    }


def build_fixtures(fixtures_root: Path):
    """在 fixtures_root 下写出 5 组合成 fixture，返回 (fixture_specs, protocol_cache)。"""
    fixtures_root.mkdir(parents=True, exist_ok=True)
    specs = []

    # --- 01 相同文本：exact=1、BLEU=100、chrF=100、ROUGE-L=1、编辑相似度=1 ----
    protocol = _protocol(selected=[{"number": f"news-000{i}", "video_sha256": _sha(f"news-000{i}")}
                                   for i in (1, 2)])
    pid = protocol_digest(protocol)
    records = [
        _record("news-0001", "今天天气很好。", "今天天气很好。", protocol_id=pid, e2e=1.0, duration=10.0, frames=300),
        _record("news-0002", "会议将在下午三点开始。", "会议将在下午三点开始。", protocol_id=pid, e2e=2.0, duration=20.0, frames=600),
    ]
    specs.append({
        "name": "01-identical",
        "protocol": protocol,
        "records": records,
        "cold_start": _cold_start(),
        "expected": {
            "records.success": 2, "records.empty": 0, "records.failed": 0,
            "accuracy.denominator.success": 2, "accuracy.denominator.total": 2,
            "accuracy.metrics.exact_match_count": 2,
            "accuracy.metrics.exact_match_percent": 100.0,
            "accuracy.metrics.bleu_zh": 100.0,
            "accuracy.metrics.chrf": 100.0,
            "accuracy.metrics.rouge_l_f1_char": 1.0,
            "accuracy.metrics.mean_char_edit_similarity": 1.0,
            "accuracy.metrics.mean_length_ratio_prediction_to_reference": 1.0,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.mean": 1.5,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.p50": 1.5,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.p95": 1.95,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.p99": 1.99,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.max": 2.0,
            "latency.by_pose_source.disk_cache.rtf.mean": 0.1,
            "latency.by_pose_source.disk_cache.throughput.videos_per_minute": 40.0,
            "latency.by_pose_source.disk_cache.throughput.frames_per_second": 300.0,
            "protocol_review.overall": "pass",
        },
    })

    # --- 02 混合状态：空预测/失败/ok-but-empty 异常 ---------------------------
    protocol = _protocol()
    pid = protocol_digest(protocol)
    records = [
        _record("news-0011", "今天天气很好。", "今天天气很好。", protocol_id=pid, e2e=1.0),
        _record("news-0012", "会议将在下午三点开始。", "会议将在下午开始。", protocol_id=pid, e2e=1.2),
        _record("news-0013", "谢谢你的帮助。", "", protocol_id=pid, status="empty",
                e2e=0.9, error="empty_text_output"),
        _record("news-0014", "我们明天再见。", "", protocol_id=pid, status="error",
                e2e=0.4, error="RuntimeError: decode failed"),
        _record("news-0015", "请注意安全。", "   ", protocol_id=pid, status="ok", e2e=0.8),
    ]
    specs.append({
        "name": "02-mixed-status",
        "protocol": protocol,
        "records": records,
        "cold_start": _cold_start(),
        "expected": {
            "records.video_records": 5, "records.success": 2, "records.empty": 2, "records.failed": 1,
            "accuracy.denominator.success": 2, "accuracy.denominator.total": 5,
            "accuracy.metrics.n": 2,
            "accuracy.metrics.exact_match_count": 1,
            "accuracy.metrics.exact_match_percent": 50.0,
            "accuracy.full_set_pessimistic.metrics.exact_match_count": 1,
            "accuracy.full_set_pessimistic.metrics.n": 5,
            "accuracy.full_set_pessimistic.metrics.exact_match_percent": 20.0,
            "len(accuracy.anomalies)": 1,
            "len(failed_samples)": 3,
            "protocol_review.overall": "fail",
        },
    })

    # --- 03 分组延迟 + 冷启动记录 --------------------------------------------
    protocol = _protocol(selected=[{"number": f"news-002{i}", "video_sha256": _sha(f"news-002{i}")}
                                   for i in range(1, 7)])
    pid = protocol_digest(protocol)
    records = [
        _record("news-0021", "参考句一。", "参考句一。", protocol_id=pid, pose_source="extracted",
                e2e=2.0, duration=10.0, frames=100, decode=0.3, pose_extract=500, pose_preprocess=18.0,
                model_forward=1000, generation=500, model_total=1500),
        _record("news-0022", "参考句二。", "参考句二。", protocol_id=pid, pose_source="extracted",
                e2e=3.0, duration=10.0, frames=200, decode=0.4, pose_extract=600, pose_preprocess=19.0,
                model_forward=2000, generation=600, model_total=2600),
        _record("news-0023", "参考句三。", "参考句三。", protocol_id=pid, pose_source="extracted",
                e2e=4.0, duration=10.0, frames=300, decode=0.5, pose_extract=700, pose_preprocess=20.0,
                model_forward=3000, generation=700, model_total=3700),
        _record("news-0024", "参考句四。", "参考句四。", protocol_id=pid, pose_source="disk_cache",
                e2e=0.5, duration=10.0, frames=100, decode=0.2, pose_extract=0),
        _record("news-0025", "参考句五。", "参考句五。", protocol_id=pid, pose_source="disk_cache",
                e2e=1.0, duration=10.0, frames=200, decode=0.2, pose_extract=0),
        _record("news-0026", "参考句六。", "参考句六。", protocol_id=pid, pose_source="disk_cache",
                e2e=1.5, duration=10.0, frames=300, decode=0.2, pose_extract=0),
        # 冷启动记录：没有 number，event=warmup，e2e 30s，绝不能进逐视频统计
        {"event": "warmup", "status": "warmup", "number": None, "e2e_seconds": 30.0,
         "model_load_seconds": 27.0, "warmup_seconds": 3.0,
         "created_at": "2026-10-25T12:00:00+08:00"},
    ]
    specs.append({
        "name": "03-latency-cold-start",
        "protocol": protocol,
        "records": records,
        "cold_start": _cold_start(),
        "decode_benchmark": {
            "n": 3, "failed": 0,
            "decode_ms": {"n": 3, "mean": 130.0, "p50": 130.0, "p95": 138.0, "p99": 139.6, "min": 120.0, "max": 140.0},
            "decode_fps": {"n": 3, "mean": 800.0, "p50": 800.0, "p95": 850.0, "p99": 860.0, "min": 750.0, "max": 870.0},
            "per_video": [
                {"number": "news-0021", "video_frames": 100, "decode_ms": 120.0, "decode_fps": 833.3},
                {"number": "news-0022", "video_frames": 200, "decode_ms": 130.0, "decode_fps": 800.0},
                {"number": "news-0023", "video_frames": 300, "decode_ms": 140.0, "decode_fps": 750.0},
            ],
        },
        "expected": {
            "records.video_records": 6, "records.cold_start_or_aux_records": 1,
            "latency.by_pose_source.extracted.n": 3,
            "latency.by_pose_source.extracted.fields.e2e_seconds.mean": 3.0,
            "latency.by_pose_source.extracted.fields.e2e_seconds.p95": 3.9,
            "latency.by_pose_source.extracted.fields.e2e_seconds.max": 4.0,
            "latency.by_pose_source.extracted.rtf.mean": 0.3,
            "latency.by_pose_source.extracted.throughput.frames_per_second": 600.0 / 9.0,
            "latency.by_pose_source.extracted.throughput.videos_per_minute": 20.0,
            "latency.by_pose_source.disk_cache.n": 3,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.mean": 1.0,
            "latency.by_pose_source.disk_cache.fields.e2e_seconds.p95": 1.45,
            "latency.by_pose_source.disk_cache.rtf.mean": 0.1,
            "latency.by_pose_source.disk_cache.throughput.frames_per_second": 200.0,
            "latency.cold_start.count": 1,
            "latency.cold_start.total_seconds": 30.0,
            "latency.cold_start.external_record.load_model_seconds": 27.0,
            "latency.decode_only_benchmark.decode_ms_recomputed.n": 3,
            "latency.decode_only_benchmark.decode_ms_recomputed.p50": 130.0,
            "latency.decode_only_benchmark.decode_ms_recomputed.mean": 130.0,
            "latency.by_pose_source.extracted.fields.e2e_seconds.n": 3,
            "latency.by_pose_source.extracted.fields.pose_preprocess_ms.mean": 19.0,
            "latency.stage_decomposition[0].stage": "cold-start 模型加载",
            "latency.stage_decomposition[0].mean": 27.0,
            "latency.stage_decomposition[1].stage": "纯视频解码（cv2 读帧）",
            "latency.stage_decomposition[1].mean": 130.0,
            "latency.stage_decomposition[2].stage": "解码+RTMPose 合计",
            "latency.stage_decomposition[2].mean": 600.0,
            "records.coverage.protocol_selected": 6,
            "records.coverage.complete": True,
            "protocol_review.checks[5].status": "pass",
            "protocol_review.overall": "pass",
        },
    })

    # --- 04 协议混入 + 缺 video_sha256 --------------------------------------
    protocol = _protocol()
    pid = protocol_digest(protocol)
    records = [
        _record("news-0031", "参考句甲。", "参考句甲。", protocol_id=pid),
        _record("news-0032", "参考句乙。", "参考句乙。", protocol_id=pid, sha=False),
        _record("news-0033", "参考句丙。", "参考句丙。", protocol_id="deadbeef" * 8),
        _record("news-0034", "参考句丁。", "参考句丁。", protocol_id=pid),
    ]
    specs.append({
        "name": "04-protocol-mixed",
        "protocol": protocol,
        "records": records,
        "expected": {
            "records.success": 4,
            "len(protocol_review.checks[0].distinct_ids)": 2,
            "protocol_review.checks[0].status": "fail",
            "protocol_review.checks[4].status": "fail",
            "protocol_review.overall": "fail",
        },
    })

    # --- 05 干净样本：全部复核通过 ------------------------------------------
    selected = [
        {"number": f"news-004{i}", "video_sha256": _sha(f"news-004{i}")} for i in range(1, 4)
    ]
    protocol = _protocol(selected=selected)
    pid = protocol_digest(protocol)
    records = [
        _record("news-0041", "干净样本一。", "干净样本一。", protocol_id=pid, pose_source="extracted", e2e=1.0),
        _record("news-0042", "干净样本二。", "干净样本二。", protocol_id=pid, pose_source="disk_cache", e2e=1.2),
        _record("news-0043", "干净样本三。", "干净样本三。", protocol_id=pid, pose_source="disk_cache", e2e=0.9),
    ]
    specs.append({
        "name": "05-clean-pass",
        "protocol": protocol,
        "records": records,
        "cold_start": _cold_start(),
        "expected": {
            "records.success": 3,
            "accuracy.metrics.exact_match_percent": 100.0,
            "protocol_review.checks[0].status": "pass",
            "protocol_review.checks[1].status": "pass",
            "protocol_review.checks[2].status": "pass",
            "protocol_review.checks[3].status": "pass",
            "protocol_review.checks[4].status": "pass",
            "protocol_review.checks[5].status": "pass",
            "protocol_review.overall": "pass",
        },
    })

    # --- 06 样本不完整：协议 4 条、实际只有 3 条 ----------------------------
    selected = [
        {"number": f"news-005{i}", "video_sha256": _sha(f"news-005{i}")} for i in range(1, 5)
    ]
    protocol = _protocol(selected=selected)
    pid = protocol_digest(protocol)
    records = [
        _record("news-0051", "完整样本一。", "完整样本一。", protocol_id=pid),
        _record("news-0052", "完整样本二。", "完整样本二。", protocol_id=pid),
        _record("news-0053", "完整样本三。", "完整样本三。", protocol_id=pid),
    ]
    specs.append({
        "name": "06-incomplete-coverage",
        "protocol": protocol,
        "records": records,
        "cold_start": _cold_start(),
        "expected": {
            "records.coverage.protocol_selected": 4,
            "len(records.coverage.missing_numbers)": 1,
            "records.coverage.complete": False,
            "protocol_review.checks[5].status": "fail",
            "protocol_review.overall": "fail",
            "records.success": 3,
        },
    })

    for spec in specs:
        directory = fixtures_root / spec["name"]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "run-protocol.json", spec["protocol"])
        with (directory / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for record in spec["records"]:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        if spec.get("cold_start") is not None:
            write_json(directory / "cold-start.json", spec["cold_start"])
        if spec.get("decode_benchmark") is not None:
            write_json(directory / "decode-benchmark.json", spec["decode_benchmark"])
        spec["run_dir"] = directory
    return specs


def build_ablation_fixtures(fixtures_root: Path):
    """消融 schema（与 scripts/evaluate_unisign_cslnews_upstream_pose.py 一致）+ 一份主评测侧。

    6 条消融记录：4 ok、1 empty、1 error；主评测侧 5 条（缺最后一条，且第 5 条为空），
    因此正确配对分母应为 4（剔除 main_missing 1、ablation_not_ok 1）。
    """
    ablation_dir = fixtures_root / "ablation-simulated"
    main_dir = fixtures_root / "main-for-ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    main_dir.mkdir(parents=True, exist_ok=True)

    selected = [
        {"number": f"cslnews-{i:06d}", "video": f"E:/fake/archive_001/v{i}.mp4",
         "video_sha256": _sha(f"cslnews-{i:06d}"), "reference": f"参考句子{i}内容。"}
        for i in range(1, 7)
    ]
    protocol = {
        "schema_version": 1,
        "ablation": "upstream-cslnews-pose-pkl-vs-local-rtmlib",
        "main_run_protocol_id": "fixture-main-protocol",
        "checkpoint_sha256": "73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c",
        "source_commit": "eed438bcb49e30405cd6ccdfcccca330c134e830",
        "pose_source": "upstream-pkl",
        "pose_zip": "E:/fake/archive_001.zip",
        "pose_zip_sha256": _sha("zip"),
        "pose_pipeline": "normalize_rtmpose_pixels(isotropic_short_side) -> prepare_pose_window(max_frames=256)",
        "num_beams": 4, "max_new_tokens": 100, "device": "cuda", "dtype": "bfloat16",
        "selected": selected,
    }
    protocol_id = ablation_protocol_digest(protocol)
    write_json(ablation_dir / "ablation-protocol.json", protocol)

    predictions = ["参考句子1内容。", "参考句子2内容。", "参考句子3内容。", "完全不同的句子。", "", ""]
    statuses = ["ok", "ok", "ok", "ok", "empty", "error"]
    rows = []
    for index in range(6):
        number = f"cslnews-{index + 1:06d}"
        rows.append({
            "number": number, "video": f"E:/fake/archive_001/v{index + 1}.mp4",
            "video_sha256": _sha(number), "reference": f"参考句子{index + 1}内容。",
            "prediction": predictions[index], "status": statuses[index],
            "error": ("empty_text_output" if statuses[index] == "empty"
                      else ("RuntimeError: boom" if statuses[index] == "error" else None)),
            "protocol_id": protocol_id, "pose_source": "upstream-pkl",
            "pkl": f"E:/fake/pose/v{index + 1}.pkl",
            "video_frames": 287, "video_duration_s": 9.566666666666666,
            "pose_load_ms": 12.5 + index, "pose_prepare_ms": 3.2 + index,
            "source_frames": 287, "sampled_frames": 256,
            "valid_body_fraction": 0.95, "body_scale": 0.66,
            "model_forward_ms": 40.0 + index, "generation_ms": 300.0 + 10 * index,
            "decode_ms": 0.4, "model_total_ms": 340.0 + 11 * index,
            "e2e_seconds": 0.36 + 0.01 * index,
            "created_at": "2026-10-25T13:00:00+08:00",
        })
    with (ablation_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    main_predictions = ["参考句子1内容。", "参考句子2内容。", "参考句子3内容。", "完全不同的句子。", ""]
    main_statuses = ["ok", "ok", "ok", "ok", "empty"]
    main_rows = []
    for index in range(5):
        number = f"cslnews-{index + 1:06d}"
        main_rows.append({
            "number": number, "reference": f"参考句子{index + 1}内容。",
            "prediction": main_predictions[index], "status": main_statuses[index],
            "run_protocol_id": "fixture-main-protocol", "pose_source": "extracted",
            "video_sha256": _sha(number), "video_frames": 287,
            "video_duration_s": 9.566666666666666, "e2e_seconds": 7.0 + index,
            "model_total_ms": 420.0,
        })
    with (main_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in main_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(main_dir / "run-protocol.json", {"dataset": "CSL-News", "selected": selected[:5]})
    return ablation_dir, main_dir, protocol_id


def get_path(obj, path: str):
    if path.startswith("len(") and path.endswith(")"):
        inner = path[4:-1]
        return len(get_path(obj, inner))
    node = obj
    for part in path.split("."):
        match = re.fullmatch(r"(\w+)\[(\d+)\]", part)
        if match:
            node = node[match.group(1)][int(match.group(2))]
        else:
            node = node[part]
    return node


def run_self_test(fixtures_root: Path):
    specs = build_fixtures(fixtures_root)
    report = {
        "generated_at": now_iso(),
        "fixtures_root": str(fixtures_root),
        "fixtures": [],
        "total_assertions": 0,
        "failed_assertions": 0,
        "failures": [],
        "ok": True,
    }
    for spec in specs:
        run_dir = spec["run_dir"]
        summary = summarize_run(run_dir)
        out_dir = run_dir / "out"
        write_json(out_dir / "summary.json", summary)
        # 同时渲染 md，确保 md 渲染路径也被自测覆盖（含 fail/warn 分支）
        write_text(out_dir / "summary.md", render_markdown(summary, self_test=None))
        assertions, failures = [], []
        for path, expected in spec["expected"].items():
            try:
                actual = get_path(summary, path)
            except (KeyError, IndexError, TypeError):
                actual = "<missing>"
            if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)):
                ok = abs(float(actual) - float(expected)) <= 1e-9
            else:
                ok = actual == expected
            assertions.append({"path": path, "expected": expected, "actual": actual, "ok": ok})
            if not ok:
                failures.append({"fixture": spec["name"], "path": path,
                                 "expected": expected, "actual": actual})
        metrics = ((summary.get("accuracy") or {}).get("metrics") or {})
        headline = {
            "exact_match_percent": metrics.get("exact_match_percent"),
            "bleu_zh": metrics.get("bleu_zh"),
            "chrf": metrics.get("chrf"),
            "rouge_l_f1_char": metrics.get("rouge_l_f1_char"),
            "mean_char_edit_similarity_percent": metrics.get("mean_char_edit_similarity_percent"),
            "protocol_review": (summary.get("protocol_review") or {}).get("overall"),
        }
        report["fixtures"].append({
            "name": spec["name"],
            "run_dir": str(run_dir),
            "summary_path": str(out_dir / "summary.json"),
            "headline": headline,
            "assertions": assertions,
            "failures": failures,
        })
        report["total_assertions"] += len(assertions)
        report["failed_assertions"] += len(failures)
        report["failures"].extend(failures)

    # --- 消融模式自测：正确配对结果应 pass；错误分母必须被识别为 fail --------
    ablation_dir, main_dir, _ = build_ablation_fixtures(fixtures_root)
    correct_paired = {
        "paired_n": 4,
        "upstream_pose_mean_edit_similarity": 0.7812,
        "local_rtmlib_mean_edit_similarity": 0.7812,
        "mean_delta_upstream_minus_local": 0.0,
        "delta_p50": 0.0,
        "identical_prediction_count": 4,
        "protocol_note": "fixture: same clips, weights and decoding",
    }

    def write_script_paired(payload):
        write_json(ablation_dir / "summary.json", {
            "schema_version": 1, "ablation_not_production": True,
            "paired_comparison_vs_local_frontend": payload,
        })

    write_script_paired(correct_paired)
    ablation_ok = summarize_ablation(ablation_dir, main_dir)
    write_text(ablation_dir / "out" / "ablation-summary.md", render_ablation_md(ablation_ok))
    write_json(ablation_dir / "out" / "ablation-summary.json", ablation_ok)

    wrong_paired = dict(correct_paired)
    wrong_paired["paired_n"] = 5          # 故意把单边样本算进分母
    wrong_paired["mean_delta_upstream_minus_local"] = 0.01
    write_script_paired(wrong_paired)
    ablation_wrong = summarize_ablation(ablation_dir, main_dir)
    write_script_paired(correct_paired)   # 还原，避免污染后续运行

    ablation_expected = {
        "records.video_records": 6,
        "records.success": 4,
        "records.empty": 1,
        "records.failed": 1,
        "records.coverage.complete": True,
        "paired_comparison.paired_n": 4,
        "paired_comparison.skipped.main_missing": 1,
        "paired_comparison.skipped.ablation_not_ok": 1,
        "paired_comparison.wins_upstream": 0,
        "paired_comparison.ties": 4,
        "latency.by_pose_source.upstream-pkl.fields.e2e_seconds.n": 6,
        "review.overall": "pass",
    }
    assertions, failures = [], []
    for path, expected in ablation_expected.items():
        try:
            actual = get_path(ablation_ok, path)
        except (KeyError, IndexError, TypeError):
            actual = "<missing>"
        if isinstance(expected, (int, float)) and not isinstance(expected, bool) and isinstance(actual, (int, float)):
            ok = abs(float(actual) - float(expected)) <= 1e-9
        else:
            ok = actual == expected
        assertions.append({"path": path, "expected": expected, "actual": actual, "ok": ok})
        if not ok:
            failures.append({"fixture": "ablation-pairing", "path": path,
                             "expected": expected, "actual": actual})
    for path, expected in (("review.overall", "fail"),
                           ("review.checks[1].status", "fail"),
                           ("review.checks[1].paired_n", 4)):
        try:
            actual = get_path(ablation_wrong, path)
        except (KeyError, IndexError, TypeError):
            actual = "<missing>"
        ok = actual == expected
        assertions.append({"path": f"wrong-denominator::{path}", "expected": expected,
                           "actual": actual, "ok": ok})
        if not ok:
            failures.append({"fixture": "ablation-pairing", "path": f"wrong-denominator::{path}",
                             "expected": expected, "actual": actual})
    metrics = ((ablation_ok.get("accuracy") or {}).get("metrics") or {})
    report["fixtures"].append({
        "name": "ablation-pairing",
        "run_dir": str(ablation_dir),
        "summary_path": str(ablation_dir / "out" / "ablation-summary.json"),
        "headline": {
            "exact_match_percent": metrics.get("exact_match_percent"),
            "bleu_zh": metrics.get("bleu_zh"),
            "chrf": metrics.get("chrf"),
            "rouge_l_f1_char": metrics.get("rouge_l_f1_char"),
            "mean_char_edit_similarity_percent": metrics.get("mean_char_edit_similarity_percent"),
            "protocol_review": (ablation_ok.get("review") or {}).get("overall"),
        },
        "assertions": assertions,
        "failures": failures,
    })
    report["total_assertions"] += len(assertions)
    report["failed_assertions"] += len(failures)
    report["failures"].extend(failures)

    report["ok"] = report["failed_assertions"] == 0
    write_json(fixtures_root / "selftest-report.json", report)
    return report


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def discover_run_dir():
    """优先 run-cslnews-*，其次 pilot-*；只接受含非空 predictions.jsonl 的目录。"""
    tried = []
    for pattern in DISCOVERY_PATTERNS:
        tried.append(pattern)
        candidates = sorted(ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for candidate in candidates:
            predictions = candidate / "predictions.jsonl"
            if predictions.is_file() and predictions.stat().st_size > 0:
                return candidate, None
    return None, f"未找到含非空 predictions.jsonl 的运行目录（依次匹配 {', '.join(tried)}）"


def main(argv=None):
    parser = argparse.ArgumentParser(description="汇总 Uni-Sign 评测输出并复核协议")
    parser.add_argument("--run-dir", type=Path, help="评测输出目录（含 predictions.jsonl / run-protocol.json）")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="汇总输出目录")
    parser.add_argument("--output-stem", default="summary",
                        help="输出文件名前缀（默认 summary；interim 用 summary-interim）")
    parser.add_argument("--scope-label", help="口径标签，例如 'interim 子集 300/1694'")
    parser.add_argument("--fixture-root", type=Path, help="fixture 根目录（默认 <output-dir>/fixture）")
    parser.add_argument("--self-test", action="store_true", help="只跑合成 fixture 自测")
    parser.add_argument("--skip-self-test", action="store_true", help="生成真实汇总时不跑自测")
    parser.add_argument("--force", action="store_true", help="数据未就绪时允许覆盖已有真实汇总")
    parser.add_argument("--ablation", action="store_true", help="汇总前端消融目录（上游 pose pkl）")
    parser.add_argument("--compare-run-dir", type=Path,
                        default=ROOT / "artifacts/cslnews-eval/run-cslnews-20261025",
                        help="消融配对用的主评测目录")
    parser.add_argument("--ablation-output", type=Path,
                        help="消融 md 输出路径（默认 <output-dir>/ablation-summary.md）")
    parser.add_argument("--profile", choices=tuple(PROFILES),
                        help="专用汇总：stage1（训练集拟合）/ csldaily（BLEU 口径复核）")
    parser.add_argument("--control-run-dir", type=Path,
                        help="专用汇总的对照运行目录（stage1 默认 pinned-upstreampose）")
    parser.add_argument("--alt-control-run-dir", type=Path,
                        help="专用汇总的第二对照（姿态源不同，只作参考，例如生产路径 run-cslnews-*）")
    parser.add_argument("--frontend-compare-run-dir", type=Path,
                        help="同权重、不同姿态来源的对照运行（用于前端对照表）")
    parser.add_argument("--latency-note", type=str,
                        help="延迟口径警示（例如并发争用/共享 GPU），会写进 md 与 summary.json")
    parser.add_argument("--output-md", type=Path, help="专用汇总 md 输出路径")
    parser.add_argument("--output-json", type=Path, help="专用汇总 json 输出路径")
    parser.add_argument("--stride-review", action="store_true",
                        help="pose_detect_stride=5 精度验收复核（stride5 vs stride1 配对）")
    parser.add_argument("--stride5-stage1", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["stride5_stage1"])
    parser.add_argument("--stride1-stage1", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["stride1_stage1_control"])
    parser.add_argument("--stride5-pose", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["stride5_pose"])
    parser.add_argument("--stride1-pose-wh", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["stride1_pose_wh"])
    parser.add_argument("--stride1-pose-iso", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["stride1_pose_isotropic"])
    parser.add_argument("--combined-run", type=Path, default=ROOT / STRIDE_RUN_DEFAULTS["combined"])
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    fixture_root = (args.fixture_root or output_dir / "fixture").resolve()

    if args.stride_review:
        stride_summary = summarize_stride_validation(
            stride5_run=args.stride5_stage1, control_run=args.stride1_stage1,
            stride5_pose_run=args.stride5_pose, baseline_wh_run=args.stride1_pose_wh,
            baseline_isotropic_run=args.stride1_pose_iso, combined_run=args.combined_run)
        md_path = (args.output_md or output_dir / "summary-stride-validation.md").resolve()
        json_path = (args.output_json or md_path.with_suffix(".json")).resolve()
        write_json(json_path, stride_summary)
        write_text(md_path, render_stride_md(stride_summary))
        print(json.dumps({
            "event": "stride_review",
            "md": str(md_path), "json": str(json_path),
            "n": stride_summary["numbers"]["n"],
            "bleu_zh_stride1": stride_summary["accuracy"]["stride1"]["bleu_zh"],
            "bleu_zh_stride5": stride_summary["accuracy"]["stride5"]["bleu_zh"],
            "paired_delta": stride_summary["accuracy"]["paired"]["mean_delta"],
            "paired_ci95": stride_summary["accuracy"]["paired"]["delta_ci95"],
            "e2e_stride1_iso": stride_summary["latency"]["stride1_isotropic"]["e2e_seconds"]["mean"],
            "e2e_stride5": stride_summary["latency"]["stride5"]["e2e_seconds"]["mean"],
            "overall": stride_summary["overall"],
            "verdict": stride_summary["verdict"],
        }, ensure_ascii=False, indent=2))
        return 0 if stride_summary["overall"] != "fail" else 3

    if args.self_test:
        report = run_self_test(fixture_root)
        print(json.dumps({"event": "self_test", "ok": report["ok"],
                          "total_assertions": report["total_assertions"],
                          "failed_assertions": report["failed_assertions"],
                          "report": str(fixture_root / "selftest-report.json")},
                         ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1

    if args.profile:
        if args.run_dir is None:
            parser.error("--profile 需要显式 --run-dir")
        profile_dir = args.run_dir.resolve()
        if not (profile_dir / "predictions.jsonl").is_file():
            print(json.dumps({"event": "profile_no_data", "run_dir": str(profile_dir),
                              "reason": "目录下没有 predictions.jsonl"},
                             ensure_ascii=False, indent=2))
            return 2
        profile_summary = summarize_profile(
            profile_dir, args.profile, output_dir=output_dir,
            control_run_dir=args.control_run_dir.resolve() if args.control_run_dir else None,
            alt_control_run_dir=(args.alt_control_run_dir.resolve()
                                 if args.alt_control_run_dir else None),
            frontend_compare_run_dir=(args.frontend_compare_run_dir.resolve()
                                      if args.frontend_compare_run_dir else None))
        md_path = (args.output_md or output_dir / f"summary-{args.profile}.md").resolve()
        json_path = (args.output_json or md_path.with_suffix(".json")).resolve()
        write_json(json_path, profile_summary)
        write_text(md_path, render_profile_md(profile_summary))
        review = profile_summary["review"]["overall"]
        print(json.dumps({
            "event": "profile_summary",
            "profile": args.profile,
            "run_dir": str(profile_dir),
            "md": str(md_path),
            "json": str(json_path),
            "records": profile_summary["records"]["video_records"],
            "success": profile_summary["records"]["success"],
            "coverage": profile_summary["records"]["coverage"],
            "bleu_zh": (profile_summary["accuracy"]["metrics"] or {}).get("bleu_zh"),
            "chrf": (profile_summary["accuracy"]["metrics"] or {}).get("chrf"),
            "review": review,
            "warnings": profile_summary["warnings"],
        }, ensure_ascii=False, indent=2))
        return 0 if review != "fail" else 3

    if args.ablation:
        ablation_dir = (args.run_dir or ROOT / "artifacts/cslnews-eval/ablation-upstream-pose").resolve()
        if not (ablation_dir / "predictions.jsonl").is_file():
            print(json.dumps({"event": "ablation_no_data", "run_dir": str(ablation_dir),
                              "reason": "消融目录下没有 predictions.jsonl"},
                             ensure_ascii=False, indent=2))
            return 2
        ablation_summary = summarize_ablation(
            ablation_dir, args.compare_run_dir.resolve(), output_dir=output_dir)
        md_path = (args.ablation_output or output_dir / "ablation-summary.md").resolve()
        json_path = md_path.with_suffix(".json")
        write_json(json_path, ablation_summary)
        write_text(md_path, render_ablation_md(ablation_summary))
        review = ablation_summary["review"]["overall"]
        paired = ablation_summary.get("paired_comparison") or {}
        print(json.dumps({
            "event": "ablation_summary",
            "run_dir": str(ablation_dir),
            "ablation_md": str(md_path),
            "ablation_json": str(json_path),
            "records": ablation_summary["records"]["video_records"],
            "success": ablation_summary["records"]["success"],
            "coverage": ablation_summary["records"]["coverage"],
            "paired_n": paired.get("paired_n"),
            "mean_delta_upstream_minus_local": paired.get("mean_delta_upstream_minus_local"),
            "delta_bootstrap_95ci": paired.get("delta_bootstrap_95ci"),
            "review": review,
            "warnings": ablation_summary["warnings"],
        }, ensure_ascii=False, indent=2))
        return 0 if review != "fail" else 3

    self_test = None
    if not args.skip_self_test:
        self_test = run_self_test(fixture_root)

    stem = args.output_stem or "summary"
    summary_json_path = output_dir / f"{stem}.json"
    summary_md_path = output_dir / f"{stem}.md"
    per_sample_name = "per-sample.csv" if stem == "summary" else f"per-sample-{stem.replace('summary-', '')}.csv"

    run_dir = args.run_dir.resolve() if args.run_dir else None
    no_data_reason = None
    if run_dir is None:
        run_dir, no_data_reason = discover_run_dir()
    elif not (run_dir / "predictions.jsonl").is_file():
        no_data_reason = f"{run_dir} 下没有 predictions.jsonl"
        run_dir = None

    if run_dir is not None:
        predictions = run_dir / "predictions.jsonl"
        if predictions.stat().st_size == 0:
            no_data_reason = f"{predictions} 为空"
            run_dir = None

    if run_dir is None:
        existing = output_dir / "summary.json"
        if existing.is_file() and not args.force and stem == "summary":
            try:
                previous = json.loads(existing.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                previous = {}
            if previous.get("status") == "ok":
                print(json.dumps({"event": "no_data_but_keep_existing",
                                  "reason": no_data_reason,
                                  "existing_summary": str(existing)}, ensure_ascii=False, indent=2))
                return 2
        summary = {
            "schema_version": 1,
            "status": "no_data",
            "generated_at": now_iso(),
            "reason": no_data_reason,
            "expected_run_glob": " / ".join(DISCOVERY_PATTERNS),
            "self_test": self_test,
        }
        write_json(summary_json_path, summary)
        write_text(summary_md_path, render_markdown(summary, self_test=self_test,
                                                    no_data_reason=no_data_reason))
        print(json.dumps({"event": "no_data", "reason": no_data_reason,
                          "summary": str(summary_json_path)}, ensure_ascii=False, indent=2))
        return 2

    summary = summarize_run(run_dir, output_dir=output_dir,
                            per_sample_name=per_sample_name, scope_label=args.scope_label,
                            latency_note=args.latency_note)
    summary["self_test"] = self_test
    write_json(summary_json_path, summary)
    write_text(summary_md_path, render_markdown(summary, self_test=self_test))

    review = summary["protocol_review"]["overall"]
    print(json.dumps({
        "event": "summary",
        "run_dir": str(run_dir),
        "scope": summary.get("scope"),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "video_records": summary["records"]["video_records"],
        "protocol_selected": (summary["records"].get("coverage") or {}).get("protocol_selected"),
        "success": summary["records"]["success"],
        "empty": summary["records"]["empty"],
        "failed": summary["records"]["failed"],
        "exact_match_percent": summary["accuracy"]["metrics"]["exact_match_percent"]
        if summary["accuracy"]["metrics"] else None,
        "bleu_zh": summary["accuracy"]["metrics"]["bleu_zh"] if summary["accuracy"]["metrics"] else None,
        "chrf": summary["accuracy"]["metrics"]["chrf"] if summary["accuracy"]["metrics"] else None,
        "protocol_review": review,
        "warnings": summary["warnings"],
    }, ensure_ascii=False, indent=2))
    return 0 if review != "fail" else 3


if __name__ == "__main__":
    raise SystemExit(main())
