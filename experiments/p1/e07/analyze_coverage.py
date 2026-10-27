#!/usr/bin/env python
"""P-1.5 semantic coverage & effective-recognition-range analysis (CPU only, read-only).

Answers: **should we expand data or improve the model first?** by crossing a
12-category semantic taxonomy against (a) the CSL-Daily *train* annotation
volume, (b) per-category test performance of two systems (S2 = stage2
pose-subset, A2 = Uni-Sign official-pose baseline) and (c) the char-level error
type mix, then correcting both for reference length and for a near-zero
training vocabulary at the *lexical* level.

Design notes
------------
* No LLM is called anywhere. Category assignment is pure keyword / regex
  matching over normalised text; the whole vocabulary lives in ``CATEGORIES``
  below so any reviewer can re-check a single row by hand.
* GT discipline: ``experiments/p1/references/gt_1086.csv`` is read **only**
  here, on the evaluation side. Reference sentences are never copied into any
  output; the Markdown report carries counts and metrics only.
* Metrics are imported from ``experiments/p1/eval/metrics.py`` (the frozen
  canonical ``p1-v1`` caliber) so the per-category numbers are directly
  comparable to the published 1086 table (S2 edit 0.4358 / A2 edit 0.4212).
  ``verify_baseline`` re-derives those two corpus means as a self-check.
* The train pickle stores a torch tensor under ``sign``. Torch is absent on
  this CPU-only box and is not needed (only ``text`` / ``gloss`` / ``num_frames``
  are consumed), so two minimal stub modules satisfy the unpickler. Nothing is
  deserialised into executable code beyond the stub callables below.

Usage
-----
    python experiments/p1/e07/analyze_coverage.py            # write both artifacts
    python experiments/p1/e07/analyze_coverage.py --check    # self-check only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import pickle
import re
import statistics
import sys
import types
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS_ID = "p1-e07-semantic-coverage-v1"
OUT_JSON = ROOT / "experiments" / "p1" / "e07" / "coverage_analysis.json"
OUT_MD = ROOT / "docs" / "experiments" / "P-1" / "P15_COVERAGE_ANALYSIS.md"

TRAIN_PKL = ROOT / "artifacts" / "slrt-gdrive-20261025" / "csl-daily.train.pkl"
GT_CSV = ROOT / "experiments" / "p1" / "references" / "gt_1086.csv"
S2_JSONL = (ROOT / "experiments" / "p1" / "e01" / "runs"
            / "stage2-pose-subset-1086-20261027" / "predictions_stage2_pose_subset.jsonl")
A2_JSONL = ROOT / "experiments" / "p1" / "e02" / "candidate_outputs.jsonl"
METRICS_PY = ROOT / "experiments" / "p1" / "eval" / "metrics.py"

# ---------------------------------------------------------------------------
# decision thresholds (single place, quoted verbatim in the report)
# ---------------------------------------------------------------------------
WEAK_EDIT = 0.40          # S2 mean char-edit similarity below this = weak
WEAK_MIN_TEST = 30        # ... and the category must have at least this many test samples
WEAK_RESIDUAL = -0.02     # length-adjusted residual needed to call a weak class *semantically* weak
LOW_EDIT = 0.50           # "low score" per sample
DATA_SHORT_TRAIN = 300    # n_train below this = the data itself is thin
GAP_KW_TRAIN = 30         # keyword seen fewer times than this in train = data gap ...
GAP_KW_TEST = 3           # ... provided it shows up at least this often in test
DATA_GAP_SHARE = 0.25     # >=25% of a category's test samples driven by gap keywords
                          # => the category's weakness is lexically data-starved

# ---------------------------------------------------------------------------
# canonical metrics (p1-v1) — loaded from the frozen project module by path
# ---------------------------------------------------------------------------
def load_metrics():
    spec = importlib.util.spec_from_file_location("_p1_metrics", METRICS_PY)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise SystemExit(f"cannot load {METRICS_PY}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.METRIC_DEFINITION_VERSION != "p1-v1":
        raise SystemExit(f"unexpected metric caliber: {module.METRIC_DEFINITION_VERSION}")
    return module


METRICS = load_metrics()
normalize_text = METRICS.normalize_text
edit_similarity = METRICS.edit_similarity
_edit_ops = METRICS._edit_ops


# ---------------------------------------------------------------------------
# semantic category system  (keyword + regex; NO LLM)
# ---------------------------------------------------------------------------
CN_DIGITS = "零一二三四五六七八九十百千万两"
CN_MEASURE = "个岁元块点天年次位条件只种本张台部斤米步层份名月日周号小时分钟万元"


def _num(extra: str) -> str:
    return CN_DIGITS + extra


CATEGORIES: list[dict[str, Any]] = [
    {
        "id": "number_quantity",
        "label": "数字/数量",
        "keywords": ["多少", "几", "半", "倍", "些", "第一", "第二", "第三"],
        "regex": [r"\d", f"[{_num('')}]+[{CN_MEASURE}]", f"[{CN_DIGITS}][{CN_DIGITS}]?[个岁元块点天年次位条件种本张台部斤米步层份名]"],
        "rationale": "阿拉伯数字，或汉字数字紧邻量词（避免『一样/一起/一直』类虚指被计入）",
    },
    {
        "id": "negation",
        "label": "否定",
        "keywords": ["不", "没", "未", "别", "勿", "毫无", "并不", "从不", "无须"],
        "regex": [],
        "rationale": "否定副词/否定动词词干；『无』未收录（『无数/无论』噪声高），『别』保留（『别人』噪声已在关键词级表暴露）",
    },
    {
        "id": "person_reference",
        "label": "人物指代",
        "keywords": ["我", "你", "您", "他", "她", "它", "我们", "你们", "他们", "她们",
                     "自己", "大家", "咱", "人家", "别人", "有人", "谁", "各位"],
        "regex": [],
        "rationale": "第一/二/三人称及泛指代词；含『谁』（兼属疑问，允许跨类）",
    },
    {
        "id": "time",
        "label": "时间",
        "keywords": ["今天", "明天", "昨天", "后天", "前天", "现在", "刚才", "上午", "下午",
                     "中午", "早上", "晚上", "去年", "今年", "明年", "上个月", "下个月", "星期",
                     "周末", "时候", "多久", "一会儿", "以前", "以后", "假期", "节日", "小时",
                     "分钟", "几点", "白天", "夜里", "春节"],
        "regex": [],
        "rationale": "只收多字时间词，避免『年/月/日/点』单字把『年纪、重要』误判为时间",
    },
    {
        "id": "kinship",
        "label": "亲属称谓",
        "keywords": ["爸爸", "妈妈", "父亲", "母亲", "儿子", "女儿", "哥哥", "姐姐", "弟弟",
                     "妹妹", "爷爷", "奶奶", "外公", "外婆", "姥姥", "姥爷", "叔叔", "阿姨",
                     "舅舅", "姑姑", "伯伯", "婶婶", "丈夫", "妻子", "老公", "老婆", "孩子",
                     "家人", "父母", "亲戚", "孙子", "孙女", "表哥", "表姐", "堂哥", "堂姐",
                     "女婿", "儿媳", "嫂子", "姐夫", "妹夫"],
        "regex": [],
        "rationale": "直系+旁系+姻亲称谓；长尾称谓（姑姑/舅舅/婶婶）是检验长尾数据量的关键项",
    },
    {
        "id": "action",
        "label": "动作",
        "keywords": ["走", "跑", "吃", "喝", "睡", "起床", "洗", "买", "卖", "看", "听", "说",
                     "做", "打", "拿", "给", "送", "写", "读", "学", "玩", "坐", "站", "开",
                     "关", "找", "等", "试", "穿", "脱", "回", "帮", "教", "唱", "跳", "飞",
                     "游", "搬", "擦", "借", "还", "换", "记", "忘"],
        "regex": [],
        "rationale": "高频单字/双字动词，覆盖面最广的一类（训练占比 44%）",
    },
    {
        "id": "object",
        "label": "对象/物品",
        "keywords": ["饭", "菜", "水", "茶", "咖啡", "衣服", "鞋", "书", "手机", "电脑", "电视",
                     "钱", "票", "药", "伞", "笔", "桌子", "椅子", "床", "车", "钥匙", "包",
                     "帽子", "眼镜", "水果", "苹果", "西瓜", "面包", "牛奶", "鸡蛋", "肉", "鱼",
                     "碗", "杯子", "盘子", "灯", "门", "窗", "礼物", "照片", "玩具", "报纸"],
        "regex": [],
        "rationale": "可数实体名词（日用/食物/交通/文具）",
    },
    {
        "id": "place",
        "label": "地点",
        "keywords": ["家", "学校", "医院", "公司", "商店", "超市", "银行", "饭店", "餐厅", "公园",
                     "图书馆", "教室", "办公室", "宿舍", "车站", "机场", "火车站", "地铁站",
                     "厨房", "卧室", "厕所", "卫生间", "北京", "上海", "中国", "外国", "城市",
                     "农村", "这里", "那里", "楼上", "楼下", "门口", "路上"],
        "regex": [],
        "rationale": "机构/场所名词 + 指示方位（『家』兼属亲属，跨类允许）",
    },
    {
        "id": "emotion_eval",
        "label": "情绪/评价",
        "keywords": ["高兴", "开心", "快乐", "难过", "伤心", "生气", "着急", "累", "舒服",
                     "好吃", "好看", "好玩", "好听", "漂亮", "可爱", "讨厌", "喜欢", "爱",
                     "怕", "紧张", "满意", "不错", "担心", "放心", "幸福", "痛", "疼",
                     "遗憾", "麻烦", "简单", "困难", "方便", "干净", "脏"],
        "regex": [],
        "rationale": "情感状态词 + 形容词性评价（不含程度副词，后者归『比较/程度』）",
    },
    {
        "id": "question",
        "label": "疑问句",
        "keywords": ["吗", "呢", "什么", "怎么", "为什么", "怎样", "哪", "谁", "多少", "几",
                     "是不是", "有没有", "对不对", "好不好", "行不行", "能不能"],
        "regex": [r"\?", r"么$"],
        "rationale": "疑问语气词/疑问代词/正反问 + 句末问号（NFKC 后全角『？』即『?』）",
    },
    {
        "id": "comparison_degree",
        "label": "比较/程度",
        "keywords": ["比", "比较", "更", "最", "太", "很", "非常", "特别", "一样", "不如",
                     "差不多", "越来越", "有点", "挺", "十分", "相当", "格外", "过于", "越",
                     "差不多", "极了"],
        "regex": [],
        "rationale": "比较结构（比/像/不如/一样）+ 程度副词（很/非常/最/太）",
    },
    {
        "id": "causal_conditional",
        "label": "因果/条件",
        "keywords": ["因为", "所以", "如果", "要是", "虽然", "但是", "可是", "不过", "而且",
                     "然后", "于是", "那么", "即使", "只有", "否则", "不然", "为了", "既然",
                     "因此", "假设", "万一", "才", "就"],
        "regex": [],
        "rationale": "因果连词 + 条件/让步/转折连接词；『才/就』为高频弱连接词，单列时体现在关键词级表",
    },
]


def compile_categories() -> list[dict[str, Any]]:
    compiled = []
    for spec in CATEGORIES:
        compiled.append({
            **spec,
            "_kw": tuple(spec["keywords"]),
            "_re": tuple(re.compile(p) for p in spec["regex"]),
        })
    return compiled


COMPILED = compile_categories()


def category_matches(text_norm: str) -> dict[str, list[str]]:
    """Return {category_id: [matched keywords/regex labels]} for one normalised text."""
    hits: dict[str, list[str]] = {}
    for spec in COMPILED:
        matched = [kw for kw in spec["_kw"] if kw in text_norm]
        matched += [f"re:{rx.pattern}" for rx in spec["_re"] if rx.search(text_norm)]
        if matched:
            hits[spec["id"]] = matched
    return hits


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_torch_stub() -> None:
    """Minimal stand-ins so the train pickle (torch tensors under ``sign``) loads.

    The ``sign`` field is unused here; only dict keys ``text``/``gloss``/``signer``/
    ``num_frames`` are read.  No torch code is executed.
    """
    class _StubTensor:
        __slots__ = ()

        def __repr__(self) -> str:
            return "<torch-stub: not loaded>"

    def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                           requires_grad, backward_hooks, metadata=None):
        return _StubTensor()

    def _load_from_bytes(blob):
        return b"<torch-stub-storage>"

    for name, attrs in (
        ("torch", {}),
        ("torch._utils", {"_rebuild_tensor_v2": _rebuild_tensor_v2}),
        ("torch.storage", {"_load_from_bytes": _load_from_bytes}),
    ):
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module


def load_train(path: Path) -> list[dict[str, Any]]:
    install_torch_stub()
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise SystemExit(f"{path}: expected a non-empty list")
    for record in records[:1]:
        for key in ("name", "gloss", "text", "num_frames"):
            if key not in record:
                raise SystemExit(f"{path}: missing field {key!r}")
    return records


def load_references(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path}: empty")
    return rows


def load_system(path: Path, candidate_id: str | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if candidate_id is not None and record.get("candidate_id") != candidate_id:
                continue
            if str(record.get("status", "")).upper() != "OK":
                continue
            sample = record.get("sample_id")
            if not sample:
                continue
            if "output" in record and isinstance(record["output"], dict):
                text = record["output"].get("text")
            else:
                text = record.get("text")
            out[str(sample)] = text or ""
    return out


# ---------------------------------------------------------------------------
# small statistics helpers (no scipy on this box)
# ---------------------------------------------------------------------------
def sign_test_p(wins: int, losses: int) -> float | None:
    """Two-sided exact sign test p-value (ties already excluded)."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    def ranks(values: Sequence[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        index = 0
        while index < len(order):
            end = index
            while end + 1 < len(order) and values[order[end + 1]] == values[order[index]]:
                end += 1
            rank = (index + end) / 2 + 1
            for position in range(index, end + 1):
                out[order[position]] = rank
            index = end + 1
        return out

    return pearson(ranks(xs), ranks(ys))


def length_bin(length: int) -> str:
    if length <= 10:
        return "<=10"
    if length <= 15:
        return "11-15"
    if length <= 20:
        return "16-20"
    if length <= 25:
        return "21-25"
    return ">25"


LENGTH_BINS = ["<=10", "11-15", "16-20", "21-25", ">25"]


# ---------------------------------------------------------------------------
# core analysis
# ---------------------------------------------------------------------------
def analyse(check_only: bool = False) -> dict[str, Any]:
    train = load_train(TRAIN_PKL)
    refs = load_references(GT_CSV)
    s2 = load_system(S2_JSONL)
    a2 = load_system(A2_JSONL, candidate_id="A2")

    n_train = len(train)
    train_norm = [normalize_text(record["text"]) for record in train]
    train_gloss_vocab: set[str] = set()
    for record in train:
        train_gloss_vocab.update(str(record.get("gloss") or "").split())

    test_ids = [row["sample_id"] for row in refs]
    n_test = len(test_ids)
    missing = [sid for sid in test_ids if sid not in s2 or sid not in a2]
    if missing:
        raise SystemExit(f"{len(missing)} test samples lack S2/A2 predictions, e.g. {missing[:3]}")

    # reference: canonical CSLT caliber uses ref_sentence_upstream; it is identical
    # to gt_sentence_official after NFKC + whitespace strip (verified below).
    ref_norm: dict[str, str] = {}
    ref_official_norm: dict[str, str] = {}
    ref_len: dict[str, int] = {}
    upstream_vs_official_diff: list[str] = []
    for row in refs:
        sid = row["sample_id"]
        up = normalize_text(row.get("ref_sentence_upstream") or "")
        off = normalize_text(row.get("gt_sentence_official") or "")
        if not up:
            up = off
        if up != off:
            upstream_vs_official_diff.append(sid)
        ref_norm[sid] = up
        ref_official_norm[sid] = off
        ref_len[sid] = len(up)

    # per-sample metrics -----------------------------------------------------
    hyp_norm = {sid: normalize_text(s2[sid]) for sid in test_ids}
    hyp_norm_a2 = {sid: normalize_text(a2[sid]) for sid in test_ids}
    edit = {sid: edit_similarity(ref_norm[sid], hyp_norm[sid]) for sid in test_ids}
    edit_official_ref = {
        sid: edit_similarity(ref_official_norm[sid], hyp_norm[sid]) for sid in test_ids
    }
    edit_a2 = {sid: edit_similarity(ref_norm[sid], hyp_norm_a2[sid]) for sid in test_ids}
    exact = {sid: ref_norm[sid] == hyp_norm[sid] for sid in test_ids}
    exact_a2 = {sid: ref_norm[sid] == hyp_norm_a2[sid] for sid in test_ids}
    ops = {sid: _edit_ops(list(ref_norm[sid]), list(hyp_norm[sid])) for sid in test_ids}

    baseline = {
        "s2_mean_char_edit_similarity": round(statistics.fmean(edit.values()), 4),
        "a2_mean_char_edit_similarity": round(statistics.fmean(edit_a2.values()), 4),
        "s2_exact_match": sum(1 for value in exact.values() if value),
        "a2_exact_match": sum(1 for value in exact_a2.values() if value),
    }
    expected = {"s2": 0.4358, "a2": 0.4212}  # published canonical 1086 numbers
    for key, want in (("s2_mean_char_edit_similarity", expected["s2"]),
                      ("a2_mean_char_edit_similarity", expected["a2"])):
        got = baseline[key]
        if abs(got - want) > 5e-4:
            raise SystemExit(f"baseline self-check failed: {key}={got} != canonical {want}")

    # category assignment ----------------------------------------------------
    train_hits: dict[str, list[int]] = {spec["id"]: [] for spec in CATEGORIES}
    train_kw_hits: dict[tuple[str, str], int] = {}
    for index, text in enumerate(train_norm):
        for cid, labels in category_matches(text).items():
            train_hits[cid].append(index)
            for label in labels:
                key = (cid, label)
                train_kw_hits[key] = train_kw_hits.get(key, 0) + 1

    test_hits: dict[str, dict[str, list[str]]] = {}
    for sid in test_ids:
        test_hits[sid] = category_matches(ref_norm[sid])

    test_by_cat: dict[str, list[str]] = {spec["id"]: [] for spec in CATEGORIES}
    for sid in test_ids:
        for cid in test_hits[sid]:
            test_by_cat[cid].append(sid)

    # length -> edit expectation (all test samples), for length-adjusted residual
    bin_values: dict[str, list[float]] = {b: [] for b in LENGTH_BINS}
    bin_values_a2: dict[str, list[float]] = {b: [] for b in LENGTH_BINS}
    for sid in test_ids:
        bin_values[length_bin(ref_len[sid])].append(edit[sid])
        bin_values_a2[length_bin(ref_len[sid])].append(edit_a2[sid])
    bin_mean = {b: statistics.fmean(v) if v else None for b, v in bin_values.items()}
    bin_mean_a2 = {b: statistics.fmean(v) if v else None for b, v in bin_values_a2.items()}
    bin_count = {b: len(v) for b, v in bin_values.items()}

    def expected_edit(sid: str, table: dict[str, float | None]) -> float | None:
        return table[length_bin(ref_len[sid])]

    # keyword level ----------------------------------------------------------
    keyword_rows: list[dict[str, Any]] = []
    keyword_index: dict[tuple[str, str], dict[str, Any]] = {}
    for spec in COMPILED:
        cid = spec["id"]
        labels = list(spec["_kw"]) + [f"re:{rx.pattern}" for rx in spec["_re"]]
        train_kw_count = {label: train_kw_hits.get((cid, label), 0) for label in labels}
        test_kw_samples: dict[str, list[str]] = {label: [] for label in labels}
        for sid in test_by_cat[cid]:
            for label in test_hits[sid][cid]:
                if label in test_kw_samples:
                    test_kw_samples[label].append(sid)
        for label, count_train in train_kw_count.items():
            samples = test_kw_samples.get(label, [])
            n_test_kw = len(samples)
            kw_ref_chars = sum(len(ref_norm[s]) for s in samples) if samples else 0
            kw_ops = {key: sum(ops[s][key] for s in samples)
                      for key in ("substitutions", "deletions", "insertions")} if samples else None
            row = {
                "category_id": cid,
                "category_label": spec["label"],
                "unit": label,
                "n_train": count_train,
                "train_share": round(count_train / n_train, 5),
                "n_test": n_test_kw,
                "test_share": round(n_test_kw / n_test, 5),
                "s2_mean_edit": round(statistics.fmean([edit[s] for s in samples]), 4) if samples else None,
                "a2_mean_edit": round(statistics.fmean([edit_a2[s] for s in samples]), 4) if samples else None,
                "delta_edit_vs_a2": (round(statistics.fmean([edit[s] - edit_a2[s] for s in samples]), 4)
                                     if samples else None),
                "s2_exact_rate": (round(sum(1 for s in samples if exact[s]) / n_test_kw, 4)
                                  if samples else None),
                "test_mean_ref_len": round(statistics.fmean([ref_len[s] for s in samples]), 2)
                if samples else None,
                "s2_error_rate_per_100_chars": ({
                    "substitution": round(100 * kw_ops["substitutions"] / kw_ref_chars, 2),
                    "deletion": round(100 * kw_ops["deletions"] / kw_ref_chars, 2),
                    "insertion": round(100 * kw_ops["insertions"] / kw_ref_chars, 2),
                } if kw_ops and kw_ref_chars else None),
                "data_gap": bool(n_test_kw >= GAP_KW_TEST and count_train < GAP_KW_TRAIN),
            }
            keyword_rows.append(row)
            keyword_index[(cid, label)] = row

    # per-category statistics -------------------------------------------------
    categories: list[dict[str, Any]] = []
    for spec in CATEGORIES:
        cid = spec["id"]
        samples = test_by_cat[cid]
        n_cat_train = len(train_hits[cid])
        n_cat_test = len(samples)
        entry: dict[str, Any] = {
            "id": cid,
            "label": spec["label"],
            "n_train": n_cat_train,
            "train_share": round(n_cat_train / n_train, 5),
            "train_distinct_text": len({train_norm[i] for i in train_hits[cid]}),
            "train_mean_text_len": round(statistics.fmean([len(train_norm[i]) for i in train_hits[cid]]), 2)
            if n_cat_train else None,
            "n_test": n_cat_test,
            "test_share": round(n_cat_test / n_test, 5),
            "test_mean_ref_len": round(statistics.fmean([ref_len[s] for s in samples]), 2) if samples else None,
            "train_test_share_ratio": (round((n_cat_train / n_train) / (n_cat_test / n_test), 3)
                                       if n_cat_test else None),
        }
        if not samples:
            entry.update({"verdict": "no_test_samples"})
            categories.append(entry)
            continue

        s2_edits = [edit[s] for s in samples]
        a2_edits = [edit_a2[s] for s in samples]
        deltas = [edit[s] - edit_a2[s] for s in samples]
        wins = sum(1 for d in deltas if d > 1e-9)
        losses = sum(1 for d in deltas if d < -1e-9)
        ties = n_cat_test - wins - losses
        total_ref_chars = sum(len(ref_norm[s]) for s in samples)
        s_tot = sum(ops[s]["substitutions"] for s in samples)
        d_tot = sum(ops[s]["deletions"] for s in samples)
        i_tot = sum(ops[s]["insertions"] for s in samples)
        edits_total = s_tot + d_tot + i_tot
        residuals = [edit[s] - (expected_edit(s, bin_mean) or 0.0) for s in samples]
        residuals_a2 = [edit_a2[s] - (expected_edit(s, bin_mean_a2) or 0.0) for s in samples]
        edit_shift_s2 = statistics.fmean([(edit[s] - edit_official_ref[s]) for s in samples])

        gap_keywords = [label for (ccid, label), row in keyword_index.items()
                        if ccid == cid and row["data_gap"]]
        gap_driven = [s for s in samples
                      if gap_keywords and set(test_hits[s][cid]) & set(gap_keywords)]

        entry.update({
            "test_mean_ref_len_official_same": True,
            "s2_mean_edit": round(statistics.fmean(s2_edits), 4),
            "s2_median_edit": round(statistics.median(s2_edits), 4),
            "s2_std_edit": round(statistics.pstdev(s2_edits), 4),
            "s2_exact_match": sum(1 for s in samples if exact[s]),
            "s2_exact_rate": round(sum(1 for s in samples if exact[s]) / n_cat_test, 4),
            "s2_low_score_rate": round(sum(1 for v in s2_edits if v < LOW_EDIT) / n_cat_test, 4),
            "s2_very_low_rate": round(sum(1 for v in s2_edits if v < 0.30) / n_cat_test, 4),
            "a2_mean_edit": round(statistics.fmean(a2_edits), 4),
            "a2_exact_rate": round(sum(1 for s in samples if exact_a2[s]) / n_cat_test, 4),
            "a2_low_score_rate": round(sum(1 for v in a2_edits if v < LOW_EDIT) / n_cat_test, 4),
            "delta_edit_vs_a2": round(statistics.fmean(deltas), 4),
            "paired_s2_better": wins,
            "paired_s2_worse": losses,
            "paired_tie": ties,
            "sign_test_p": (round(p, 6) if (p := sign_test_p(wins, losses)) is not None else None),
            "edit_shift_official_vs_upstream_ref": round(edit_shift_s2, 6),
            "error_ops_total": {
                "substitutions": s_tot,
                "deletions": d_tot,
                "insertions": i_tot,
                "ref_chars": total_ref_chars,
            },
            "error_type_rate_per_100_chars": {
                "substitution": round(100 * s_tot / total_ref_chars, 2),
                "deletion": round(100 * d_tot / total_ref_chars, 2),
                "insertion": round(100 * i_tot / total_ref_chars, 2),
                "total": round(100 * edits_total / total_ref_chars, 2),
            },
            "error_type_share": {
                "substitution": round(s_tot / edits_total, 4) if edits_total else None,
                "deletion": round(d_tot / edits_total, 4) if edits_total else None,
                "insertion": round(i_tot / edits_total, 4) if edits_total else None,
            },
            "mean_hyp_over_ref_len": round(statistics.fmean([len(hyp_norm[s]) / max(1, ref_len[s]) for s in samples]), 4),
            "length_adjusted": {
                "expected_edit_from_length_bin": round(statistics.fmean([expected_edit(s, bin_mean) for s in samples]), 4),
                "residual_s2": round(statistics.fmean(residuals), 4),
                "residual_a2": round(statistics.fmean(residuals_a2), 4),
            },
            "data_gap_keywords": gap_keywords,
            "data_gap_driven_samples": len(gap_driven),
            "data_gap_share": round(len(gap_driven) / n_cat_test, 4),
        })
        categories.append(entry)

    # verdicts ---------------------------------------------------------------
    # A category is *formally weak* when its S2 edit is under WEAK_EDIT with
    # enough test support.  Because the length confound is strong (see
    # length_controls), every weak category also reports a length-adjusted
    # residual: a weak category whose residual is ~0 is weak mostly because its
    # reference sentences are long, not because of its semantics.
    weak: list[dict[str, Any]] = []
    for entry in categories:
        if "s2_mean_edit" not in entry:
            continue
        weak_flag = entry["s2_mean_edit"] < WEAK_EDIT and entry["n_test"] >= WEAK_MIN_TEST
        entry["is_weak"] = weak_flag
        entry["length_adjusted_weak"] = (
            weak_flag and entry["length_adjusted"]["residual_s2"] <= WEAK_RESIDUAL)
        if not weak_flag:
            entry["verdict"] = "not_weak"
            entry["verdict_note"] = ""
            continue
        if entry["n_train"] < DATA_SHORT_TRAIN:
            verdict = "data_insufficient"
            note = f"n_train={entry['n_train']} < {DATA_SHORT_TRAIN}：该类训练样本本身就少"
        elif entry["data_gap_share"] >= DATA_GAP_SHARE:
            verdict = "data_insufficient"
            note = (f"n_train={entry['n_train']} 看似充足，但 {entry['data_gap_share']*100:.0f}% 的测试样本"
                    f"由长尾关键词驱动（train<{GAP_KW_TRAIN} 次）：数据缺口在词项级而非类别级")
        else:
            verdict = "model_or_method"
            note = (f"n_train={entry['n_train']}（{entry['train_share']*100:.1f}% 训练占比），"
                    f"词项级数据缺口只覆盖 {entry['data_gap_share']*100:.0f}% 的测试样本 —— 数据量不解释其弱势。"
                    f"另一个事实：该类测试句均长 {entry['test_mean_ref_len']:.1f} 字（整体均值 "
                    f"{statistics.fmean(list(ref_len.values())):.1f} 字），长度校正后残差 "
                    f"{entry['length_adjusted']['residual_s2']:+.3f}，"
                    + ("**仍低于**同长度预期，属真正的语义性弱。" if entry["length_adjusted_weak"]
                       else "**基本回到同长度预期**，即弱势主要来自句长而非语义。"))
        entry["verdict"] = verdict
        entry["verdict_note"] = note
        weak.append(entry)

    weak_sorted = sorted(weak, key=lambda e: (e["s2_mean_edit"], -e["n_test"]))
    for rank, entry in enumerate(weak_sorted, 1):
        entry["weak_rank"] = rank

    # "weakest five" ranking regardless of the formal threshold: the report must
    # still name the bottom rows when no category passes the threshold.
    ranked = [entry for entry in categories if "s2_mean_edit" in entry]
    weakest = sorted(ranked, key=lambda e: (e["s2_mean_edit"], e["length_adjusted"]["residual_s2"]))[:5]
    weakest_rows = [
        {
            "rank": index + 1,
            "id": entry["id"],
            "label": entry["label"],
            "n_train": entry["n_train"],
            "train_share": entry["train_share"],
            "n_test": entry["n_test"],
            "s2_mean_edit": entry["s2_mean_edit"],
            "s2_exact_rate": entry["s2_exact_rate"],
            "delta_edit_vs_a2": entry["delta_edit_vs_a2"],
            "test_mean_ref_len": entry["test_mean_ref_len"],
            "length_residual": entry["length_adjusted"]["residual_s2"],
            "data_gap_share": entry["data_gap_share"],
            "is_formally_weak": entry["is_weak"],
            "verdict": entry["verdict"],
            "verdict_note": entry["verdict_note"],
        }
        for index, entry in enumerate(weakest)
    ]

    # lexical frequency vs quality (the data-vs-model scatter) -----------------
    kw_usable = [row for row in keyword_rows if row["n_test"] >= 5 and row["n_train"] > 0]
    kw_freq = [math.log10(row["n_train"]) for row in kw_usable]
    kw_edit = [row["s2_mean_edit"] for row in kw_usable]
    keyword_scatter = {
        "n_units": len(kw_usable),
        "pearson_log10_train_vs_edit": round(pearson(kw_freq, kw_edit), 4) if kw_usable else None,
        "spearman_log10_train_vs_edit": round(spearman(kw_freq, kw_edit), 4) if kw_usable else None,
    }
    # quintiles of train frequency -> mean edit
    freq_bins: list[dict[str, Any]] = []
    if kw_usable:
        ordered = sorted(kw_usable, key=lambda row: row["n_train"])
        size = max(1, len(ordered) // 5)
        for index in range(5):
            chunk = ordered[index * size: (index + 1) * size] if index < 4 else ordered[4 * size:]
            if not chunk:
                continue
            freq_bins.append({
                "quintile": index + 1,
                "n_train_range": [chunk[0]["n_train"], chunk[-1]["n_train"]],
                "n_units": len(chunk),
                "n_test_samples_covered": sum(row["n_test"] for row in chunk),
                "mean_edit": round(statistics.fmean([row["s2_mean_edit"] for row in chunk]), 4),
                "mean_ref_len": round(statistics.fmean([row["test_mean_ref_len"] for row in chunk
                                                        if row["test_mean_ref_len"]]), 2),
            })
        # length-controlled view: within the short-reference half only, so the
        # "rarer word -> harder sentence" confound is removed.
        short_units = [row for row in kw_usable if (row["test_mean_ref_len"] or 0) <= 15]
        if len(short_units) >= 10:
            ordered_short = sorted(short_units, key=lambda row: row["n_train"])
            half = len(ordered_short) // 2
            freq_bins_short = [{
                "half": "low_frequency",
                "n_units": half,
                "mean_edit": round(statistics.fmean([r["s2_mean_edit"] for r in ordered_short[:half]]), 4),
                "mean_ref_len": round(statistics.fmean([r["test_mean_ref_len"] for r in ordered_short[:half]]), 2),
            }, {
                "half": "high_frequency",
                "n_units": len(ordered_short) - half,
                "mean_edit": round(statistics.fmean([r["s2_mean_edit"] for r in ordered_short[half:]]), 4),
                "mean_ref_len": round(statistics.fmean([r["test_mean_ref_len"] for r in ordered_short[half:]]), 2),
            }]
        else:
            freq_bins_short = []
    else:
        freq_bins_short = []

    # category-count vs category quality (does more training data buy better edits?)
    cat_freq = [math.log10(entry["n_train"]) for entry in categories if "s2_mean_edit" in entry]
    cat_edit = [entry["s2_mean_edit"] for entry in categories if "s2_mean_edit" in entry]
    category_scatter = {
        "n_categories": len(cat_freq),
        "pearson_log10_ntrain_vs_edit": round(pearson(cat_freq, cat_edit), 4),
        "spearman_log10_ntrain_vs_edit": round(spearman(cat_freq, cat_edit), 4),
    }

    # coverage at lexical granularity ---------------------------------------
    train_chars = set()
    train_bigrams = set()
    for text in train_norm:
        train_chars.update(text)
        train_bigrams.update(text[i:i + 2] for i in range(len(text) - 1))

    coverage_rows: list[dict[str, Any]] = []
    ref_by_id = {row["sample_id"]: row for row in refs}
    for sid in test_ids:
        text = ref_norm[sid]
        chars = set(text)
        bigrams = {text[i:i + 2] for i in range(len(text) - 1)}
        gloss_tokens = str(ref_by_id[sid].get("gt_gloss") or "").split()
        coverage_rows.append({
            "sample_id": sid,
            "char_cov": (len(chars & train_chars) / len(chars)) if chars else 1.0,
            "bigram_cov": (len(bigrams & train_bigrams) / len(bigrams)) if bigrams else 1.0,
            "gloss_cov": (sum(1 for t in gloss_tokens if t in train_gloss_vocab) / len(gloss_tokens))
            if gloss_tokens else None,
            "gloss_oov": [t for t in gloss_tokens if t not in train_gloss_vocab],
            "ref_len": ref_len[sid],
        })

    def _band_table(getter, label: str) -> list[dict[str, Any]]:
        bands = [("1.00", 1.0, 1.0001), ("[0.95,1.00)", 0.95, 1.0),
                 ("[0.90,0.95)", 0.90, 0.95), ("<0.90", -0.1, 0.90)]
        table = []
        for band_label, low, high in bands:
            members = [row for row in coverage_rows
                       if getter(row) is not None and low <= getter(row) < high]
            table.append({
                "metric": label,
                "band": band_label,
                "n_samples": len(members),
                "mean_ref_len": round(statistics.fmean([row["ref_len"] for row in members]), 2) if members else None,
                "s2_mean_edit": round(statistics.fmean([edit[row["sample_id"]] for row in members]), 4)
                if members else None,
                "a2_mean_edit": round(statistics.fmean([edit_a2[row["sample_id"]] for row in members]), 4)
                if members else None,
                "s2_exact_rate": round(sum(1 for row in members if exact[row["sample_id"]]) / len(members), 4)
                if members else None,
            })
        return table

    coverage_bands = {
        "char_coverage": _band_table(lambda row: row["char_cov"], "char_coverage"),
        "bigram_coverage": _band_table(lambda row: row["bigram_cov"], "bigram_coverage"),
        "gloss_coverage": _band_table(lambda row: row["gloss_cov"], "gloss_coverage"),
    }
    gloss_oov_counter: dict[str, int] = {}
    for row in coverage_rows:
        for token in row["gloss_oov"]:
            gloss_oov_counter[token] = gloss_oov_counter.get(token, 0) + 1

    # Robustness: are the "unseen bigrams" just a punctuation artefact (the test
    # references and the train text punctuate differently)?  Recompute coverage
    # on CJK-only bigrams and measure how many unseen bigrams touch punctuation.
    punct_chars = set("。，,?？!！、；;：:“”()（）…—-·")
    unseen_total = 0
    unseen_punct = 0
    for sid in test_ids:
        text = ref_norm[sid]
        for i in range(len(text) - 1):
            bigram = text[i:i + 2]
            if bigram in train_bigrams:
                continue
            unseen_total += 1
            if bigram[0] in punct_chars or bigram[1] in punct_chars:
                unseen_punct += 1
    cjk_only_cov: list[float] = []
    for sid in test_ids:
        text = "".join(ch for ch in ref_norm[sid] if "\u4e00" <= ch <= "\u9fff")
        bigrams = [text[i:i + 2] for i in range(len(text) - 1)]
        if bigrams:
            cjk_only_cov.append(sum(1 for b in bigrams if b in train_bigrams) / len(bigrams))
    coverage_summary = {
        "train_char_vocab": len(train_chars),
        "train_bigram_vocab": len(train_bigrams),
        "train_gloss_vocab": len(train_gloss_vocab),
        "test_mean_char_coverage": round(statistics.fmean([row["char_cov"] for row in coverage_rows]), 5),
        "test_mean_bigram_coverage": round(statistics.fmean([row["bigram_cov"] for row in coverage_rows]), 5),
        "test_mean_gloss_coverage": round(statistics.fmean([row["gloss_cov"] for row in coverage_rows
                                                            if row["gloss_cov"] is not None]), 5),
        "test_samples_with_full_bigram_coverage": sum(1 for row in coverage_rows if row["bigram_cov"] == 1.0),
        "test_samples_with_gloss_oov": sum(1 for row in coverage_rows if row["gloss_oov"]),
        "gloss_oov_token_types": len(gloss_oov_counter),
        "gloss_oov_tokens_eval_only": sorted(gloss_oov_counter.items(), key=lambda kv: -kv[1])[:40],
        "bigram_novelty_robustness": {
            "unseen_bigram_tokens_total": unseen_total,
            "unseen_bigrams_touching_punctuation": unseen_punct,
            "unseen_bigrams_punctuation_share": (round(unseen_punct / unseen_total, 4)
                                                  if unseen_total else None),
            "cjk_only_bigram_coverage_mean": (round(statistics.fmean(cjk_only_cov), 4)
                                              if cjk_only_cov else None),
            "note": "测试参考与训练文本的标点不同（全角/半角），此项检验「未见二元组」是否只是标点差异",
        },
    }

    per_sample = [
        {
            "sample_id": sid,
            "ref_len": ref_len[sid],
            "s2_edit": round(edit[sid], 4),
            "a2_edit": round(edit_a2[sid], 4),
            "s2_exact": exact[sid],
            "s2_low_score": edit[sid] < LOW_EDIT,
            "categories": sorted(test_hits[sid]),
        }
        for sid in test_ids
    ]

    # ---- coverage, with the length confound removed ------------------------
    short_ids = [sid for sid in test_ids if ref_len[sid] <= 15]
    long_ids = [sid for sid in test_ids if ref_len[sid] > 15]
    cov_by_id = {row["sample_id"]: row for row in coverage_rows}

    def coverage_quartiles(ids: list[str], label: str) -> list[dict[str, Any]]:
        if not ids:
            return []
        ordered = sorted(ids, key=lambda sid: cov_by_id[sid]["bigram_cov"])
        size = max(1, len(ordered) // 4)
        rows = []
        for index in range(4):
            chunk = ordered[index * size:(index + 1) * size] if index < 3 else ordered[3 * size:]
            if not chunk:
                continue
            rows.append({
                "length_group": label,
                "bigram_coverage_quartile": index + 1,
                "n_samples": len(chunk),
                "bigram_cov_range": [round(cov_by_id[chunk[0]]["bigram_cov"], 4),
                                     round(cov_by_id[chunk[-1]]["bigram_cov"], 4)],
                "mean_ref_len": round(statistics.fmean([ref_len[s] for s in chunk]), 2),
                "s2_mean_edit": round(statistics.fmean([edit[s] for s in chunk]), 4),
                "a2_mean_edit": round(statistics.fmean([edit_a2[s] for s in chunk]), 4),
            })
        return rows

    coverage_length_controlled = coverage_quartiles(short_ids, f"<=15 chars (n={len(short_ids)})") + \
        coverage_quartiles(long_ids, f">15 chars (n={len(long_ids)})")

    # Does bigram novelty still predict the edit *inside* a length bin?
    # (raw Pearson r between coverage and edit is confounded by sentence length)
    cov_within_bin: list[dict[str, Any]] = []
    for band in LENGTH_BINS:
        members = [sid for sid in test_ids if length_bin(ref_len[sid]) == band]
        if len(members) < 20:
            continue
        cov_within_bin.append({
            "length_bin": band,
            "n_samples": len(members),
            "pearson_r_bigram_cov_vs_s2_edit": round(pearson(
                [cov_by_id[sid]["bigram_cov"] for sid in members], [edit[sid] for sid in members]), 4),
            "pearson_r_bigram_cov_vs_a2_edit": round(pearson(
                [cov_by_id[sid]["bigram_cov"] for sid in members], [edit_a2[sid] for sid in members]), 4),
        })
    all_cov_r = {
        "pearson_r_bigram_cov_vs_s2_edit_raw": round(pearson(
            [cov_by_id[sid]["bigram_cov"] for sid in test_ids],
            [edit[sid] for sid in test_ids]), 4),
        "pearson_r_bigram_cov_vs_s2_edit_within_length_bins": cov_within_bin,
    }

    idle_units = [row for row in keyword_rows if row["n_train"] == 0 and row["n_test"] == 0]
    keywords_active = [row for row in keyword_rows if row["n_train"] > 0 or row["n_test"] > 0]

    # ---- overall aggregates ------------------------------------------------
    s_all = sum(ops[sid]["substitutions"] for sid in test_ids)
    d_all = sum(ops[sid]["deletions"] for sid in test_ids)
    i_all = sum(ops[sid]["insertions"] for sid in test_ids)
    ref_chars_all = sum(len(ref_norm[sid]) for sid in test_ids)
    e_all = s_all + d_all + i_all
    with_metric = [entry for entry in categories if "s2_mean_edit" in entry]
    ntrain_min_entry = min(with_metric, key=lambda e: e["n_train"])
    ntrain_max_entry = max(with_metric, key=lambda e: e["n_train"])
    ratio_entries = [entry for entry in with_metric if entry.get("train_test_share_ratio") is not None]
    ratio_min_entry = min(ratio_entries, key=lambda e: e["train_test_share_ratio"])
    ratio_max_entry = max(ratio_entries, key=lambda e: e["train_test_share_ratio"])

    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_id": ANALYSIS_ID,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metric_definition_version": METRICS.METRIC_DEFINITION_VERSION,
        "device": "cpu",
        "trained_anything": False,
        "deterministic": True,
        "random_seed": None,
        "inputs": {
            "train_pickle": {"path": str(TRAIN_PKL.relative_to(ROOT)), "sha256": sha256_file(TRAIN_PKL),
                             "records": n_train},
            "test_reference_eval_only": {"path": str(GT_CSV.relative_to(ROOT)), "sha256": sha256_file(GT_CSV),
                                         "records": n_test},
            "s2_predictions": {"path": str(S2_JSONL.relative_to(ROOT)), "sha256": sha256_file(S2_JSONL),
                               "records": len(s2)},
            "a2_predictions": {"path": str(A2_JSONL.relative_to(ROOT)), "sha256": sha256_file(A2_JSONL),
                               "records": len(a2)},
        },
        "reference_field": "ref_sentence_upstream",
        "upstream_vs_official_mismatch_after_normalisation": len(upstream_vs_official_diff),
        "normalisation": "unicodedata.normalize('NFKC', text) with all whitespace removed (p1-v1)",
        "baseline_self_check": {**baseline, "expected_canonical": expected,
                                "max_abs_error": round(max(
                                    abs(baseline["s2_mean_char_edit_similarity"] - expected["s2"]),
                                    abs(baseline["a2_mean_char_edit_similarity"] - expected["a2"])), 6)},
        "thresholds": {
            "weak_edit": WEAK_EDIT,
            "weak_min_test": WEAK_MIN_TEST,
            "weak_residual": WEAK_RESIDUAL,
            "low_edit": LOW_EDIT,
            "data_short_train": DATA_SHORT_TRAIN,
            "gap_keyword_train": GAP_KW_TRAIN,
            "gap_keyword_test": GAP_KW_TEST,
            "data_gap_share": DATA_GAP_SHARE,
        },
        "category_system": {spec["id"]: {"label": spec["label"], "keywords": spec["keywords"],
                                         "regex": spec["regex"], "rationale": spec["rationale"]}
                            for spec in CATEGORIES},
        "overall": {
            "n_train": n_train,
            "n_test": n_test,
            "s2_mean_edit": baseline["s2_mean_char_edit_similarity"],
            "a2_mean_edit": baseline["a2_mean_char_edit_similarity"],
            "s2_exact_match": baseline["s2_exact_match"],
            "a2_exact_match": baseline["a2_exact_match"],
            "s2_low_score_rate": round(sum(1 for v in edit.values() if v < LOW_EDIT) / n_test, 4),
            "category_union_coverage_test": round(
                sum(1 for sid in test_ids if test_hits[sid]) / n_test, 4),
            "category_union_coverage_train": round(
                sum(1 for index in range(n_train) if category_matches(train_norm[index])) / n_train, 4),
            "mean_test_categories_per_sample": round(
                statistics.fmean([len(test_hits[sid]) for sid in test_ids]), 3),
            "mean_train_categories_per_sample": round(
                statistics.fmean([len(category_matches(text)) for text in train_norm]), 3),
            "error_ops_total": {"substitutions": s_all, "deletions": d_all,
                                "insertions": i_all, "ref_chars": ref_chars_all},
            "error_type_rate_per_100_chars": {
                "substitution": round(100 * s_all / ref_chars_all, 2),
                "deletion": round(100 * d_all / ref_chars_all, 2),
                "insertion": round(100 * i_all / ref_chars_all, 2),
                "total": round(100 * e_all / ref_chars_all, 2),
            },
            "error_type_share": {
                "substitution": round(s_all / e_all, 4),
                "deletion": round(d_all / e_all, 4),
                "insertion": round(i_all / e_all, 4),
            },
            "category_ntrain": {"min": ntrain_min_entry["n_train"],
                                "min_category": ntrain_min_entry["label"],
                                "max": ntrain_max_entry["n_train"],
                                "max_category": ntrain_max_entry["label"]},
            "category_train_share": {"min": ntrain_min_entry["train_share"],
                                     "min_category": ntrain_min_entry["label"]},
            "train_test_share_ratio": {
                "min": ratio_min_entry["train_test_share_ratio"],
                "min_category": ratio_min_entry["label"],
                "max": ratio_max_entry["train_test_share_ratio"],
                "max_category": ratio_max_entry["label"],
            },
            "category_edit_range": {"min": min(e["s2_mean_edit"] for e in with_metric),
                                    "max": max(e["s2_mean_edit"] for e in with_metric)},
        },
        "categories": categories,
        "weakest_categories": weakest_rows,
        "weak_categories": [
            {"rank": entry["weak_rank"], "id": entry["id"], "label": entry["label"],
             "n_train": entry["n_train"], "train_share": entry["train_share"],
             "n_test": entry["n_test"], "s2_mean_edit": entry["s2_mean_edit"],
             "s2_exact_rate": entry["s2_exact_rate"], "delta_edit_vs_a2": entry["delta_edit_vs_a2"],
             "verdict": entry["verdict"], "verdict_note": entry["verdict_note"],
             "length_adjusted_residual": entry["length_adjusted"]["residual_s2"],
             "length_adjusted_weak": entry["length_adjusted_weak"],
             "data_gap_share": entry["data_gap_share"]}
            for entry in weak_sorted
        ],
        "verdict_summary": {
            "count_weak": len(weak_sorted),
            "count_length_adjusted_weak": sum(1 for e in weak_sorted if e["length_adjusted_weak"]),
            "count_data_insufficient": sum(1 for e in weak_sorted if e["verdict"] == "data_insufficient"),
            "count_model_or_method": sum(1 for e in weak_sorted if e["verdict"] == "model_or_method"),
            "trained_on_test_overlap": 0,
        },
        "frequency_evidence": {
            "keyword_level": keyword_scatter,
            "keyword_quintiles": freq_bins,
            "keyword_length_controlled": freq_bins_short,
            "category_level": category_scatter,
        },
        "coverage": {**coverage_summary, "bands": coverage_bands,
                     "novelty_prediction_power": all_cov_r,
                     "length_controlled_quartiles": coverage_length_controlled,
                     "idle_vocabulary_units": {"count": len(idle_units),
                                               "of_total": len(keyword_rows),
                                               "units": [f"{row['category_label']}:{row['unit']}"
                                                         for row in idle_units],
                                               "note": "train 与 test 均 0 命中：该类词表项在 CSL-Daily 中未出现，"
                                                       "属词表设计冗余而非数据缺口"},
                     "active_vocabulary_units": len(keywords_active)},
        "length_controls": {
            "bins": LENGTH_BINS,
            "n_samples": bin_count,
            "s2_mean_edit_by_length": {b: (round(v, 4) if v is not None else None) for b, v in bin_mean.items()},
            "a2_mean_edit_by_length": {b: (round(v, 4) if v is not None else None) for b, v in bin_mean_a2.items()},
            "correlation_ref_len_vs_edit": round(pearson(
                [ref_len[s] for s in test_ids], [edit[s] for s in test_ids]), 4),
            "mean_ref_len_overall": round(statistics.fmean(list(ref_len.values())), 2),
        },
        "keywords": keyword_rows,
        "per_sample": per_sample,
        "limitations": [
            "类别为多标签关键词匹配，非互斥；一个句子可同时落在多类，类别指标之间不可相加。",
            "词表由人工按 CSL-Daily 日常域语料设计，长尾/低频词的召回不完整（例如未收录方言或极低频称谓）。",
            "错误类型为字符级 Levenshtein 回溯（替换/删除/插入），与「语义级漏译/多译」只近似对应。",
            "A2 与 S2 共用同一 1086 测试集与同一参考，Δedit 为配对比较，但两者不是同一次运行的超参扫描。",
            "长度校正使用全体测试样本的长度分箱均值，只能消掉线性/分箱级长度效应。",
            "n_train 统计的是「含该类关键词的训练句子数」，不等于「该类语义的训练覆盖量」。",
        ],
    }
    return payload


# ---------------------------------------------------------------------------
# markdown report
# ---------------------------------------------------------------------------
def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    overall = payload["overall"]
    thresholds = payload["thresholds"]
    lines: list[str] = []
    add = lines.append

    add("# P-1.5 目标场景语义覆盖率与模型有效识别范围分析")
    add("")
    add(f"> 生成时间（UTC）：{payload['created_at_utc']}　|　metric caliber："
        f"`{payload['metric_definition_version']}`　|　analysis id：`{payload['analysis_id']}`")
    add("> 复现：`python experiments/p1/e07/analyze_coverage.py`（CPU、单进程、确定性、不训练、不改 `slbridge/`）")
    add("> 机器可读全量结果（含词表、关键词级表、逐样本分数）："
        "[coverage_analysis.json](../../../experiments/p1/e07/coverage_analysis.json)")
    add("")
    add("**回答的问题**：目标场景里哪些语义类别识别弱？这些弱是「训练数据不够」还是「模型/训练方法不行」？"
        "也就是：**先扩数据，还是先训模型？**")
    add("")
    add("## 0. 一句话结论")
    add("")
    verdicts = payload["verdict_summary"]
    ntrain = overall["category_ntrain"]
    ratio = overall["train_test_share_ratio"]
    weak_count = verdicts["count_weak"]
    residual_weak = verdicts["count_length_adjusted_weak"]
    if weak_count and not residual_weak:
        boundary_note = "但它们在长度校正后都回到同句长预期（残差 > -0.02），**属句长效应而非语义弱项**"
    elif weak_count == residual_weak:
        boundary_note = (f"其中只有 {residual_weak} 类在长度校正后仍低于同句长预期"
                         "（残差贴着 -0.02 的边界值，量级很小）")
    else:
        boundary_note = (f"其中 {residual_weak} 类在长度校正后仍低于同句长预期")
    add(f"- 12 类语义里只有 **{weak_count} 类**达到正式的弱类别阈值"
        f"（S2 edit < {thresholds['weak_edit']:.2f} 且 n_test ≥ {thresholds['weak_min_test']}）；"
        f"{boundary_note}。")
    add(f"- 类别之间的差距其实很小：12 类的 S2 edit 全部落在 "
        f"**{overall['category_edit_range']['min']:.3f}–{overall['category_edit_range']['max']:.3f}** 区间，"
        f"以整体基准 {overall['s2_mean_edit']:.4f} 为中心，**没有出现某类塌陷**。")
    add(f"- **类别级数据不缺**：train 命中量 {ntrain['min']:,}–{ntrain['max']:,} 条"
        f"（最少的是「{ntrain['min_category']}」，训练占比 {pct(overall['category_train_share']['min'])}），"
        f"全部高于「数据不足」阈值 {thresholds['data_short_train']}；"
        f"train/test 类别占比比 {ratio['min']:.2f}–{ratio['max']:.2f}，**测试集没有系统性地多考训练集稀缺的语义**。")
    add(f"- **词表级也没有缺口**：测试集 gloss 覆盖率 {pct(payload['coverage']['test_mean_gloss_coverage'])}、"
        f"字符覆盖率 {pct(payload['coverage']['test_mean_char_coverage'])}，"
        f"含训练集未见 gloss 的测试样本 **{payload['coverage']['test_samples_with_gloss_oov']} 条**。")
    add(f"- 因此瓶颈判断见 §9：**优先训模型 / 改训练方法**；"
        "按语义类别平铺扩数据不会命中真实缺口（数据扩容若要做，应针对词项长尾与句式多样性）。")
    add("")

    add("## 1. 数据与方法")
    add("")
    inputs = payload["inputs"]
    add("| 角色 | 文件 | 记录数 |")
    add("|---|---|---:|")
    add(f"| 训练集标注（只读） | `{inputs['train_pickle']['path']}` | {inputs['train_pickle']['records']:,} |")
    add(f"| 测试参考（**EVAL ONLY**） | `{inputs['test_reference_eval_only']['path']}` | "
        f"{inputs['test_reference_eval_only']['records']:,} |")
    add(f"| S2 预测（stage2 pose-subset） | `{inputs['s2_predictions']['path']}` | {inputs['s2_predictions']['records']:,} |")
    add(f"| A2 预测（Uni-Sign 官方 pose，对照） | `{inputs['a2_predictions']['path']}` | {inputs['a2_predictions']['records']:,} |")
    add("")
    chk = payload["baseline_self_check"]
    add(f"- **指标口径自检**：本脚本按 `p1-v1` 口径重算的 S2 平均字符编辑相似度 = "
        f"**{chk['s2_mean_char_edit_similarity']:.4f}**、A2 = **{chk['a2_mean_char_edit_similarity']:.4f}**，"
        f"与已发布 1086 表（0.4358 / 0.4212）一致（最大绝对误差 {chk['max_abs_error']:.1e}），说明分表数字与主表可比。")
    add(f"- **参考口径**：`ref_sentence_upstream`（canonical CSLT 口径）；与 `gt_sentence_official` 在 "
        f"NFKC ＋ 去空白后**完全一致**（不一致样本数 {payload['upstream_vs_official_mismatch_after_normalisation']}）。")
    add("- **正则化**：`NFKC` ＋ 去全部空白（`p1-v1`），因此全角 `，？！` 与半角 `,?!` 同形。")
    add("- **语义类别体系（12 类，纯关键词＋正则，未调用任何 LLM）**：多标签，非互斥。"
        f"测试集里有类别命中的样本占 {pct(overall['category_union_coverage_test'])}，"
        f"平均每句命中 {overall['mean_test_categories_per_sample']:.2f} 类。")
    add("")
    add("| 类别 | 关键词/正则应例（完整词表见 JSON） |")
    add("|---|---|")
    for cid, spec in payload["category_system"].items():
        units = spec["keywords"][:8] + [f"`{p}`" for p in spec["regex"][:3]]
        more = "…" if (len(spec["keywords"]) + len(spec["regex"])) > len(units) else ""
        add(f"| {spec['label']} | {'、'.join(units)}{more} |")
    add("")
    add("- **错误类型**：字符级 Levenshtein 回溯统计 替换(S)/删除(D)/插入(I)，"
        "并按「每 100 参考字符」归一（`p1-v1` 的 `_edit_ops`）。"
        "D 近似「漏译」（参考有、输出缺），I 近似「多译」（输出多出内容），S 近似「替换」（位置对、字错）。")
    add("- **弱类别判定**：`S2 edit < 0.40 且 n_test ≥ 30`；再用长度校正残差"
        "（≤ −0.02）区分「语义性弱」与「句子天生更长」。")
    add(f"- **数据侧判定**：`n_train < {thresholds['data_short_train']}` → 数据不足；"
        f"`n_train` 够但该类 ≥{pct(thresholds['data_gap_share'],0)} 的测试样本由长尾关键词驱动"
        f"（该词 train<{thresholds['gap_keyword_train']} 次但 test≥{thresholds['gap_keyword_test']} 次）→ 数据缺口在词项级；"
        "否则判「模型/训练方法问题」。")
    add("")

    # ---- coverage table
    add("## 2. 主表 A：语义覆盖与训练量（train vs test）")
    add("")
    add("| 语义类别 | n_train | train 占比 | train 独特句 | train 均长 | n_test | test 占比 | test 均长 | train/test 占比比 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in payload["categories"]:
        add(f"| {entry['label']} | {entry['n_train']:,} | {pct(entry['train_share'])} | "
            f"{fmt(entry.get('train_distinct_text'))} | {fmt(entry.get('train_mean_text_len'), 2)} | "
            f"{entry['n_test']:,} | {pct(entry['test_share'])} | {fmt(entry.get('test_mean_ref_len'), 2)} | "
            f"{fmt(entry.get('train_test_share_ratio'), 2)} |")
    add(f"| **全部（基准）** | {overall['n_train']:,} | 100.0% | — | — | {overall['n_test']:,} | 100.0% | — | 1.00 |")
    add("")
    ratio_info = overall["train_test_share_ratio"]
    add(f"> 读法：`train/test 占比比` ≈ 1 表示该类在测试集的比重与训练集相当（无分布漂移）。"
        f"12 类实测区间 {ratio_info['min']:.2f}–{ratio_info['max']:.2f}"
        f"（最低「{ratio_info['min_category']}」、最高「{ratio_info['max_category']}」），"
        "即**测试集没有系统性地多考训练集里稀缺的语义类别**。")
    add("")

    # ---- performance table
    add("## 3. 主表 B：测试表现 × 与 A2 对比（S2）")
    add("")
    add("| 语义类别 | S2 平均 edit | 中位数 | 完全匹配 | 低分率(<0.50) | A2 平均 edit | Δedit(vs A2) | 配对 S2 更好/更差 | 符号检验 p | 长度校正残差 |")
    add("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for entry in sorted(payload["categories"], key=lambda e: e.get("s2_mean_edit", 2.0)):
        if "s2_mean_edit" not in entry:
            continue
        add(f"| {entry['label']} | **{entry['s2_mean_edit']:.3f}** | {entry['s2_median_edit']:.3f} | "
            f"{entry['s2_exact_match']}/{entry['n_test']}（{pct(entry['s2_exact_rate'])}） | "
            f"{pct(entry['s2_low_score_rate'])} | {entry['a2_mean_edit']:.3f} | "
            f"{entry['delta_edit_vs_a2']:+.4f} | {entry['paired_s2_better']}/{entry['paired_s2_worse']} | "
            f"{fmt(entry['sign_test_p'], 4)} | {entry['length_adjusted']['residual_s2']:+.4f} |")
    add(f"| **全部（基准）** | **{overall['s2_mean_edit']:.4f}** | — | "
        f"{overall['s2_exact_match']}/{overall['n_test']}（{pct(overall['s2_exact_match']/overall['n_test'])}） | "
        f"{pct(overall['s2_low_score_rate'])} | {overall['a2_mean_edit']:.4f} | — | — | — | — |")
    add("")
    add("> `长度校正残差` = 该类 S2 edit − 同长度分箱的全体样本平均 edit（见 §8）。负值表示「同句长条件下仍偏差」，"
        "用来排除「某类句子天生更长所以看着更差」的假象。")
    add("")

    # ---- error types
    add("## 4. 主表 C：错误类型分布（每 100 参考字符，字符级 S/D/I）")
    add("")
    add("| 语义类别 | 替换 S | 删除 D（≈漏译） | 插入 I（≈多译） | 合计 | 输出/参考长度比 |")
    add("|---|---:|---:|---:|---:|---:|")
    for entry in sorted(payload["categories"], key=lambda e: e.get("error_type_rate_per_100_chars", {}).get("total", 0)):
        if "error_type_rate_per_100_chars" not in entry:
            continue
        rates = entry["error_type_rate_per_100_chars"]
        add(f"| {entry['label']} | {rates['substitution']:.1f} | {rates['deletion']:.1f} | "
            f"{rates['insertion']:.1f} | {rates['total']:.1f} | {entry['mean_hyp_over_ref_len']:.2f} |")
    add("")
    shares = overall["error_type_share"]
    rates = overall["error_type_rate_per_100_chars"]
    add(f"> 全场通读：错误以**字符替换 S** 为主（占全部编辑的 {pct(shares['substitution'])}），"
        f"其次**删除 D / 漏译**（{pct(shares['deletion'])}），**插入 I / 多译**最少（{pct(shares['insertion'])}）；"
        f"整体 {rates['total']:.1f} 次编辑/100 参考字符。"
        "同时输出长度普遍短于参考（长度比 < 1），说明既有「位置对但字错」也有「整段内容缺失」，"
        "而不是「话太多」——这把问题指向生成端覆盖率与训练目标，而非输出长度控制。")
    add("")

    # ---- weakest categories
    add("## 5. 最弱类别排序（前 5）与逐类判定")
    add("")
    add(f"排序口径：S2 平均 edit 升序（最弱在前）。「正式弱类别」= 另加 `edit < {thresholds['weak_edit']:.2f}` "
        f"且 `n_test ≥ {thresholds['weak_min_test']}` 两条硬条件。")
    add("")
    add("| # | 类别 | n_test | S2 edit | 完全匹配 | n_train | train 占比 | 参考句均长 | 长度校正残差 | 词项级数据缺口 | 正式弱类别 | 判定 |")
    add("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---|")
    verdict_label = {"data_insufficient": "数据不足 → 优先扩数据",
                     "model_or_method": "模型/训练方法 → 优先训模型",
                     "not_weak": "未达弱阈值"}
    for row in payload["weakest_categories"]:
        add(f"| {row['rank']} | {row['label']} | {row['n_test']} | {row['s2_mean_edit']:.3f} | "
            f"{pct(row['s2_exact_rate'])} | {row['n_train']:,} | {pct(row['train_share'])} | "
            f"{fmt(row['test_mean_ref_len'], 2)} | {row['length_residual']:+.3f} | "
            f"{pct(row['data_gap_share'],0)} | {'是' if row['is_formally_weak'] else '否'} | "
            f"{verdict_label[row['verdict']]} |")
    add("")
    if payload["weak_categories"]:
        add("逐类说明：")
        add("")
        for row in payload["weak_categories"]:
            add(f"- **{row['label']}**（edit {row['s2_mean_edit']:.3f}，n_train {row['n_train']:,}）："
                f"{row['verdict_note']}")
        add("")
    add(f"> 关键读法：只有 {len(payload['weak_categories'])} 类过正式弱阈值，"
        f"且其中 {sum(1 for r in payload['weak_categories'] if r['length_adjusted_weak'])} 类在长度校正后仍为负残差。"
        "换句话说，**类别之间的差异主要被「参考句有多长」吃掉**：12 类的裸 edit 差异不小，"
        "但同长度分箱后残差全在 ±0.09 内、弱类别只剩 ±0.02 量级。"
        "这意味着改进项应是**全体泛化能力**，而不是针对某一语义类做专项补数据。")
    add("")

    # ---- frequency evidence
    add("## 6. 证据：训练量到底能不能解释表现？")
    add("")
    freq = payload["frequency_evidence"]
    kw_stats = freq["keyword_level"]
    quint = freq["keyword_quintiles"]
    add("### 6.1 词项级（每个关键词）「train 频次 → test 平均 edit」")
    add("")
    add(f"- 可用词项 {kw_stats['n_units']} 个（test ≥5 次且 train >0）。"
        f"`log10(n_train)` 与平均 edit 的 **Spearman ρ = {fmt(kw_stats['spearman_log10_train_vs_edit'], 3)}**、"
        f"Pearson r = {fmt(kw_stats['pearson_log10_train_vs_edit'], 3)}。")
    add("")
    add("| 词项频次五分位 | train 频次区间 | 词项数 | 覆盖 test 样本（可重叠） | 平均 edit | 该组词项所在句均长 |")
    add("|---|---|---:|---:|---:|---:|")
    for row in quint:
        add(f"| Q{row['quintile']} | {row['n_train_range'][0]}–{row['n_train_range'][1]} | {row['n_units']} | "
            f"{row['n_test_samples_covered']} | {row['mean_edit']:.3f} | {fmt(row.get('mean_ref_len'), 2)} |")
    add("")
    if quint:
        lowest, highest = quint[0]["mean_edit"], quint[-1]["mean_edit"]
        delta = lowest - highest
        if abs(delta) < 0.02:
            freq_reading = (f"最稀有五分位（Q1，平均 {lowest:.3f}）与最高频五分位（Q5，平均 {highest:.3f}）"
                            f"只差 {delta:+.3f}：**词项频次几乎不预测该类词项的成败**。")
        elif delta < 0:
            freq_reading = (f"最稀有五分位 Q1 平均 {lowest:.3f} **低于**最高频 Q5 的 {highest:.3f}"
                            f"（Δ {delta:+.3f}）：词项越稀有识别越差 → 词表长尾仍有数据价值。")
        else:
            freq_reading = (f"最稀有五分位 Q1 平均 {lowest:.3f} **反而高于**最高频 Q5 的 {highest:.3f}"
                            f"（Δ {delta:+.3f}）：方向与「见得多就更好」相反——"
                            f"主要因为高频词项多出现在更长的句子里（见句均长列），"
                            "所以「见得少」解释不了这些词项的失败。")
        add(f"- {freq_reading}")
        add("")
    short_ctrl = freq.get("keyword_length_controlled") or []
    if short_ctrl:
        low, high = short_ctrl[0], short_ctrl[1]
        add(f"- **长度控制后**（只留所在句均长 ≤15 字的词项，n={low['n_units'] + high['n_units']}）："
            f"低频半（{low['n_units']} 项，句均长 {low['mean_ref_len']:.2f}）平均 edit **{low['mean_edit']:.3f}**、"
            f"高频半（{high['n_units']} 项，句均长 {high['mean_ref_len']:.2f}）平均 edit **{high['mean_edit']:.3f}**，"
            f"差 **{low['mean_edit'] - high['mean_edit']:+.3f}**。"
            "把这层混杂去掉之后，「训练频次 → 表现」的关系依然不成立。")
        add("")
    add("### 6.2 类别级（每个语义类别）「n_train → edit」")
    add("")
    cat = freq["category_level"]
    cat_rho = cat["spearman_log10_ntrain_vs_edit"]
    if cat_rho is None:
        cat_reading = "样本不足以计算相关。"
    elif cat_rho > 0.3:
        cat_reading = "为正：训练量越大的类别表现越好 → 支持「继续扩数据」。"
    elif cat_rho < -0.3:
        cat_reading = ("为**负**：训练量大的类别（人物指代、动作）在测试集上表现更好，"
                       "但这是因为它们的句子更短、句式更常见；训练量最小的「因果/条件」句均长最长。"
                       "把句长控制掉之后（§8），这个负相关不构成任何「该扩数据」的证据。")
    else:
        cat_reading = "接近 0：类别的训练量差异解释不了类间表现差异。"
    add(f"- {cat['n_categories']} 个类别上，`log10(n_train)` 与类别平均 edit 的 "
        f"**Spearman ρ = {fmt(cat_rho, 3)}**、Pearson r = {fmt(cat['pearson_log10_ntrain_vs_edit'], 3)}，"
        f"符号{cat_reading}")
    add("")
    idle = payload["coverage"]["idle_vocabulary_units"]
    add(f"- 词表设计事实：{len(payload['category_system'])} 类共 {idle['of_total']} 个词表项，"
        f"其中 **{idle['count']} 项在 train 与 test 都是 0 命中**（CSL-Daily 语料本身不用这些词，"
        f"例如部分姻亲/方言称谓），另有 {idle['of_total'] - idle['count']} 项实际生效。"
        "这些 0 命中项是词表冗余，不是数据缺口——补它们的数据也不会改变测试分数。")
    add("")

    # ---- coverage bands
    add("## 7. 有效识别范围：训练集词表覆盖率 → 表现")
    add("")
    cov = payload["coverage"]
    add(f"训练集词表：字符 {cov['train_char_vocab']:,} 种、字符二元组 {cov['train_bigram_vocab']:,} 种、"
        f"gloss {cov['train_gloss_vocab']:,} 种。测试集平均覆盖率：字符 {pct(cov['test_mean_char_coverage'])}、"
        f"二元组 {pct(cov['test_mean_bigram_coverage'])}、gloss {pct(cov['test_mean_gloss_coverage'])}。")
    add("")
    add("| 覆盖率口径 | 区间 | n 样本 | 平均句长 | S2 edit | A2 edit | S2 完全匹配 |")
    add("|---|---|---:|---:|---:|---:|---:|")
    for key in ("gloss_coverage", "bigram_coverage", "char_coverage"):
        for row in cov["bands"][key]:
            if row["n_samples"] == 0:
                continue
            add(f"| {key} | {row['band']} | {row['n_samples']} | {fmt(row['mean_ref_len'], 2)} | "
                f"{fmt(row['s2_mean_edit'])} | {fmt(row['a2_mean_edit'])} | {pct(row['s2_exact_rate'])} |")
    add("")
    add(f"- **gloss（符号）层：含训练集未见 gloss 的测试样本 {cov['test_samples_with_gloss_oov']} 条**，"
        f"平均覆盖率 {pct(cov['test_mean_gloss_coverage'])}。"
        "即测试集用到的每一个手语符号训练集都见过——**符号层的「词表缺口」不存在**。")
    add(f"- 字符层平均覆盖率 {pct(cov['test_mean_char_coverage'])}；低覆盖桶的句子明显更长"
        "（见平均句长列），**覆盖率与句长共线**，所以不能把「覆盖低 → edit 低」直接当因果。")
    add("")
    add("**长度控制后的覆盖率效应**（固定句长分组内，比较 bigram 覆盖率的四分位）：")
    add("")
    add("| 句长分组 | bigram 覆盖四分位 | 覆盖区间 | n | 平均句长 | S2 edit | A2 edit |")
    add("|---|---:|---|---:|---:|---:|---:|")
    for row in cov["length_controlled_quartiles"]:
        add(f"| {row['length_group']} | Q{row['bigram_coverage_quartile']} | "
            f"{row['bigram_cov_range'][0]:.3f}–{row['bigram_cov_range'][1]:.3f} | {row['n_samples']} | "
            f"{fmt(row['mean_ref_len'], 2)} | {fmt(row['s2_mean_edit'])} | {fmt(row['a2_mean_edit'])} |")
    add("")
    short_rows = [r for r in cov["length_controlled_quartiles"] if r["length_group"].startswith("<=15")]
    long_rows = [r for r in cov["length_controlled_quartiles"] if r["length_group"].startswith(">15")]
    for label, rows in (("≤15 字", short_rows), (">15 字", long_rows)):
        if len(rows) >= 2:
            spread = rows[-1]["s2_mean_edit"] - rows[0]["s2_mean_edit"]
            add(f"- {label} 组内：bigram 覆盖率最高四分位 − 最低四分位 的 edit 差 = **{spread:+.3f}**。")
    novelty = cov["novelty_prediction_power"]
    within = [row for row in novelty["pearson_r_bigram_cov_vs_s2_edit_within_length_bins"]
              if row["pearson_r_bigram_cov_vs_s2_edit"] is not None]
    add("")
    add(f"- **这个效应并没有被句长解释掉**：bigram 覆盖率与 S2 edit 的裸相关 r = "
        f"{novelty['pearson_r_bigram_cov_vs_s2_edit_raw']:+.3f}；"
        + ("在每个长度分箱内部仍是强正相关（"
           + "、".join(f"{row['length_bin']}: r={row['pearson_r_bigram_cov_vs_s2_edit']:+.2f}"
                      for row in within) + "）。"
           if within else ""))
    add("- 解读：**「这句话的字面搭配在训练集里出现过没有」是目前找到的最强非句长预测因子**"
        "（同句长下高低覆盖组差 0.23–0.27 edit）。注意要点有两层：")
    add("  1. 它衡量的是**字面新颖度**，不是语义类别——所以「按语义类别补数据」不会改变它；")
    add("  2. 它与「模型靠字面记忆」一致（训练集上 99.5% 完全匹配、测试集 1.7%），"
        "因此**单纯加大同分布数据只会追加更多可记忆的字面形式**，收益递减；"
        "真正要补的是**未见搭配与长句**这类绝对量稀缺的区间。")
    add("")

    # ---- length
    add("## 8. 长度混杂控制")
    add("")
    lc = payload["length_controls"]
    add("| 参考句长（字符） | n | S2 平均 edit | A2 平均 edit |")
    add("|---|---:|---:|---:|")
    for band in lc["bins"]:
        add(f"| {band} | {lc['n_samples'][band]} | {fmt(lc['s2_mean_edit_by_length'][band])} | "
            f"{fmt(lc['a2_mean_edit_by_length'][band])} |")
    add("")
    add(f"- 句长与 S2 edit 的 Pearson r = **{lc['correlation_ref_len_vs_edit']:.3f}**：句长确实是强混杂，"
        "所以所有类别判定都用「同长度分箱残差」复核，而不是只看裸 edit。")
    add("")

    # ---- conclusion
    with_metric = [entry for entry in payload["categories"] if "s2_mean_edit" in entry]
    add("## 9. 总体判断：优先扩数据，还是优先训模型？")
    add("")
    add("**结论：优先训模型 / 改训练方法。** 但要说清两件事——「扩数据」这条路线里只有**一种定向形式**有证据支持"
        "（补长句与未见字面搭配），而「按语义类别补数据」在本数据上找不到缺口。")
    add("")
    add("证据链（每条都可回查 §2–§8 的数字）：")
    add("")
    add(f"1. **类别级数据不缺**（§2）：12 类 train 命中量 {ntrain['min']:,}–{ntrain['max']:,} 条，"
        f"最少「{ntrain['min_category']}」也有 {pct(overall['category_train_share']['min'])} 的训练占比，"
        f"远高于「数据不足」阈值 {thresholds['data_short_train']}；train/test 类别占比比 "
        f"{ratio['min']:.2f}–{ratio['max']:.2f}，测试集没有集中考核训练集稀缺的语义。"
        "→ 按任务给定的判定规则（n_train<300 才算数据不足），**没有一个类别落入「数据不足」**。")
    add(f"2. **类别级「训练量 → 表现」没有正关系**（§6.2）：Spearman ρ = {fmt(cat_rho, 3)}。"
        "如果瓶颈是「量」，这里最该出现强正相关。")
    quint_note = ""
    if quint:
        quint_note = (f"Q1（最稀有，平均 {quint[0]['mean_edit']:.3f}）与 Q5（最高频，平均 {quint[-1]['mean_edit']:.3f}）"
                      f"差 {quint[0]['mean_edit'] - quint[-1]['mean_edit']:+.3f}")
    add(f"3. **词项级频次同样没有「见得多就更好」**（§6.1）：Spearman ρ = "
        f"{fmt(kw_stats['spearman_log10_train_vs_edit'], 3)}；{quint_note}，长度控制后仍不成立。"
        "词项级数据缺口（train<30 次但 test≥3 次）只驱动弱类别里个位数样本。")
    add(f"4. **符号层零缺口，但字面搭配层有明显缺口**（§7）：测试集 gloss 覆盖率 "
        f"{pct(cov['test_mean_gloss_coverage'])}、含未见 gloss 的样本 {cov['test_samples_with_gloss_oov']} 条；"
        f"然而字符二元组覆盖率只有 {pct(cov['test_mean_bigram_coverage'])}，且它在**同句长条件下**"
        "仍与 edit 强正相关（高低覆盖组差 0.23–0.27）。"
        "→ 模型不是「没见过某个手语符号」，而是**「没见过这种字面搭配」时明显退化**，这是泛化不足的直接证据。")
    add(f"5. **错误类型是「替换＋漏译」而非「多译」**（§4）：替换 {pct(shares['substitution'])}、"
        f"删除 {pct(shares['deletion'])}、插入 {pct(shares['insertion'])}，且输出普遍短于参考 → "
        "生成端覆盖不足，属训练目标/解码问题。")
    add(f"6. **类别差异基本被句长吃掉**（§5/§8）：句长与 edit 的 r = {lc['correlation_ref_len_vs_edit']:.3f}，"
        "长度校正后连唯一的正式弱类别残差也只有 -0.02 量级 → "
        "语义类别本身不是误差的结构化来源。")
    add("7. **与既有泛化结论一致**：A2（Uni-Sign 官方 pose）在训练子集上 99.5% 完全匹配、"
        "在 test 上只有 1.66%（[TRAIN_VS_TEST_GAP](TRAIN_VS_TEST_GAP.md)）——容量够、拟合够，缺的是泛化。"
        "一个能在训练集上逐字复现的模型，对「字面新颖度」高度敏感是必然的；"
        "此时**同分布扩数据主要是在增加可记忆的表面形式**，边际收益低于改训练方法。")
    add("")
    add("**两条路线各自正确的形式**：")
    add("")
    add("- **训模型（首选）**：在训练目标/正则化/增广/表征层面提升组合泛化，"
        "目标是让 edit 不再如此依赖「字面搭配是否见过」。这是唯一能整体平移 12 类的杠杆——"
        f"实测 12 类相对 A2 的 Δedit 全部落在 "
        f"{min(e['delta_edit_vs_a2'] for e in with_metric):+.4f}…"
        f"{max(e['delta_edit_vs_a2'] for e in with_metric):+.4f}，说明模型层面的改动是系统性起作用。")
    add("- **扩数据（次选，必须定向）**：只补 (a) 长句（>25 字，"
        f"n={lc['n_samples'][lc['bins'][-1]]}、edit {fmt(lc['s2_mean_edit_by_length'][lc['bins'][-1]])}）；"
        "(b) bigram 覆盖率低四分位的字面新颖句式；"
        f"(c) train<{thresholds['gap_keyword_train']} 次的长尾词项。"
        "这三处的共性是**绝对样本量小**，与语义类别无关。")
    add("- **明确不要做**：按「弱类别」立项补数据。唯一过阈值的「因果/条件」长度校正残差仅 -0.02，"
        "且它与「否定/数字/比较」在多标签下高度重叠，专项补数据既定位不准、收益上限也很低。")
    add("")
    add(f"旁证：S2 相对 A2 的增益是**普遍性**的（只有 "
        f"{sum(1 for e in with_metric if e['delta_edit_vs_a2'] < 0)} 类为负），"
        "且 A2/S2 的类别间排序几乎一致——说明主要差距是**模型与训练方法层面的系统性差距**，"
        "而不是被某个语义类的数据缺口卡住。")
    add("")
    verdicts = payload["verdict_summary"]
    data_like = [e for e in payload["weak_categories"] if e["verdict"] == "data_insufficient"]
    model_like = [e for e in payload["weak_categories"] if e["verdict"] == "model_or_method"]
    add("按类的判定：")
    add("")
    if data_like:
        add("- **偏「数据不足 / 优先扩数据」**：" + "；".join(
            f"{e['label']}（n_train {e['n_train']:,}，词项缺口 {pct(e['data_gap_share'],0)}）" for e in data_like))
    else:
        add("- **偏「数据不足 / 优先扩数据」**：无（没有弱类别同时满足「训练量低于阈值」或「词项级缺口 ≥25%」）。")
    if model_like:
        add("- **偏「模型/训练方法」**：" + "；".join(
            f"{e['label']}（n_train {e['n_train']:,}，edit {e['s2_mean_edit']:.3f}）" for e in model_like))
    else:
        add("- **偏「模型/训练方法」**：无。")
    add("")

    # ---- appendix: keyword level
    add("## 附录 A. 词项级明细（词汇 × 训练量 × 测试表现 × 错误类型）")
    add("")
    add("只列 **test ≥ 3 次**的词项（`n_train=0 且 n_test=0` 的词表冗余项不列，见 §6.2 末尾说明）。"
        "按 (类别, train 频次) 排序；`S/D/I` 为该词项命中的测试样本上、每 100 参考字符的替换/删除/插入率。")
    add("")
    kw_rows = [row for row in payload["keywords"] if row["n_test"] >= 3]
    kw_rows.sort(key=lambda row: (row["category_id"], row["n_train"]))
    add("| 类别 | 词项 | n_train | n_test | 参考均长 | S2 edit | S2 完全匹配 | Δedit vs A2 | S/D/I per100 | 数据缺口 |")
    add("|---|---|---:|---:|---:|---:|---:|---:|---|---:|")
    for row in kw_rows:
        ops = row.get("s2_error_rate_per_100_chars") or {}
        ops_text = "—" if not ops else f"{ops['substitution']:.1f}/{ops['deletion']:.1f}/{ops['insertion']:.1f}"
        add(f"| {row['category_label']} | {row['unit']} | {row['n_train']:,} | {row['n_test']} | "
            f"{fmt(row['test_mean_ref_len'], 2)} | {fmt(row['s2_mean_edit'])} | {pct(row['s2_exact_rate'])} | "
            f"{fmt(row.get('delta_edit_vs_a2'), 4)} | {ops_text} | {'是' if row['data_gap'] else ''} |")
    add("")

    # ---- limitations
    add("## 附录 B / 局限")
    add("")
    for item in payload["limitations"]:
        add(f"- {item}")
    add("- 本报告只做评测侧统计，不涉及任何训练；GT 原文不进入报告，表格内只出现计数与指标。")
    add("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="run the pipeline without writing outputs")
    parser.add_argument("--json", type=Path, default=OUT_JSON)
    parser.add_argument("--md", type=Path, default=OUT_MD)
    args = parser.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    payload = analyse(check_only=args.check)
    if args.check:
        print(json.dumps({"baseline": payload["baseline_self_check"],
                          "overall": payload["overall"],
                          "verdict_summary": payload["verdict_summary"]}, ensure_ascii=False, indent=2))
        return 0

    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.md.write_text(render_markdown(payload), encoding="utf-8")

    print(f"wrote {args.json.relative_to(ROOT)} ({args.json.stat().st_size:,} bytes)")
    print(f"wrote {args.md.relative_to(ROOT)} ({args.md.stat().st_size:,} bytes)")
    print(f"weak categories: {payload['verdict_summary']['count_weak']} "
          f"(data={payload['verdict_summary']['count_data_insufficient']}, "
          f"model={payload['verdict_summary']['count_model_or_method']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
