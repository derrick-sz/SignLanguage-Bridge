"""Round-3 reviewer-A AI 预填（供人工确认；不替代独立双人盲评）。

用途：把 90 行按 Rubric v2 预填一份，人工只做确认/修正，从而降低单侧评审成本。
**B 份不接受预填**——双人独立盲评必须由两位真人各自独立完成，否则会人为制造"一致性"。

纪律：
- 参考译文仅用于评审判断（与人类评审看到的材料完全相同），**不用于训练/调 prompt**；
- 密钥只在内存中解析，不打印、不落盘；输出 manifest 只记 `credentials_ref`；
- 每行调用一次、temperature=0、固定 prompt 版本，便于审计；
- 输出标记 `评审人="AI-PREFILL(待人工确认)"`。

用法：
    python prefill_reviewer_a.py [--limit N] [--out prefix]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE, *HERE.parents) if (p / "experiments" / "p1").is_dir())

sys.path.insert(0, str(REPO))
from experiments.p1.e02.llm_fusion.backends.openai_compat import _read_api_key  # noqa: E402

ENDPOINT = os.environ.get("P1_OPENAI_BASE_URL", "http://127.0.0.1:8788/v1").rstrip("/")
MODEL = os.environ.get("P1_OPENAI_MODEL", "deepseek-v4.1-flash")
PROMPT_VERSION = "round3-rubric-v2-prefill-v1"

SYSTEM = (
    "你是手语翻译译文的人工评审助手。你会看到一句【参考译文】和一句【候选译文】，"
    "请严格按给定量表打分，只输出一个 JSON 对象，不要解释、不要 markdown。"
)
USER_TMPL = """【参考译文】{ref}
【候选译文】{cand}

量表：
- score（0-4）：0 完全错误/相反；1 主要错误；2 部分正确但关键信息缺漏；3 核心意思正确、局部偏差；4 完整忠实自然。
- slots（人物/否定/数字/时间/动作/对象）：候选与参考在这些信息上是否一致，取值 "是"/"否"/"无法判断"。
- severe（严重语义错误，"是"/"否"）：六槽位中任意一项被改变且会让读者得到与参考不同的信息 → "是"（与 score 独立，score=3 也可能是"是"）。
- understandable（可理解部分信息，"是"/"否"/"无法判断"）：不看参考时能否大致猜到候选的意思。
- reliable（可支持可靠沟通，"是"/"否"/"无法判断"）：候选含义是否准确到可直接用于交流（建议仅当 score>=3 且 severe="否" 时为"是"）。
- fluency（1-5）：候选本身的中文流畅度。

只输出如下 JSON（键必须齐全，取值必须来自上面允许的集合）：
{{"score": 0-4, "slots": {{"人物": "...", "否定": "...", "数字": "...", "时间": "...", "动作": "...", "对象": "..."}}, "severe": "...", "understandable": "...", "reliable": "...", "fluency": 1-5}}"""

SLOT_KEYS = ["人物", "否定", "数字", "时间", "动作", "对象"]
ALLOWED3 = {"是", "否", "无法判断"}
COL_MAP = {"人物": "人物_是否一致", "否定": "否定_是否一致", "数字": "数字_是否一致",
           "时间": "时间_是否一致", "动作": "动作_是否一致", "对象": "对象_是否一致"}


def _post(payload: dict, key: str) -> dict:
    import urllib.request

    request = urllib.request.Request(
        f"{ENDPOINT}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _parse(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def _validate(obj: dict | None) -> dict | None:
    if not isinstance(obj, dict):
        return None
    try:
        score = int(obj["score"])
        fluency = int(obj["fluency"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= score <= 4 and 1 <= fluency <= 5):
        return None
    slots = obj.get("slots") or {}
    if not isinstance(slots, dict) or any(slots.get(k) not in ALLOWED3 for k in SLOT_KEYS):
        return None
    if obj.get("severe") not in {"是", "否"}:
        return None
    if obj.get("understandable") not in ALLOWED3 or obj.get("reliable") not in ALLOWED3:
        return None
    return {"score": score, "fluency": fluency,
            "slots": {k: slots[k] for k in SLOT_KEYS},
            "severe": obj["severe"], "understandable": obj["understandable"],
            "reliable": obj["reliable"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Round-3 reviewer-A AI prefill")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--template", default="round3_review_template.csv")
    parser.add_argument("--out", default="round3_reviewer_A_ai_prefill")
    args = parser.parse_args(argv)

    template = HERE / args.template
    if not template.is_file():
        raise SystemExit(f"missing template: {template}")
    key, source = _read_api_key(None, "WB_PROXY_API_KEY")
    if not key:
        raise SystemExit(f"no API key ({source})")

    with open(template, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if args.limit:
        rows = rows[: args.limit]

    prompt_sha = hashlib.sha256((SYSTEM + USER_TMPL).encode("utf-8")).hexdigest()
    out_rows, failures, tokens = [], [], {"prompt": 0, "completion": 0}
    started = time.time()
    for index, row in enumerate(rows, 1):
        ref = row["参考译文_EVAL_ONLY_仅评审用"]
        cand = row["候选译文"]
        payload = {
            "model": MODEL, "temperature": 0, "max_tokens": 2048,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": USER_TMPL.format(ref=ref, cand=cand)}],
        }
        parsed = None
        for attempt in range(2):
            try:
                data = _post(payload, key)
                usage = data.get("usage") or {}
                tokens["prompt"] += int(usage.get("prompt_tokens") or 0)
                tokens["completion"] += int(usage.get("completion_tokens") or 0)
                choice = (data.get("choices") or [{}])[0]
                parsed = _validate(_parse((choice.get("message") or {}).get("content") or ""))
            except Exception as error:  # noqa: BLE001 - record and retry once
                if attempt == 1:
                    failures.append({"review_id": row["review_id"], "error": f"{type(error).__name__}: {error}"[:200]})
            if parsed:
                break
        if not parsed:
            failures.append({"review_id": row["review_id"], "error": "unparseable-or-invalid"})
            out = dict(row)
            out["备注"] = "AI 预填失败：需人工填写"
        else:
            out = dict(row)
            out["评分0-4"] = str(parsed["score"])
            for slot, col in COL_MAP.items():
                out[col] = parsed["slots"][slot]
            out["严重语义错误"] = parsed["severe"]
            out["可理解部分信息"] = parsed["understandable"]
            out["可支持可靠沟通"] = parsed["reliable"]
            out["流畅度1-5"] = str(parsed["fluency"])
            out["评审人"] = "AI-PREFILL(待人工确认)"
            out["评审日期"] = time.strftime("%Y-%m-%d")
            out["备注"] = "AI 预填：请人工逐行确认/修正"
        out_rows.append(out)
        if index % 10 == 0 or index == len(rows):
            print(json.dumps({"done": index, "total": len(rows), "failures": len(failures)}), flush=True)

    csv_out = HERE / f"{args.out}.csv"
    with open(csv_out, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    xlsx_note = None
    try:
        from xlsx_writer_v2 import write_workbook
        write_workbook(csv_out, HERE / f"{args.out}.xlsx", list(rows[0].keys()), reviewer="A (AI 预填)")
        xlsx_note = f"{args.out}.xlsx"
    except Exception as error:  # noqa: BLE001
        xlsx_note = f"xlsx skipped: {type(error).__name__}: {error}"

    manifest = {
        "purpose": "Round-3 reviewer-A AI prefill (accelerator only; reviewer B must stay human and independent)",
        "prompt_version": PROMPT_VERSION, "prompt_sha256": prompt_sha,
        "endpoint": ENDPOINT, "model": MODEL, "temperature": 0, "max_tokens": 2048,
        "credentials_ref": "WB_PROXY_API_KEY@dsh-credentials", "credentials_source": source,
        "rows": len(out_rows), "failures": failures,
        "tokens": tokens, "elapsed_seconds": round(time.time() - started, 1),
        "outputs": {"csv": csv_out.name, "xlsx": xlsx_note},
        "gt_discipline": "reference text used only as review material (same as a human reviewer sees); "
                         "never used to train or tune the model; prefill is labelled and must be human-confirmed",
    }
    (HERE / f"{args.out}_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "failures", "tokens", "elapsed_seconds", "outputs")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())