# Round-3 预填结果（Rubric v2 · AI 预填，**待人工确认**）

> 状态：**PRELIMINARY / AI-PREFILL**，不是最终评审结论。
> 材料：30 样本 × 3 候选（A/A2/S2）；评审表 `round3_reviewer_A_ai_prefill.xlsx`（评审人列标注 `AI-PREFILL(待人工确认)`）。
> 生成：`experiments/p1/e06/round3_tools/prefill_reviewer_a.py`（prompt v1、temp=0、max_tokens=2048、90/90 成功、0 失败）。
> **B 份为空**——双人独立盲评必须由两位真人各自完成，预填只用于降低单人工作量。

## 1. v2 口径下的三候选（AI 预填，n=30）

| 候选 | 平均分 | 严重事实错误率 | 可理解部分信息 | **可支持可靠沟通** | 流畅度 |
|---|---:|---:|---:|---:|---:|
| A（视频） | 1.267 | 86.7% | 86.7% | **13.3%** | 3.63 |
| A2（stage1 pose，prod 默认） | 1.433 | 83.3% | 90.0% | **13.3%** | 3.33 |
| S2（stage2 pos 子集，dev 默认） | 1.467 | 86.7% | 90.0% | **13.3%** | 3.43 |

## 2. 配对 S2 vs A2（AI 预填）

| 指标 | 结果 |
|---|---|
| 平均分 Δ | **+0.033**（v1 同题曾为 +0.30） |
| 严重事实错误 | gained 3 / lost 2（率 86.7% vs 83.3%，S2 略差） |
| 可理解部分信息 | 都是 90.0%；gained 2 / lost 2 |
| **可支持可靠沟通** | 都是 **13.3%**；gained 2 / lost 2 → **净 0** |

## 3. 与 Round-2（v1 口径）的对照 —— 这是本轮最重要的发现

| 口径 | A2 | S2 | S2−A2 |
|---|---:|---:|---:|
| v1「可用于交流」（=能猜大意） | 86.7% | 93.3% | **+6.6pp** |
| v2「可支持可靠沟通」（=可直接用） | 13.3% | 13.3% | **0** |
| v1 平均分 | 1.600 | 1.900 | +0.30 |
| v2 平均分 | 1.433 | 1.467 | **+0.03** |

**解读**：Round-2 中 S2 的优势（+0.30 分、+6.6pp "可用率"）在 v2 严格口径下**基本归零**。
v1 的「可用于交流」测到的是"更通顺、更容易猜到大意"，而不是"事实更准"。
这与 Round-2 口径审计的结论完全吻合（58/90 行"严重错误且判可交流"）。

## 4. 另一个关键数字：可靠沟通率只有 ~13%

三个候选的 **可支持可靠沟通率都是 13.3%**（30 条里 4 条），而严重事实错误率 83–87%。
含义：**当前模型在"可直接用于交流"的标准下，覆盖范围约 1/8**；即使最好的候选，绝大多数句子仍需人工修正。
这与 P-1.5 的"字面新颖度"结论、以及训练集 99.5% vs 测试 1.7% 的泛化落差一致。

## 5. 必须保留的限定（不要越界引用）

1. **这是 AI 预填**（LLM 按 v2 量表打分），不是真人评审；真人可能系统性偏严/偏松，需人工确认后才算证据。
2. v2 的「可支持可靠沟通」判定依赖 `score≥3 且 severe=否` 的建议规则；AI 执行时较严格，可能低估该项。
3 样本 n=30、0–4 粗量表，+0.03 的分差在噪声内，不能判定"谁更好"——只能判定"**没有证据支持 S2 更好**"。
4. 数据是 `retrospective_seen_1086` 子集，非新盲测。

## 6. 对决策的影响

| 决策 | 影响 |
|---|---|
| 生产默认 A2（已选 B） | **进一步获得支持**：v2 下 S2 无优势，不应推进生产切换 |
| dev/test 默认 S2 | 可保留（便于实验对比），但需注明"v2 预填未显示质量优势" |
| 产品定位 | 可靠沟通率 ~13% ⇒ 应以"**辅助 + 人工确认**"为产品形态，而非"自动可用字幕" |
| 下一步 | **真人双评**（A 确认 + B 独立填写）；若 v2 下仍无差异，则锁定 A2 为长期默认，把资源转向训练侧（P-1.5 指向的组合泛化） |

## 7. 复现

```powershell
# 预填（.conda-env，11 分钟，90/90）
E:\githubproject\SignLanguage-Bridge\.conda-env\python.exe -X utf8 prefill_reviewer_a.py

# CSV → xlsx（C:\Python314 有 openpyxl）
C:\Python314\python.exe -X utf8 -c "import csv,sys;sys.path.insert(0,'.');from xlsx_writer_v2 import write_workbook;from pathlib import Path;rows=list(csv.DictReader(open('round3_reviewer_A_ai_prefill.csv',encoding='utf-8-sig')));print(write_workbook(Path('round3_reviewer_A_ai_prefill.csv'),Path('round3_reviewer_A_ai_prefill.xlsx'),list(rows[0].keys()),reviewer='A (AI 预填)'))"

# 统计（可只给 A，或 A+B 双评审）
C:\Python314\python.exe -X utf8 make_round3_package.py stats --reviewer-a round3_reviewer_A_ai_prefill.xlsx
```
