# P-1 冻结证据与 G2 首轮结果 · 独立交叉审计

- 审计人：独立审计子智能体（读多写一；交叉核验，非机械脚本）
- 审计时间：2026-10-27（+08:00）
- 范围：`experiments/p1/FREEZE.json`、`summary.json`、`P1_EXECUTION_REPORT.md`（E00–E06/Gate/ADR）、`experiments/p1/runs/*/manifest.json`、E02/E03/E04/E05/E06 证据、G2 首轮真实批（`g2a-wbproxy-50-v2-20261026`）与 `report_real_50.md`
- 方法：对每个检查项**实际执行命令**（`git ls-files` / `git check-ignore -v` / `gt_guard.find_gt_keys` 逐行扫描 / JSONL 行数与 status 计数 / 报告数字 vs 证据数字逐字段比对），不只看文字。
- 说明：本文件**取代**此前同路径由 `experiments/p1/verify_all.py` 自动生成的机械审计（其 E04 检查只验证「数字出现在报告」，未验证「报告数字 == 实测证据」，详见「问题 1」）。

## 结论（TL;DR）

**可引用**：P-1 冻结证据与 G2 首轮结果整体自洽，未发现阻断级问题；GT 隔离与 FREEZE 边界实测成立。**需修正 4 处**（1 重要 + 3 次要），均为「报告叙事数字与机器可读证据不一致/措辞不精确」，不改变任何实验结论。

问题计数：**阻断 0 / 重要 1 / 次要 3**。

---

## 一、核验结果总表

| # | 检查项 | 方法 | 结果 | 证据路径 |
|---|---|---|---|---|
| 1 | 9 个 run 目录存在且 manifest PASS | 遍历 `experiments/p1/runs/*/manifest.json` 取 status | **PASS**：9/9 存在，status 全部 `PASS` | `runs/*/manifest.json` |
| 2 | model_outputs.jsonl GT-free | `gt_guard.find_gt_keys` 逐行扫描全部 4 个 model_outputs.jsonl | **PASS**：4 文件 3,408 行，泄漏 0 | `runs/e01-*/model_outputs.jsonl`（50×3）、`runs/e02-offline-1086-*/model_outputs.jsonl`（3,258） |
| 3 | eval_records.jsonl 含 GT（预期，单独说明） | 读首行键名 | **符合预期**：键含 `reference/reference_official/reference_tokens/wer`，仅存在于 E01/G2 run 的 EVAL-ONLY 侧，`.gitignore` 已排除 | `runs/e01-*/eval_records.jsonl`；`.gitignore:34/39` |
| 4 | FREEZE 边界（gt_1086.csv / eval_records.jsonl / predictions-raw / artifact / Paper-algorithm 不入 git） | `git ls-files` + `git check-ignore -v` | **PASS**：8 项被禁路径均被 `.gitignore` 忽略；`git ls-files` 无任何被禁文件被跟踪 | `.gitignore:29–36,39`；`FREEZE.json:16–49` |
| 5 | E02 BLEU 抽查（24.8157 / 25.5846 / 12.2183） | 比对 summary.json ↔ translation_metrics.json ↔ 报告 §5 | **PASS**：三值一致 | `summary.json:251/271/290`、`e02/translation_metrics.json:27/105/183` |
| 6 | E03 raw ECE 0.304 / OOF AUC 0.745 | 比对 summary.json ↔ calibration_results.json ↔ 报告 §6 | **PASS**：raw_token ECE 0.3041、logistic(OOF) roc_auc 0.7449 | `e03/calibration_results.json:107–110,125–129` |
| 7 | E04 warm P50（pose/video/SLRT） | 比对 run metrics.json / metrics_summary.md / summary.json ↔ 报告 §7 / hardware_report.md | **不一致（重要）**：报告 276/2679/9775，证据 277.642/2690.395/9865.084 | 见「问题 1」 |
| 8 | E05 first_stable mean 90% / median 100% | 比对 summary.json ↔ prefix_probe_summary.md | **PASS**：mean 90.0 / median 100.0 | `summary.json:3519–3521`、`e05/prefix_probe_summary.md:8` |
| 9 | E06 A2 调用率 0.3812 | 比对 summary.json ↔ router_ablation.jsonl | **PASS**：P2_rule_cascade_A2 `second_path_call_rate=0.3812` | `summary.json:4225`、`e06/router_ablation.jsonl`（9 策略） |
| 10 | G2 manifest ↔ 报告（provider/model/base_url/prompt/decoding） | 逐字段比对 | **PASS**：provider=wb-proxy、model=deepseek-v4.1-flash、base_url=`http://127.0.0.1:8788/v1`、prompt v1_zh、temperature=0/max_tokens=2048/top_p=1.0 全部一致 | `llm_fusion/runs/g2a-wbproxy-50-v2-20261026/manifest.json:14–80` |
| 11 | G2 fusion_outputs status≠ok 行数 ↔ 报告失败数 | 统计 status 分布 | **PASS**：200 行，ok=197、error=3（D:1/E:2），与报告「失败 3」一致 | `fusion_outputs.jsonl`；`manifest.json:149`（n_errors=3） |
| 12 | G2 metrics n_ok ↔ fusion 成功行数 | 汇总各组 n_ok | **PASS**：A50+C50+D49+E48=197 == ok=197 | `metrics.json:18/75/132/189` |
| 13 | G2 是否「成功子集算指标但未披露」 | 核对报告 n 与失败披露 | **PASS**：报告 §2 与 metrics 均明确 D n=49、E n=48，§1 写明失败 3 及原因，无隐藏子集 | `report_real_50.md:19–30` |
| 14 | G2 首轮事故记录一致性 | 比对 incident_note ↔ 首轮 manifest/metrics ↔ 报告 §0 | **PASS**：首轮 max_tokens=256、73/150 空 completion（C8/D34/E31）、n_ok=77，与 §0 一致；如实披露 harness 计数缺陷 | `g2a-wbproxy-50-20261026/incident_note.md`、`report_real_50.md:9–14` |

---

## 二、发现的问题（按严重度排序）

### 阻断（0 项）

无。

### 重要（1 项）

**问题 1 · E04 热启动 P50/P95 报告数字与实测证据不符**

- 位置：`docs/experiments/P-1/P1_EXECUTION_REPORT.md` §7（第 113–115 行）；`experiments/p1/e04/hardware_report.md` §D.2（第 75–77 行）。
- 现象：两份报告把 warm 阶段 e2e 写为 **pose 276/339、video 2679/3271、SLRT 9775/11175 ms**，但其自己引用的 `metrics.json` / `metrics_summary.md` / `summary.json` 记录为 **pose 277.642/344.328、video 2690.395/3283.920、SLRT 9865.084/11189.703 ms**。

| 组 | 报告声称 P50/P95 | 实测证据 P50/P95 | ΔP50 |
|---|---|---|---|
| warm/unisign_pose | 276 / 339 | 277.642 / 344.328 | −1.64 ms |
| warm/unisign_video | 2679 / 3271 | 2690.395 / 3283.920 | −11.4 ms |
| warm/slrt_hrnet_npz | 9775 / 11175 | 9865.084 / 11189.703 | −90 ms |

- 证据路径（权威实测）：`experiments/p1/runs/e04-gpu-20261026T2202Z/metrics.json`（warm/*/segments/e2e_ms）；`experiments/p1/e04/metrics_summary.md:15–17`；`experiments/p1/summary.json:1444–1446/1513/1583`。
- 判定：冷启动行与 load 行基本正确（如 cold SLRT 9558.8 与报告 9559 一致），但 warm 三行 P50/P95 与底层数据系统性不符（SLRT 差 90 ms、video 差 11 ms）。属「measured 报告与其自身引用证据不一致」的保真度问题。**不影响 H1/H2/H3 结论与任何 ADR**，故不判阻断；建议把报告 warm 表改为与 metrics.json 一致的 4 位有效值。
- 附：此前同路径的机器审计（`verify_all.py`）对 E04 的检查仅为「pose 276/339 … 出现在 hardware_report.md → PASS」，未与 metrics.json 逐值比对，因而漏报此不一致。本次人工交叉核验即针对该盲区。

### 次要（3 项）

**问题 2 · 执行报告 §2 声称 model_outputs 共 3,458 行，实际为 3,408**

- 位置：`P1_EXECUTION_REPORT.md:39`（「各 run 的 model_outputs 共 3,458 行」）。
- 实测：4 个 `model_outputs.jsonl` 合计 3,408 行（50+50+50+3,258）；`summary.json` runs 段 `model_outputs_rows` 同证。差值 50。
- 影响：「GT 泄漏 0」结论经独立复扫成立（3,408 行 0 泄漏），此计数错误不影响结论，仅需更正数字。

**问题 3 · 执行报告 §5 的 SLRT WER 分解用历史 R18 拆分，未注明与 canonical 重算拆分不同**

- 位置：`P1_EXECUTION_REPORT.md:84`（「WER 0.3018（S1,154/D805/I522）」）。
- 实测：canonical 证据为 **S1330/D717/I434**（`e02/translation_metrics.json:261–264`、`summary.json:309–313`、`e03/calibration_results.json:102–106`、`e02/report.md:30`）。两者总编辑数均为 2,481、WER 均 0.3018，差异源于 Levenshtein 回溯并列时的选择顺序。
- 说明：`e03/report.md:12` 已显式披露这一 tie-breaking 差异，但执行报告 §5 未带注脚，读者对照 translation_metrics.json 会误以为数据不一致。WER 本身无争议。建议 §5 补充「拆分同 R18 历史口径」或直接改用 canonical 拆分。

**问题 4 · report_real_50.md §2 标题「2 次截断」与实际的 3 次失败不符**

- 位置：`experiments/p1/e02/llm_fusion/report_real_50.md:23`（「D/E 因 **2 次截断**为 49/48」）。
- 实测：失败 3 次（D:1、E:2），D 由 50→49、E 由 50→48。同报告 §1 已正确写「失败 3（D:1,E:2）」；§2 标题的「2 次」是措辞笔误。
- 影响：纯文字不一致，数字表（n=49/48）正确。

---

## 三、结论与建议

1. **GT 隔离成立**：全部 4 个 `model_outputs.jsonl` 共 3,408 行经 `gt_guard.find_gt_keys` 复扫零泄漏；`eval_records.jsonl` 的 GT 仅存在于 EVAL-ONLY 侧且已被 `.gitignore` 排除，符合 FREEZE.json 声明。
2. **FREEZE 边界成立**：`git ls-files` 未跟踪任何 `gt_1086.csv / eval_records.jsonl / predictions-raw / artifact / Paper-algorithm / inputs` 内容；`git check-ignore -v` 8/8 命中 `.gitignore`。
3. **G2 首轮真实批自洽**：manifest↔报告、fusion_outputs 失败数↔报告、metrics n_ok↔fusion 成功数、成功子集披露，四项全部一致，无「未披露子集」问题。
4. **需修正清单**（均不改结论）：① E04 warm 三行 P50/P95 与 metrics.json 对齐；② §2 model_outputs 计数 3,458→3,408；③ §5 SLRT WER 分解注明历史 R18 拆分口径；④ report_real_50.md §2「2 次」→「3 次」。

---

## 四、修正记录（2026-10-27，Lead）

本审计的 4 项发现已在提交 `13241ba` 前全部修正，并加入复用检查：

| # | 问题 | 修正 | 复验 |
|---|---|---|---|
| 1 | E04 warm P50/P95 与 metrics.json 不符 | `P1_EXECUTION_REPORT.md` §7、`hardware_report.md` §D.2、坑点 §11、项目报告统一改为 277.6/344.3、2690.4/3283.9、9865.1/11189.7 ms | `verify_all.py` 新增"E04 warm P50 与 metrics.json 对齐"检查（14/14 PASS） |
| 2 | §2 model_outputs 计数 3,458 | 改为 3,408 | `verify_all.py` 全量扫描为 3,408 行、0 GT 泄漏 |
| 3 | §5 SLRT WER 拆分口径 | 改为 canonical S1,330/D717/I434，并注明与 R18 历史拆分的 tie-breaking 差异 | 与 `translation_metrics.json` 一致 |
| 4 | report_real_50 §2 "2 次截断" | 改为"3 次失败" | 与 metrics failed_calls 一致 |

另：`verify_all.py` 的输出已从 `P1_INDEPENDENT_AUDIT.md` 改为 `P1_MECHANICAL_CHECKS.md`，避免机械检查覆盖本独立审计。
