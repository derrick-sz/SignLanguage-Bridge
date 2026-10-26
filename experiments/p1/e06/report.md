# E06 · Rule-only 仲裁与级联 shadow 分析（P-1）

- run_id: `20261026_e06_rule_judge_cascade`，生成于 2026-10-26T13:42:47+00:00
- 数据：1086 条 CSL-Daily 本地 test 子集（`retrospective_seen_1086`，**不是新盲测**）；
  候选文本全部是 **measured imported 历史预测**（E02 `candidate_outputs.jsonl`），
  路由决策与选择是 **simulated 离线模拟**；本轮**没有**重跑任何模型。
- A = Uni-Sign RGB/video 句子；A2 = Uni-Sign released-pose 句子；B = SLRT Gloss 的确定性 join baseline。
- 状态：**PASS**（Rule-only 评测 + 级联 shadow 完成；LLM/Jev 分支 **BLOCKED**，见 §7）。

## 1. 结论摘要

- 单独 A：BLEU-4 **24.8157** / chrF **21.7974** / mean char edit sim **0.4157** / bad(edit<0.5) **0.6483**；A2：BLEU-4 25.5846 / chrF 22.4284 / edit 0.4212。A 与 A2 文本相同的样本 507/1086。
- 口径校验：本管线对 A/A2 复算的 BLEU-4（24.8157 / 25.5846）与项目坑点记忆 [R18] 的历史记录（24.8157 / 25.5846）**完全一致**，说明候选构造与参考口径正确。
- 主策略 `P2_rule_cascade_A2`（默认 A，规则触发才跑 A2）：BLEU-4 **24.9861** / chrF **21.9575** / mean edit **0.4179**，A2 调用率 **0.3812**（节省 0.6188）；error-ACCEPT 率 **0.5952**、missed-upgrade 率 **0.5568**、missed-winner 率 **0.0521**、纠错率 **0.0071**、误改率 **0.0052**。
- `P3_full_run_rule_select`（全跑+规则选择，成本=两模型都跑）：BLEU-4 24.9559 / edit 0.4174 / error-ACCEPT 0.6455 / missed-upgrade 0.0 / 纠错率 0.0071 / 误改率 0.0052。
- GT oracle（按 char edit sim 逐样本选 A/A2/B 最优，**不可部署**）：BLEU-4 **27.295** / chrF **24.2314** / mean edit **0.4805**；仅 A/A2 的 oracle：BLEU-4 **26.598**。**注意**：char edit sim 会奖励 gloss-token 覆盖，因此含 B 的 oracle 会在 299 条上选 B（不流畅的 gloss join），该上限对「句子翻译」偏乐观；A/A2 oracle 才是可参考的句子级上限。
- 加入 E03 SLRT 风险分数的变体 `P2c`：A2 调用率 0.4153、BLEU-4 25.1061、missed-upgrade 率 0.5227（分数阈值 0.5 未做 CV 选择，仅探索）。
- 关键判断：级联规则在几乎不损失质量的前提下把 A2 调用率从 100% 降到 0.3812，且 missed-winner 仅 0.0521（跳过 A2 很少真的错停优胜）；但 **收益也很小**：P2 相对 A 只 +0.1704 BLEU、纠错率仅 0.0071，而 missed-upgrade 仍 0.5568。原因：A/A2 高度相关（507/1086 文本相同），单靠显式槽位规则无法把 A2 的潜在收益兑现；要接近 A/A2 oracle 的上限（26.598 BLEU）需要更可靠的语义/置信仲裁（Uni-Sign 自身分数 BLOCKED；LLM/Jev BLOCKED；E03 的 SLRT 分数覆盖面有限）。

## 2. 候选与槽位抽取

| 槽位 | 抽取规则（确定性、纯 Python） |
|---|---|
| 数字/数量 | 阿拉伯数字 `\d+` + 中文数词（一…九/十/百/千/万/两），中文单位数词要求后接量词/单位（岁/个/年/月/日/点/…）或是「第X」结构；排除「一些/一样/一起/一直/一定/一般/一切/一共/第一/…」等非数量词 |
| 否定极性 | `不 没 未 别 无 否 勿`（`别` 排除「特别/别人/别处」） |
| 人物指代 | 我/你/您/他/她/它/我们/你们/他们/她们/咱们/大家/自己（最长匹配） |
| 疑问语气 | 吗/呢/什么/谁/哪/怎么/怎样/如何/为什么/是否/多少/几/何时/几点/多大/多久/`?`/`？` |
| 空输出 | 归一化后为空串 |

冲突示例（共记录 12 例，完整见 `judge_error_analysis.md`）：

- `S000020_P0004_T00` [numbers, person] A=`她今年50岁。` / A2=`她今年50岁。` / B=`他今天年龄4` / GT=`他 今 年 四 岁 。`
- `S000040_P0008_T00` [question] A=`今天星期五。` / A2=`今天星期五。` / B=`今天星期多少` / GT=`今 天 星 期 几 ?`
- `S000185_P0004_T00` [person] A=`我要去超市买椅子。` / A2=`我要去超市买椅子。` / B=`超市买我要买椅子你去` / GT=`我 要 去 超 市 买 椅 子 , 你 去 吗 ?`
- `S000185_P0008_T00` [person] A=`我要去超市买椅子。` / A2=`我要去超市买椅子。` / B=`超市我要买椅子你玩` / GT=`我 要 去 超 市 买 椅 子 , 你 去 吗 ?`
- `S000196_P0008_T00` [person] A=`他们想什么时候去买?` / A2=`他们想什么时候去买椅子?` / B=`椅子你们想什么时间去买` / GT=`他 们 想 什 么 时 候 去 买 椅 子 ?`

## 3. RuleJudge 状态机

`ACCEPT`、`FUSE`、`RETRY`、`UNCERTAIN` 由确定性状态机执行（本脚本），不是模型输出：

```text
A 与 B(gloss 证据) 无槽位冲突         -> ACCEPT   (默认接受 A，不跑 A2)
A 空输出                              -> RETRY    (跑 A2；A2 也空 -> UNCERTAIN -> 回退 A)
A 与 B 有冲突（数字/否定/人物/疑问）   -> RETRY    (跑 A2)
  A2 与 B 无冲突                       -> RETRY 采纳 A2
  A2 与 B 只剩数字冲突且 A 仅数字冲突  -> FUSE    (保留 A 措辞，用 B∩A2 共识数字替换 A 的数字槽)
  A2 的严重冲突数 < A 的严重冲突数     -> RETRY 采纳 A2
  其余                                 -> UNCERTAIN (回退 A，记录供 shadow/full-run 审计)
```

`P1_rule_only_no_second` 表示「只判断、无第二路」：RETRY/FUSE 降级为 UNCERTAIN，最终永远用 A；
它衡量规则本身的触发率与判断成本，不代表可部署质量。

确定性状态机已用合成算例自检（不依赖 GT）：

- `clean_accept` A=`我吃面条。` / A2=`我吃面条。` / B=`我 吃 面条` → **ACCEPT**（final=A，fuse=False）
- `negation_conflict_retry` A=`我不去吃。` / A2=`我去吃。` / B=`我 去 吃` → **RETRY**（final=A2，fuse=False）
- `numeric_conflict_fuse` A=`我吃3个苹果。` / A2=`他吃4个苹果。` / B=`我 吃 4 个 苹果` → **FUSE**（final=A_fused，fuse=True）
- `unresolved_uncertain` A=`我吃3个苹果。` / A2=`我吃3个苹果。` / B=`我 吃 4 个 苹果` → **UNCERTAIN**（final=A，fuse=False）
- `empty_primary_retry` A=`` / A2=`他去学校。` / B=`他 去 学校` → **RETRY**（final=A2，fuse=False）

注意：在本批实测数据上 FUSE 触发 **0** 次——因为 A2 与 A 有 507/1086 相同文本，A 的数字冲突通常 A2 也冲突，无法形成共识；FUSE 逻辑本身已由上面的合成算例证明可执行。

## 4. 策略消融（全部 simulated 选择 + measured imported 候选）

| 策略 | BLEU-4 | chrF | mean edit | bad率(edit<0.5) | A2 调用率 | error-ACCEPT | missed-upgrade | 纠错率 | 误改率 | missed-winner |
|---|---|---|---|---|---|---|---|---|---|---|
| P0_always_A | 24.8157 | 21.7974 | 0.4157 | 0.6483 | 0.0 | 0.6483 | 1.0 | 0.0 | 0.0 | 0.046 |
| P0_always_A2 | 25.5846 | 22.4284 | 0.4212 | 0.6354 | 1.0 | 0.6354 | 0.0 | 0.0611 | 0.0759 | None |
| P0_always_B | 12.2183 | 13.8087 | 0.3484 | 0.7919 | 0.0 | 0.7919 | 1.0 | 0.1136 | 0.6178 | 0.046 |
| P1_rule_only_no_second | 24.8157 | 21.7974 | 0.4157 | 0.6483 | 0.0 | 0.5952 | 0.5568 | 0.0 | 0.0 | 0.046 |
| P2_rule_cascade_A2 | 24.9861 | 21.9575 | 0.4179 | 0.6455 | 0.3812 | 0.5952 | 0.5568 | 0.0071 | 0.0052 | 0.0521 |
| P2c_cascade_rule_or_score | 25.1061 | 22.0563 | 0.4188 | 0.6446 | 0.4153 | 0.5874 | 0.5227 | 0.0085 | 0.0052 | 0.0504 |
| P3_full_run_rule_select | 24.9559 | 21.9326 | 0.4174 | 0.6455 | 1.0 | 0.6455 | 0.0 | 0.0071 | 0.0052 | None |
| P4b_gt_oracle_AB | 26.598 | 23.2554 | 0.443 | 0.6087 | 1.0 | None | 0.0 | 0.0611 | 0.0 | None |
| P4_gt_oracle_all3 | 27.295 | 24.2314 | 0.4805 | 0.5516 | 1.0 | None | 0.0 | 0.1491 | 0.0 | None |

严重错误代理（槽位错误率，相对 GT：数字/否定/人物/疑问/任一）：

| 策略 | 数字 | 否定 | 人物 | 疑问 | 任一 |
|---|---|---|---|---|---|
| P0_always_A | 0.1133 | 0.1335 | 0.3564 | 0.0921 | 0.5322 |
| P0_always_A2 | 0.1123 | 0.1234 | 0.3389 | 0.0976 | 0.5129 |
| P0_always_B | 0.128 | 0.0921 | 0.3057 | 0.0856 | 0.4926 |
| P1_rule_only_no_second | 0.1133 | 0.1335 | 0.3564 | 0.0921 | 0.5322 |
| P2_rule_cascade_A2 | 0.1068 | 0.1096 | 0.349 | 0.0958 | 0.5074 |
| P2c_cascade_rule_or_score | 0.1068 | 0.1096 | 0.349 | 0.0958 | 0.5074 |
| P3_full_run_rule_select | 0.1068 | 0.1096 | 0.3481 | 0.0967 | 0.5074 |
| P4b_gt_oracle_AB | 0.1068 | 0.1197 | 0.3407 | 0.0866 | 0.5101 |
| P4_gt_oracle_all3 | 0.1059 | 0.0958 | 0.3011 | 0.082 | 0.4705 |

决策分布（ACCEPT/RETRY/FUSE/UNCERTAIN）：

| 策略 | ACCEPT | RETRY | FUSE | UNCERTAIN |
|---|---|---|---|---|
| P0_always_A | 1086 | 0 | 0 | 0 |
| P0_always_A2 | 1086 | 0 | 0 | 0 |
| P0_always_B | 1086 | 0 | 0 | 0 |
| P1_rule_only_no_second | 672 | 0 | 0 | 414 |
| P2_rule_cascade_A2 | 672 | 80 | 0 | 334 |
| P2c_cascade_rule_or_score | 635 | 114 | 0 | 337 |
| P3_full_run_rule_select | 1086 | 0 | 0 | 0 |
| P4b_gt_oracle_AB | 0 | 0 | 0 | 0 |
| P4_gt_oracle_all3 | 0 | 0 | 0 | 0 |

说明：`P0_always_*` 是候选级基线（decision 全记 ACCEPT）；`P4_*` 是 GT oracle 上限，仅离线分析。`P2_*` 的 A2 调用率即级联实际会触发的第二路调用比例，节省只按**调用次数**计，**不是实测时延节省**（时延由 E04 测）。**error-ACCEPT** = decision==ACCEPT 的样本中最终文本 bad（char edit sim<0.5）的比例；**missed-upgrade** = A bad 且备选可用（margin≥0.1）却未升级的比例（仅对可跳过第二路的策略有定义）；**missed-winner** = 被跳过 A2 的样本中 A2 本可更好（margin≥0.1）的比例；**纠错率/误改率** 以 A 为基线、bad=char edit sim<0.5。

bad 阈值敏感度（最终文本 vs GT 的 char edit sim）：A 基线在 <0.3/<0.4/<0.5/<0.6 下分别 0.3711 / 0.5387 / 0.6483 / 0.7541；0.5 是本报告的「不可接受」分析阈值（代理），不是人工语义真值；表中所有 error-ACCEPT / correction / miscorrection 都依赖该阈值，换成 0.4 或 0.6 会整体平移，需按同一阈值横向比较策略。

## 5. 级联 shadow 成本-质量 tradeoff

- 主策略 `P2_rule_cascade_A2`：A2 调用率 **0.3812**，相对全跑节省 **0.6188** 次 A2 调用；
  BLEU-4 24.9861（A 基线 24.8157，全跑规则选择 24.9559，oracle 27.295）。
- missed-winner：被跳过 A2 的 672 条中，35 条 A2 本可更好（margin≥0.1，missed-winner 率 0.0521）。
- 需要升级（A bad 且备选可用）88 条，其中 49 条被 ACCEPT（missed-upgrade 率 0.5568）。
- 结论：规则级联是「低风险、低收益」——质量接近 A 基线，成本下降明显；但它不能把 A2 的潜在收益兑现，因为触发条件只覆盖显式槽位冲突。要吃到 A2 的增益，需要更可靠的置信/语义仲裁（E03 的 SLRT 分数只覆盖 gloss 路径；Uni-Sign 自身分数 BLOCKED；LLM/Jev 仲裁 BLOCKED）。

## 6. 规则错误分析（judge_error_analysis.md 摘要）

| 槽位 | 规则触发数 | 真阳性 | 误报 | 误报率 | A 槽位错误数 | 检出率 |
|---|---|---|---|---|---|---|
| numbers | 122 | 63 | 59 | 0.4836 | 123 | 0.5122 |
| negation | 125 | 85 | 40 | 0.32 | 145 | 0.5862 |
| person | 164 | 125 | 39 | 0.2378 | 387 | 0.323 |
| question | 79 | 43 | 36 | 0.4557 | 100 | 0.43 |

规则触发 = A 与 B(gloss join) 在该槽位不一致；真阳性/误报以 GT 为参照（EVAL-ONLY）。数字/人物等槽位在 A、B 都缺失时算「一致」，因此检出率分母只含 A 真实槽位错误。

## 7. 阻塞：LLM / Jev / 融合 D/E/F/G

- 本机环境**没有任何 LLM/API 端点配置**：`Get-ChildItem Env:` 的名称中没有 `*_API_KEY` / `*_TOKEN` / `OPENAI_*` / `DEEPSEEK_*` / `ANTHROPIC_*` 等变量（本会话已核对，未打印任何值）；仓库内 grep `openai|anthropic|deepseek|api_key|chat/completions` 仅命中本地 app 测试的 httpx base_url 与 vendored 上游 transformers 的模型名，**没有可用的 LLM provider 客户端或端点配置**。
- 因此 **D（SLRT Gloss→LLM）、E（Uni-Sign→LLM）、F（双模型→LLM）、G（证据约束→LLM）、LLM-only、Rule→LLM 全部 BLOCKED**，本轮不产生任何 LLM 文本或 LLM 指标。
- 要开展这些分支，需要固定：① 可访问的 LLM 端点与密钥（本机安全注入，不入库）；② 模型版本（provider/model-id/checkpoint）；③ prompt 版本与解码参数（温度、max_tokens、系统提示）；④ 预算（调用数上限/费用）与评测协议（GT 只在 EvalWorker 侧、prompt 不含 GT）。
- Jev 同样没有端点与协议实现（项目无 Jev 客户端）；`Choice/Score/Noul` 结构化输入只能作为未来设计。

## 8. 边界

1. 1086 是 `retrospective_seen_1086`：规则与策略在同一批已见数据上设计+评估，**没有独立 dev/test 划分**，结果只作可行性证据，不是泛化性能。
2. 候选文本是历史 imported 预测（video 路径非论文协议、pose 为官方 released pose），不是本轮重跑；A2 作为级联反事实来自同一历史快照。
3. A 与 A2 有 507/1086 文本完全相同 → 级联「升级」在这些样本上不产生文本变化，也解释了部分 missed-winner 为空。
4. B 是**确定性 gloss join**，不是训练好的 Gloss2Text；它的中文不流畅，BLEU/chrF 低是预期，只作为证据/冲突检测输入，不作为产品候选。另：char edit sim 会奖励 gloss token 的字符覆盖，因此 B 的 edit sim/bad 标签不能当语义质量（含 B 的 oracle 也因此偏乐观）。
5. 槽位抽取是确定性启发式代理，不是人工语义标注；规则误报/漏报已在 §6 量化。
6. 节省只按模型调用次数计，未折算时延（E04 负责）；也未计 A2 热身/加载固定成本。

## 9. 产物

| 文件 | 说明 |
|---|---|
| `experiments/p1/e06/candidate_inputs.jsonl` | 1086 条 GT-free 候选（A/A2/B + E03 分数特征），已通过 `common.write_jsonl` 的 GT guard |
| `experiments/p1/e06/router_ablation.jsonl` | 每策略一行（质量/路由/槽位/成本指标） |
| `experiments/p1/e06/judge_error_analysis.md` | 规则定义、触发/误报/检出、示例 |
| `experiments/p1/e06/policy_candidate.yaml` | 可部署策略候选（含 shadow/full-run 与 LLM 阻塞说明） |
| `experiments/p1/runs/20261026_e06_rule_judge_cascade/` | manifest、metrics、逐样本 eval_router_decisions.jsonl |

