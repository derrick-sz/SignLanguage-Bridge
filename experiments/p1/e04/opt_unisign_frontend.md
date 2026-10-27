# 阶段 4 · Uni-Sign 前端开关在 CSL-Daily test 的配对对照（opt_unisign_frontend）

> **状态**：`PASS`（两轮子集实验均 0 失败）；**生产默认值未修改**（只出证据与建议）
> **run 目录**（均在 `experiments/p1/e04/opt-runs/`）：
> - `unisign-frontend-pilot50-20261027T1126Z`：pilot-50（10 signer × 5，signer 分层）
> - `unisign-frontend-long12-20261027T1129Z`：CSL-Daily test 中 >256 帧的 12 条（7 signer，
>   294.9 帧均值）——用于评估 `pose_sample_only`（pilot 50 全部 ≤172 帧，无帧可跳）
> **模型/协议**：Pinned Uni-Sign `csl_daily_pose_only_slt.pth`（SHA256 `73bea87a…1a6c`），
> bf16、beam=4、max_new_tokens=100、max_frames=256、isotropic_short_side、preserve；
> 输入为 `trimmed/<sample>.mp4`（本地 RTMPose 前端）；参考文本统一用上游
> `test_tmp_refs.txt`（character-separated，BLEU 用上游 mt5-fix 协议 + sacrebleu 13a）。
> **只改两个开关**：`pose_sample_only ∈ {False, True}` × `pose_detect_stride ∈ {1, 5}`。

## 1. 实验设计

| arm | pose_sample_only | pose_detect_stride | 说明 |
|---|---|---|---|
| baseline | False | 1 | 当前生产默认 |
| sample_only | True | 1 | 只对 >max_frames 的片段生效（pilot 50 无可跳帧） |
| detect_stride5 | False | 5 | 跨帧复用检测 bbox（此前只在 CSL-News/CE-CSL 验收） |
| sample_only_stride5 | True | 5 | 交互项 |

质量门槛（任务给定）：Δedit ≥ −0.005 **且** paired signer bootstrap 95% CI 下界 > −0.01。
Bootstrap：按 signer 有放回重采样 10 个 signer（1000 次），每次对同一批样本重算
corpus BLEU / 平均 edit / exact match，Δ = arm − baseline。signer 仅 10 人，CI 是
诊断性而非总体推断。

## 2. pilot-50 结果（10 signer × 5，50 条全部成功）

| 指标 | baseline | sample_only | detect_stride5 | sample_only_stride5 |
|---|---|---|---|---|
| 成功/失败 | 50/0 | 50/0 | 50/0 | 50/0 |
| BLEU-4（13a, mt5-fix） | **31.2308** | 31.2308 | 29.5402 | 29.5402 |
| chrF | **27.1849** | 27.1849 | 25.8317 | 25.8317 |
| 平均字符 edit 相似度 | **0.4831** | 0.4831 | 0.4626 | 0.4626 |
| exact match | 2 | 2 | 3 | 3 |
| pose_ms P50 / P95 | 2752.6 / 4091.2 | 2715.2 / 3708.3 | 1663.8 / 2352.5 | 1636.2 / 2300.5 |
| model_total_ms P50 | 316.6 | 280.6 | 291.4 | 292.0 |
| **e2e wall P50 / P95 ms** | **3086.2 / 4481.9** | 3029.2 / 4007.4 | **1964.1 / 2642.3** | 1957.6 / 2618.3 |
| e2e P50 变化 | — | **−57.0 ms（−1.8%，噪声级）** | **−1122.1 ms（−36.4%）** | −1128.6 ms（−36.6%） |
| 提取器调用均值 | 115.34 | 115.34 | 115.34 | 115.34 |
| 跳过帧均值 | 0 | **0** | 0 | **0** |

### 配对 bootstrap（vs baseline，1000 次 signer 重采样）

| arm | Δedit（点估计） | Δedit 95% CI | ΔBLEU-4（CI） | 预测逐字相同 | 门槛 |
|---|---|---|---|---|---|
| sample_only | 0.0 | [0.0, 0.0] | 0.0 [0.0, 0.0] | **50/50** | **PASS**（但本子集 0 跳帧，等于 no-op） |
| detect_stride5 | **−0.0205** | **[−0.0563, +0.0153]** | **−1.6906** [−4.9443, +1.1913] | 32/50 | **FAIL**（点估计低于 −0.005，CI 下界也低于 −0.01） |
| sample_only_stride5 | −0.0205 | [−0.0563, +0.0153] | −1.6906 [−4.9443, +1.1913] | 32/50 | **FAIL**（与 stride5 相同，交互项无额外影响） |

## 3. 长片段子集（`pose_sample_only` 真正生效的场景，12 条，294.9 帧均值）

CSL-Daily test 1086 条中 **只有 12 条 >256 帧（1.1%，最大 340 帧，均值 125.1）**；
pilot-50 全部 ≤172 帧，所以 `pose_sample_only` 在 pilot 上无跳帧、无收益。

| 指标 | baseline（stride1） | sample_only |
|---|---|---|
| 成功/失败 | 12/0 | 12/0 |
| 源帧均值 / 提取器调用均值 | 294.92 / 294.92 | 294.92 / **256.00** |
| **跳过帧均值** | 0 | **38.92（13.2%）** |
| pose_ms P50 / P95 | 6516.7 / 7742.3 | **5815.7 / 6401.9** |
| e2e wall P50 / P95 ms | 7013.1 / 8277.4 | **6271.5 / 6898.4（−741.6 ms, −10.6%）** |
| BLEU-4 / chrF / edit | 15.6037 / 15.2062 / 0.2856 | **完全相同** |
| 预测逐字相同 | — | **12/12** |
| Δedit 95% CI | — | [0.0, 0.0] → **PASS** |

> 长片段 baseline BLEU 明显低于 pilot-50（15.6 vs 31.2），因为该子集是不同句子且更长；
> 这是配对对照，同一批样本的两臂比较才有效。

## 4. 结论与建议

1. **`pose_sample_only`：质量零变化，可安全开启，但 CSL-Daily 域收益很小。**
   - 设计上它只跳过 `prepare_pose_window` 最终不会用到的帧：pilot-50 50/50 预测逐字
     相同（0 跳帧，应为 no-op）；>256 帧子集 12/12 逐字相同、质量指标完全一致、
     每卡 10.6% e2e 改善（跳 13.2% 帧）。
   - CSL-Daily test 仅 1.1% 片段 >256 帧 → 域级平均收益 ~0.1%，**不要宣称“CSL-Daily
     端到端提速 10%”**；它对长片段/其他域（CE-CSL、CSL-News 中 >256 帧比例更高）和
     防浪费有价值。**建议**：可在模型包 runtime 中显式开启（需按现有协议把开关写入
     姿态 protocol），或保持关闭但在长片段路径按需启用——两种都不改变识别结果。
2. **`pose_detect_stride=5`：不通过质量门槛，不建议在 CSL-Daily 上默认开启。**
   - 延迟收益明确（e2e P50 −36.4%，pose −40%），但 Δedit −0.0205、CI 下界 −0.0563，
     BLEU −1.69，仅 32/50 预测相同 → 按任务门槛 FAIL。此前它只在 CSL-News/CE-CSL
     做过精度验收，本次证明**不能直接外推到 CSL-Daily 默认值**。
   - 若产品需要在 GPU 紧张时应急提速，可保留为显式实验开关，并接受质量风险。
3. **生产默认值不变**：bundle manifest 未修改；两个开关保持 `pose_sample_only=false`、
   `pose_detect_stride=1`。若后续决定开启 sample_only，需同步维护姿态缓存身份/protocol
   （见 `docs/uni-sign-integration.md` 的 protocol 规则）。
4. 局限：pilot-50 是诊断子集（10 signer，非独立测试集）；长片段只有 12 条且仅 7 个
   signer；bootstrap 的 CI 宽度反映 signer 数量限制。延迟在单进程、有桌面/agent 后台
   负载的机器上测得，不是干净机房 SLA。

## 5. 复现命令

```powershell
$env:PYTHONIOENCODING='utf-8'
# pilot-50 四臂
.\.conda-env\python.exe experiments\p1\e04\opt_unisign_frontend.py run `
  --run-dir experiments\p1\e04\opt-runs\unisign-frontend-pilot50-<stamp> `
  --samples 50 --sample-set pilot `
  --arms baseline,sample_only,detect_stride5,sample_only_stride5 --bootstrap-iterations 1000
# 长片段两臂
.\.conda-env\python.exe experiments\p1\e04\opt_unisign_frontend.py run `
  --run-dir experiments\p1\e04\opt-runs\unisign-frontend-long12-<stamp> `
  --samples 20 --sample-set long --min-frames 256 --arms baseline,sample_only
```

产物：run 目录内 `manifest.json` / `metrics.json` / `predictions-<arm>.jsonl`（逐样本
预测+质量+延迟+跳帧）。GPU 基线 2201–2595 MiB 桌面占用、无其它 CUDA compute 进程；
单进程串行；本实验 0 次 OOM/失败，未触发重试。
