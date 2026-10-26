# SLRT-CSL 在线 Gloss 集成交付报告

日期：2026-10-24。本报告对应[实施计划](../superpowers/plans/2026-10-24-slrt-integration.md)，依据[架构与兼容性分析](slrt-architecture-compatibility-2026-10-24.md)完成。结论先行：

> 后续核验：原 `qt-offscreen-real.json` 仅到权重加载，不能证明Qt操作通过；现以修复后的 `artifacts/slrt-input-audit-20261024/qt-offscreen-fixed.json` 为据。输入前端、裁剪与镜像评测错误的更正见[输入兼容性审计](model-input-compatibility-audit-2026-10-24.md)。本文全量测试数是历史日志，当前树的相关回归另列，不混作同一次验收。

1. **已交付**：独立 `slbridge/slrt/` 模块（资产校验、网络、预处理、姿态前端、增量解码、有界流会话、采集会话、CLI、Qt 面板），统一 `ModelManager` 注册 `slrt_online_csl`，与既有 RF/TFNet/Uni-Sign 并存，原功能未删除。
2. **已通过**：真实权重严格加载（938 键、2001 类）、对官方源码的窗口级数值对照（RGB/热力图/概率最大差均为 0）、真实 MP4 的真实 Gloss 输出、无人门控负样本、CPU 功能回退、统一路由回退与语言拒绝、全量回归 827 passed / 11 skipped。
3. **未通过**：摄像头实时词汇交付。12 秒试跑 200 帧中丢弃 56 帧，`events.jsonl` 为空——没有任何 partial/final Gloss 交付。
4. **未验证**：官方 HRNet 前端端到端（压缩包无 HRNet/检测器资源）、CSL-Daily 同域识别与真人语义评审。后续 CE-CSL 跨域 Gloss WER 见[数据集实测](slrt-dataset-eval-2026-10-24.md)，不宣称实时识别能力。

## 1. 验收总表

| 验收范围 | 结果 | 关键证据（`artifacts/slrt-integration-20261024/`） |
|---|---|---|
| 真实权重严格加载（CUDA） | 通过 | `model-load.json`：938 键 strict、2001 类、blank=0、RTX 5060 Ti、Torch 2.9.1+cu128 |
| 官方源码数值对照 | 通过 | `upstream-parity-final.json`：rgb/heatmap/probability max_abs=0，top_id 一致；对照文件哈希见 `source-provenance.json` |
| 真实 MP4 推理（rtmpose-experimental） | 通过（链路） | `isolated-mp4/`、`final-mp4/`、`acceptance-mp4/`：202 帧 202 窗、0 丢帧、输出 3 个 final Gloss |
| CSL-Daily 同域语义质量 | **未验证** | 当前为CE-CSL跨域 + 实验前端；跨域WER可计算，但不等于同域能力，见第4节 |
| 摄像头实时词汇交付 | **未通过** | `camera12s/`：56 帧丢弃、47 窗、`events.jsonl` 0 行 |
| 无人门控（黑帧负样本） | 通过 | `no-person-result.json`：60/60 帧质量拒绝、0 窗、空输出、status=final |
| CPU 功能回退 | 通过（仅功能） | `cpu-cuda-parity.json`：CPU 单窗 16.89 s vs CUDA 0.58 s；概率最大差 1.8e-4，1e-4 等价不成立 |
| 统一路由回退与语言拒绝 | 通过 | `unified-manager.json`：HRNet 缺失明确报错且保留原 RF 模型；Libras 请求被拒；真实文件经统一接口返回 Gloss |
| Qt 面板（offscreen 真实权重） | 通过（操作与释放机制） | `../slrt-input-audit-20261024/qt-offscreen-fixed.json`：status=passed，启动/暂停/恢复/重置/停止、source_release_confirmed=true、两线程退出、Legacy窗口可见；不代表实时交付通过 |
| SLRT 定向测试 | 通过 | `slrt-final-tests.log`：32 passed；`targeted.log`：67 passed、1 skipped |
| 全量回归 | 通过 | `full-pytest.log`：827 passed、11 skipped（61.3 s） |
| 独立代码审查 | 通过 | 审查推动修复 3 处边界（见第 5 节），回归用例已补 |
| 官方 HRNet 端到端 | **未验证** | 资源缺失：`EMNLP24_Online` 不含 HRNet-W48 DARK / Faster-RCNN 与 legacy MMPose 栈 |
| 真人语义评审、长时间摄像头稳定性、实时 P95 | **NOT_RUN** | 无对应产物 |

## 2. 权重与资产身份

- checkpoint `models/external/EMNLP24_Online/CSL-Daily/ckpts/cslr_best.ckpt`，SHA256 `e9fdd3a6…cbd4d`（全量见 `model-load.json`）。
- 词表 `meta/csl_iso_with_blank.vocab`，SHA256 `0c4984f0…4c76`；`<blank>`=0，2001 类，按原 JSON 顺序加载不重排。
- 加载耗时 2.87 s；空闲 GPU 峰值约 160 MB，推理期间约 453 MB。
- 上游源码来源 `https://github.com/FangyunWei/SLRT`（目录 commit `cb393bec…`），6 个网络文件哈希记录于 `source-provenance.json`；对照范围排除训练/损失与 K400 初始化、官方 HRNet 提取、历史依赖版本（`upstream-parity-final.json` 的 `excludes`）。

## 3. MP4 实测（Phase 2）

输入 `D:\Zenodo\CE-CSL\CE-CSL\video\dev\A\dev-00015.mp4`：30 FPS、202 帧、6.73 s，姿态前端 `rtmpose-experimental`，窗口 16、stride=1、投票 7。

- 输出 final Gloss：`拒绝`（score 0.254）、`去`（0.242）、`不`（0.531）；首个 final 的 `available_time` 为源时间轴 2.93 s（需右侧上下文与投票窗到齐）。
- 吞吐：端到端墙钟 26.74 s（含 2.87 s 加载）处理 202 帧 ≈ **7.6 FPS**；202 个窗口累计 `window_total` 8.20 s，单窗均值 40.6 ms（p50 30.2 ms、p95 36.6 ms、最大 1 956.8 ms 为首窗 CUDA 预热）。运行中进程峰值 RSS ≈ 2.58 GB；GPU 采样为整机口径，非 SLRT 独占。
- 首个 final Gloss 的端到端墙钟交付延迟未在产物中逐事件记录（会话口头测量约 14.9 s），**不作为验收依据**；报告只引用 `available_time` 口径。
- 三次独立运行（`mp4-default`、`final-mp4`、`isolated-mp4`）输出一致，分数差异在小数点后 4–5 位，属正常数值波动。
- stride=4 实验配置（`mp4-experimental-stride4.json`）：51 窗、0 Gloss——投票对应的时间跨度改变后输出退化，结果按文档标注实验配置，不作为可用配置推荐。

## 4. 为什么不声明准确率

该视频属于 CE-CSL dev 集，而权重为 CSL-Daily 训练；当前前端又是显式选择的 `rtmpose-experimental`（检测器/估计器与官方 HRNet-W48 DARK 不同，输出中保留 `pose.experimental=true`）。输出与参考句仅“不”一词重合，跨域 + 前端差异叠加，无法单独归因。CE-CSL 有真实 Gloss 标注，后续正常方向500条的历史WER为0.8918；初版没有进行成套标注评测的范围不能被写成数据集没有标注。`scripts/benchmark_slrt.py --reference` 仅接受真实 Gloss 真值，结果不改称通用准确率。

## 5. 摄像头实测（Phase 3，未通过）

`camera12s/`：12 秒采集（含关闭共 16.26 s），接收 200 帧、处理 47 窗、**丢弃 56 帧**、结果 `status=partial`、`gloss_sequence=[]`，`events.jsonl` 为空——**没有交付任何 partial 或 final Gloss**。

机制：采集快于推理（摄像头窗均 73.4 ms vs 视频链路稳态约 30–40 ms，叠加进程内竞争），32 帧有界队列溢出丢旧帧，窗口右上下文与 7 窗投票难以稳定到齐。这是**吞吐不足**问题，不是崩溃：进程正常退出（exit 0）、无人体时门控正确、暂停/重置/释放行为有测试覆盖。按任务要求，“退出码 0、能持续采集”不计为实时识别通过。

独立审查期间修复的三处边界（回归用例已补）：快速 pause/resume 覆盖 reset 命令；native read 未退出时界面提前报告摄像头已释放；投票结果少记录左侧证据窗口范围。

## 6. 修改范围与产物

- 新增：`slbridge/slrt/`（14 个文件）、`tests/test_slrt_protocol.py`、`tests/test_slrt_stream.py`、`tests/test_slrt_capture.py`、`tests/test_slrt_resources_video.py`、`scripts/verify_slrt_parity.py`、`scripts/benchmark_slrt.py`、`configs/slrt-csl.example.json`、`requirements-slrt.txt`、`docs/slrt-integration.md`、`docs/reports/slrt-architecture-compatibility-2026-10-24.md`。
- 有限接线（改动前副本与 SHA256 存于 `before/`、`before-sha256.json`）：`.gitignore`、`README.md`、`pyproject.toml`（新增 `slbridge-slrt` 入口）、`slbridge/contracts.py`、`slbridge/desktop_cli.py`（`--slrt` 入口）、`slbridge/model_adapters.py`（注册 `slrt_online_csl`）。
- 工作区尚有本任务之前已存在的未提交改动（Uni-Sign、桌面端、评测等），本任务未覆盖、未提交；权重、视频、姿态缓存与 `artifacts/` 均不入库。
- 运行产物索引：`artifacts/slrt-integration-20261024/`（本报告所有数字的来源）。

## 7. 后续顺序建议

1. 摄像头实时交付是当前主要缺口：优先降低单窗成本（窗口批处理、姿态前端提速）或以 stride/窗口参数在真实回放下重新标定，目标是采集不丢帧且首个 Gloss 延迟有界；改进后用 `replay` + 真人摄像头各测一轮。
2. 补齐官方 HRNet 资源（Faster-RCNN + HRNet-W48 DARK + legacy MMPose 栈）后重跑数值对照与 MP4，将 `rtmpose-experimental` 结果与之并排，量化前端差异。
3. 语义质量需在 CSL-Daily 域内视频（与权重同域）建立 WER 基线，再谈 CE-CSL 跨域；Gloss-to-Text 消费带状态/版本的 Gloss 事件，另行验收。
