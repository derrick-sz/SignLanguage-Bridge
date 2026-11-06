# 指定拟合片段时序修复候选（2026-11-06）

用户指定“我去开会汇报抽样调查落实的情况”。已针对该片段制作可运行、可下载的显示修复候选，保留原始对照；当前入口为 [8875 工作台](http://127.0.0.1:8875/animation/)，选择“本轮片段对照”中的该句，确认预览。页面收尾停在 4.12 s 的同步对照。

## 身份与范围

- asset_id：`new-0ad6eaefc7647487-cap-0067`，300 帧、25 FPS、12 秒。
- 动作资源 SHA256：`91a8f852780699f4614279c7680c2c236aa4769de03f30ed5e2133b55e5b6c80`。
- 原记录的视频源 SHA256：`0ad6eaefc764748726222bfe0caa950b9b7e3d5f9d240459191777f4c793337b`，与上述动作资源摘要是不同层级。
- 原 ZIP SHA256 仍为 `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`；未修改源数据。
- 人物 SHA256：`b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c`，与上一轮相同。
- 包处理版本：`source-wrist-ik-temporal-candidate-v2`；仅该片段声明 `temporal_profile=source-quaternion-temporal-v1`。
- 工作区：`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，分支 `codex/csl-animation-workspace`；保留已有未提交修改，无提交、推送或识别模型变更。

[实施计划](../superpowers/plans/2026-11-06-targeted-temporal-repair.md)；前序[全库诊断与语料扩充核查](animation-jitter-and-corpus-expansion-2026-11-06.md)。

## 原因与处理

该段源姿态有较大的帧间变化。旧显示处理对孤立尖峰最多修正约 3°，不足以处理大幅变化；位置 IK 在平滑旋转后仍读取原始源姿态，又把原始噪声带回目标轨。对本段试验，直接增大下游旋转平滑还会放大手腕局部转角，因此没有采用该处理顺序。

新增纯数值的源空间四元数居中处理：身体核标准差 45 ms、手指 30 ms，每个源关节最大旋转修正 30°，首末帧保持原值，时间戳和帧数不变。先处理源姿态，再由同一条处理后姿态计算重定向及 FK/IK 目标，避免两条轨相互冲突。使用最短弧插值，处理轴角跨 ±π 及等价四元数表示；不覆盖原姿态。

新配置仅由制作包对指定动作启用，其他动作仍用原配置。制作包校验已将时序配置加入 GLB 内部身份绑定；未知配置被拒绝，不能通过改清单标签将旧导出冒充新修复。检查 JSON 同时记录处理版本、时序配置、当前原始/改进显示和对照状态。

这属于有界的显示候选修复。原归档不含该视频；本轮本机已查范围未定位对应 RGB/置信度，未重跑 SMPL-X 拟合，也没有从其他视频替代该资产的来源。

## 同一角色、同一片段的数值比较

对 300 帧分别编译旧处理版与候选版，以单位四元数最短弧计算相邻帧转角。下表是整个片段的各组最大值，峰值可能出现在不同帧；不是手语准确率或对真人真值的误差。

| 指标 | 旧处理版 | 候选版 |
|---|---:|---:|
| 手臂最大相邻帧转角 | 37.99264° | 10.01446° |
| 手腕局部最大相邻帧转角 | 24.71779° | 6.91105° |
| 手指最大相邻帧转角 | 58.07436° | 24.77832° |
| 右手腕位置 RMS 加速度（模型坐标单位/s²） | 80.47750 | 11.82663 |
| 左手腕位置 RMS 加速度（模型坐标单位/s²） | 23.63427 | 3.39452 |

手指相对原始重定向的最大修正为 24.76948°，骨长变化为 0，不可达目标为 0。手腕派生目标平均偏差约 `5.96e-8` 模型坐标单位；这些目标由源姿态推导，并非真人真值。

仍有 17 个手指孤立回跳候选（固定诊断：当前帧相对前后中间姿态偏差 >10°，前后相差 <15°）。这是排查线索而非已知错误；没有为了归零而继续抹平。真实快速手形、接触和含义仍需原视频/人工复核。

## 实际验证

| 检查 | 结果 | 范围 |
|---|---|---|
| Node 动画测试 | 26 passed | 新源空间处理、恒速/保持/端点/输入不变、修正幅度及逐段配置选择等回归；新用例先失败再通过 |
| Python 全量 | 1084 passed / 25 skipped / 15 warnings，67.55 s | 真实全量运行；warnings 为既有依赖弃用提示，跳过不计通过 |
| 制作包专项 | 14 passed | 新配置校验、GLB 内部绑定及旧路径/来源/审核边界 |
| 导出核验器回归 | 3 passed | 从制作清单核对完整动作 ID 集合，拒绝缺失及未列入片段；修复原来的固定五段假设 |
| Blender | 真实输出 1 个既有派生人物和 6 段 GLB | 新候选包独立保存，旧包保留 |
| 导出一致性 | 6 段、1138/1138 帧通过；最大关节位置差 `2.6052075537554223e-06` | GLB 与对应运行轨；不构成语言批准 |
| 旧五段回归 | 五段 worldFrames 与旧包逐项完全相等 | 新时序处理没有推广到其余片段 |
| 浏览器 | 指定片段完整播放至 0:12/0:12；4.08、4.12、4.16 s 画面对照；定位 3.68、6.36、8.40、8.80 s | 4.12 s 原版的短促抬臂在候选中被抑制；未测稳定 FPS/P95 |
| 下载 | 实际下载 JSON/GLB，并核对动作、配置、包版本和文件摘要 | JSON 在 4.12 s，display_profile=improved、comparison=true、language_review=NOT_RUN |
| 浏览器日志 | 本轮检查无 error/warn | 不覆盖全部设备环境 |
| 语言/手形验收 | NOT_RUN | 未继承旧动作批准，界面明确“时序修复候选” |

![原始与时序修复候选在 4.12 秒的对照](../../artifacts/animation-upgrade/temporal-repair/comparison-4.12.png)

截图左侧为原人物/未处理重定向，右侧为修复候选。上表数值比较使用同一个制作角色，基线为上一轮改进处理；这两个对照范围分别记录。

## 产物与复现

新包位于 `artifacts/animation-upgrade/temporal-repair/`，旧包保留在 `artifacts/animation-upgrade/presentation/`，二进制和日志均在忽略目录。

- `manifest.json`、`avatar-clean.glb`、六个动作 GLB、六份 tracks.json、geometry-report.json、export-verification.json。
- build.log、node-tests.log、python-full.log、export-verification.log。
- browser-target.json、comparison-4.08.png、comparison-4.12.png、comparison-4.16.png。
- downloaded-review.json、downloaded-animation.glb 为浏览器真实下载的核验副本。
- `../temporal-audit/target-comparison.mjs` 与 target-comparison.json 保存同角色旧版/候选版的运动诊断；该目录另有原输入和修复前截图。

启动/制作命令见[工作台说明](../csl-animation-workspace.md)。本轮未训练模型、扩充新语言语料、完成任意文本生成或通用接触/碰撞验收。后续应先回溯这段原片复核剩余手形，再决定是否扩展新配置到其他片段。
