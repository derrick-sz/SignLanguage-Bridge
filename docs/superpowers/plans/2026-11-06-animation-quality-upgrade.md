# Animation Quality Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** 完成五段 CSL 动画的角色、几何约束、面部控制和同步对照改进，并交付实际浏览器证据。

**Architecture:** 保留原规划与资源库，以纯数值 FK/双节 IK 加上 Three.js 骨架适配生成派生轨。人物和制作包与原动作身份分开，Blender 离线输出接入受控资源入口；单一播放时钟驱动原版、改进版与源骨架。

**Tech Stack:** Python/FastAPI、原归档固定 Three.js、原生 ES modules、Node test、pytest、Blender CLI、可见浏览器。

**Spec:** `docs/superpowers/specs/2026-11-06-animation-quality-upgrade-design.md`

## Global Constraints

- 复用 `C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，基线 `86e0d45cd371d2db6db2a0c522021c37277d9828`，保留已有未提交文件。
- 用户已要求计划后执行，计划写完后直接执行；不另行等待确认。不提交、推送、合并或改识别模型。
- 原 ZIP SHA256 `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`，只读。
- 原始轨、动作审核、语言审核、目标角色和派生轨分别记录；人工语言验收 NOT_RUN。
- 日志/大二进制放 `artifacts/animation-upgrade/`，长期源码/文档入工作区但保持未提交。

## Review Focus

- 停止后迟到的动作、人物或制作包响应不能重新启动或恢复旧轨。
- 对照、面部开关、调速和近景切换不改变同一时间点；窄屏不能裁掉扩展手形。
- 源骨骼比例不同与不可达 IK 目标不能造成非有限数、骨长变化或手腕/手指形状漂移。
- 缺少面部数据、未知人物配置与不匹配动作摘要不能静默采用其他片段或语言标记。
- 制作包路径、摘要与导出文件的校验不能开放任意本地文件或继承旧审核。

---

### Task 1: 源 FK、人物配置与双节位置约束

**Files:** Create `slbridge/animation/web/motion-geometry.mjs`, `tests/animation_geometry.test.mjs`; modify `rig.js`, `retarget.js`.

**Interfaces:** `forwardKinematics(pose, parents, restJoints) -> {positions, rotations}` uses axis-angle radians; `solveTwoBone(root, elbow, wrist, target) -> {elbow,wrist,clamped}` uses 3-vectors. `createT08Retarget(...).compile(clip,{positionConstraints:false|true})` preserves current default and source. Rig exposes canonical bones and rest positions for independent face offsets.

- [x] 写并运行 FK 旋转、非法拓扑、零长度/不可达 IK 测试，观察失败。
- [x] 实现有限值验证、FK、保持骨长的两节 IK，再实现 Three.js 的手腕目标传递；固定手腕世界朝向及手指局部旋转。
- [x] 用测试验证长度、目标误差、确定性与输入不变；运行现有前端回归。

Run: `node --test tests/animation_geometry.test.mjs tests/animation_player.test.mjs tests/animation_stage.test.mjs tests/animation_workspace.test.mjs`
Expected: all pass; original input preserved.

### Task 2: 独立面部与角色配置

**Files:** Create `slbridge/animation/web/expression.mjs`, `tests/animation_expression.test.mjs`; modify `rig.js`.

**Interfaces:** `sampleExpression(track,time,duration) -> channels`; `applyExpression(rig,channels)` maps validated bounded eyebrow/eyelid/jaw offsets from captured rest. Missing tracks return neutral; independent seek is deterministic. `validateRigProfile(profile)` rejects unknown/malformed binding/calibration.

- [x] 先运行缺轨、中间插值、越界数值及逆向定位测试的失败版本。
- [x] 实现独立时间采样、有限幅度和角色绑定；不从输入文字推断表情。
- [x] 跑新测试及几何回归。

Run: `node --test tests/animation_expression.test.mjs tests/animation_geometry.test.mjs`
Expected: all pass; neutral reset and bounds proven.

### Task 3: 离线制作与受控制作包

**Files:** Create `scripts/animation/build_presentation.py`, `scripts/animation/export_tracks.mjs`, `slbridge/animation/presentation.py`, `tests/test_animation_presentation.py`; modify `cli.py`, `api.py` and static allowlist as needed.

**Interfaces:** `PresentationBundle(path)` validates `manifest.json` schema 1, basename-owned files with SHA256, avatar identity and source-bound pilot entries. `catalog()` emits no disk paths; `resource(name)` exposes whitelisted files. CLI `--presentation DIR` is optional; default registry/API behavior is preserved.

- [x] 写并观察缺包、坏摘要、路径穿越、错误动作绑定和有效包 API 测试失败。
- [x] 取得官方便携 Blender 并校验发布摘要（仅任务产物目录）；制作原角色的可读性修正版，保留原模型，生成来源/修改清单。
- [x] 实现构建、离线轨导出及只读资源接入，创建五段试点清单，运行专项测试。

Run: Python animation registry/API/presentation tests; Blender background build and exported GLB reinspection.
Expected: valid bundle serves intended assets; corrupt/unsafe files rejected; original archive hash unchanged.

### Task 4: 舞台与工作台同步对照

**Files:** Modify `stage.js`, `workspace.js`, `index.html`, `workspace.css`, `player-core.mjs`; add `tests/animation_comparison.test.mjs`; update package-data if necessary.

**Interfaces:** `MotionStage.setComparison(bool)`, `setSkeletonVisible(bool)`, `setExpression(channels)`, `exportReview()` preserve PlaybackClock and generation ownership. Rendering uses original vs corrected tracks, optional authored avatar, and full 3D hand/head envelope. UI exposes five pilots, simultaneous comparison, source skeleton and manual expression preview with unreviewed label.

- [x] 添加停止/定位/切换对照及深度/窄画幅取景的失败回归。
- [x] 实现原版/改进版同步分屏、源骨架与可控面部、五段试点选择和诊断导出；保持现有识别确认行为。
- [x] 在浏览器实际打开后审查画面并修正遮挡/布局问题。

Run: all `tests/animation_*.test.mjs`; browser pilot interactions.
Expected: same timestamp in both views; no autoplay or stopped-clip restoration; all fingertips framed.

### Task 5: 五段实测、整体验证与交付

**Files:** Create `docs/reports/animation-quality-upgrade-2026-11-06.md`; update `docs/csl-animation-workspace.md`, `docs/agents/model-evaluation-pitfalls.md`.

**Interfaces:** Per-pilot evidence includes asset/source SHA, avatar/binding/processing version, frame count, geometry diagnostics, browser result and `language_review=NOT_RUN`. No model-quality or webcam claim.

- [x] 运行 Node 全部动画测试、Python 全量测试；保留退出码与日志。
- [x] 真实浏览器依次检查五段、原始/改进同时间、骨架、表情控制、0.5x/1x、定位停止及窄屏，留少量压缩截图。
- [x] 用独立 reviewer 做一次整体审查并处理重要问题；更新计划/记录及用户打开入口。

Expected: tests have fresh output; engineering results and unrun human acceptance separated; source ZIP unchanged.
