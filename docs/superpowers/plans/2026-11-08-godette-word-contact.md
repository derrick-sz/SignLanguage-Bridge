# v14A Word Contact Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task in the current session.

**Goal:** 修正新增词句的避让跳动，并按用户选定方案调校“西瓜回家”的整段接触。

**Architecture:** 通用路径只搜索臂腕平移；精确绑定的单条接触轨提供稳定旋转轴、时间包络和出口。重定向器应用整段轨迹，原始参数及指骨局部角保持不变。

**Tech Stack:** 现有 Three.js / GLTFLoader、Node test、Python pytest、Windows 本地 FastAPI 演示。

**Spec:** `docs/superpowers/specs/2026-11-08-godette-word-contact-design.md`

## Global Constraints

- 单次总平移 ≤0.06，附加腕转角 ≤15°，附加逐帧转角 ≤3°，附加逐帧位移 ≤0.015。
- 精确资源/参数/配置绑定；保持原始及 refined 输入、30 根指骨局部角和原始显示。
- 不改并发场景模块；核心代理通过不当作语言或完整碰撞通过。

## Review Focus

- 相同名称但资源、参数或修订配置变化时拒绝专用轨迹。
- 原始显示或未启用 v14A 避让时不应用专用接触轨。
- 首尾平移/旋转归零，帧间插值不能重新穿入头部核心。
- 实测腕部位移与旋转逐帧变化应匹配诊断。
- 所有其他片段不得引入新的残余核心重叠或指骨变动。

### Task 1: 保留通用平移修正

- [x] 保存失败自动转腕源码与日志，撤回该候选。
- [x] 运行真实资产“女儿写作业”“药物过敏吗？”和已有手形/医院 7 项回归。

### Task 2: 单条整段接触轨

- [x] 保留 S000931 失败断言和原始扫描，固定轴/固定出口探针找到可行解。
- [x] 在现有 `body-clearance.js` 导出单条轨选择器，接入 retarget 与来源提示；复用已运行服务允许的静态模块。
- [x] 在 `tests/animation_word_clearance.test.mjs` 验证实际世界坐标、附加转角、全段 1/16 帧接触、输入保留及绑定拒绝。

### Task 3: 批量和实际演示复核

- [x] 运行 104 段、28,668 帧扫描、Node 动画回归和 Python 回归。
- [x] 浏览器实际播放三条词句，保存同帧截图与控制台证据。
- [x] 写本轮报告，更新项目坑点；仅说明实际通过的范围。

全仓库 Python 一次出现 ONNX/SciPy 子进程导入失败，单测隔离重跑通过；保留两份证据，未将全仓库记录改为全绿。
