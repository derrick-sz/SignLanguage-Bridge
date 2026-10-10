# 统一模型与自动字幕实施记录

用户已批准 2026-10-11 路线进入实施。本计划细化本轮核心交付；不增加训练、数据下载或 LLM 范围。

目标：带实际时间戳的持续采集 → 自动完整片段 → 统一模型服务 → 有序字幕事件 → 桌面控件。

1. `model_adapters.py`、`runtime_contracts.py`、`runtime.py`：注册输入与任务适配器，绑定请求版本，保留 RF/ONNX/CTC 的原预处理。验证错表示拒绝、typed 输出、预热回滚、deadline、取消与独立语言路由。
2. `segmentation.py`：独立活动观察与时间状态机。验证静态动作、内部暂停、质量丢失、保持去重、重新触发、EOF 与资源上限。启发式不冒称经过人体数据验证。
3. `caption_events.py`、`session.py`：动作 ID 幂等、按顺序提交、独立历史撤销清空；最多 2 观察帧、1 在途与 1 等待推理片段。验证迟到结果、source epoch、generation、取消及背压。
4. `api.py`、`api_schemas.py`、`api_cli.py`：v3 typed multipart API、能力协商、字幕 session 与事件动作，保留 v1/v2。验证实际 HTTP 上传、重复请求、超时与会话隔离。
5. `desktop.py`、`desktop_cli.py`：挂载自动模式、候选状态、事件历史、撤销和清空；保留手动操作。通过实际 Qt 控件和真实视频回放验证。
6. `experiments/validate_auto_captions.py` 与完成报告：实际时间戳回放、真实 RF 模型、localhost HTTP、事件读回与控件读回；如实报告各阶段耗时、硬件未测项及识别质量限制。

原 checkout 的文档和审计脚本不纳入本轮提交。只提交本轮明确文件，使用 `codex/` 功能分支；验证通过后非强制同步，不合并 main。
