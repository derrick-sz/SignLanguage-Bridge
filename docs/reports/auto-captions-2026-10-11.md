# 模型统一接入与自动字幕：实现及验证报告

GPU 后续已验证：独立环境使用 RTX 5060 Ti，完整 CTC 输入的模型前向约 0.54–0.56 秒，文件推理总耗时约 1.92–1.99 秒，真实 HTTP 上传约 2.07–2.25 秒；实际卷积、LSTM、矩阵乘法由 CUDA 执行。根因修复、完整数值对照和真实 HTTP 记录见[GPU 实测报告](ctc-gpu-inference-2026-10-11.md)。下文 CUDA 不可用与 693 秒条目是当时 CPU 环境的历史记录；真人长流和字幕显示 P95 仍未验收。

后续性能修复：同一 CTC 完整视频现连续三次总耗时为 **6.31–6.62 秒**，最新真实 HTTP 上传约 7.17 秒返回；完整 logits 与原模型一致。本报告下方约 693 秒为修复前的历史单次记录，保留追溯；当前原配置复测约 54.9 秒，未重新复现那次 693 秒。根因、当前对照和 546 passed / 9 skipped 回归见[CTC CPU 性能修复报告](ctc-cpu-performance-fix-2026-10-11.md)。2 秒与 P95 目标仍未验收通过。

本轮已实现可扩展模型服务、HTTP v3、停顿分段、字幕事件及桌面自动模式。真实 RF 的 HTTP 和 Qt 工程回放通过；**自然视频默认分段、未知动作拒识、真人长流质量和端到端 P95 尚未通过验收**。当前 CTC CPU 实测远超实时目标，不应将这次交付称为实时中葡手语翻译已完成。

基线为 `b668453`，开发分支为 `codex/model-contract-auto-captions`。没有重训、下载数据或修改模型权重。既有研究文档和 audit 脚本保留在工作区，不包含在本轮实现提交中。

## 已实现的行为

| 范围 | 当前行为 |
| --- | --- |
| 模型 adapter | 注册 backend 和输入表示；声明语言、任务、shape/dtype、完整预处理版本、词表、设备和 clip/stream 能力；实现 load、warmup、infer、close。既有 RF、ONNX 分类和 CTC 通过实际 adapter 调用。 |
| 模型管理 | 预热后激活；加载失败保留当前模型；generation、精确模型/预处理绑定、租赁及回退；迟到或取消结果失效。按语言使用独立 manager，不靠全局切换伪装同时服务。 |
| typed 结果 | 分类、gloss_sequence、text 各保留自己的 prediction；输出模型与预处理版本、实际 provider/device、质量和阶段时延；原始 score 不自动校准。 |
| HTTP | v3 能力查询、协商、切换、回退、统一推理、字幕 session 和操作；旧 v1/v2 响应兼容。保留上传/字段限额、bearer 校验和临时文件清理。 |
| 自动分段 | 独立 MediaPipe 活动观察图与真实时间戳；开始候选、捕获、结束候选、重新准备；前后缓冲、静态词、内部停顿、同词再次提交；质量丢失、时间缺口、过长及 EOF 明确拒绝。 |
| 字幕事件 | 按 session、segment、source epoch、generation 和 sequence 绑定；候选与正式记录分离；乱序完成顺序提交；按动作身份去重，连续相同文字可保留两条；撤销和清空可重试且不会使旧结果补回。 |
| 桌面 | 已挂载的“自动追加字幕”、候选状态、“字幕记录”列表、撤销和清空；字幕与支持词表分标签显示；保留手动视频/录制路径。按输入能力加载模型，支持无旧 bundle 属性的新 adapter。 |

观察队列最多 2 帧，保留最新观察并统计丢帧；模型最多 1 个运行片段和 1 个待处理片段，满额明确拒绝。环形图像缓冲受秒数、帧数和字节上限约束，不能静默截断后识别。字幕历史默认最多 200 条；API session 默认 64 个，请求和 action 记录分别最多 1024 个，到限额后需新建会话。

取消、换源、换模型和清空会使旧候选及回调失效。原生模型调用不能安全强制中断；它仍占用该语言容量至返回，资源随后释放。桌面自动请求默认 10 秒期限，HTTP 使用服务配置的上限；期限不构成原生调用的强制停止保证。

## 验证结果

| 验证项 | 结果 | 证据与边界 |
| --- | --- | --- |
| 全量回归 | PASS：533 passed、9 skipped | `.venv-libras` 中启用真实 RF file/captured parity；9 项跳过全部因未安装 Torch，不能计为通过。 |
| 状态机与生命周期 | PASS | 静态动作、动作内暂停、连续同词、缓冲限额、遮挡、换源、超时、取消、模型切换、失败保留及关闭等确定性回归。行为测试不等于真人质量测试。 |
| HTTP v3 | PASS | 真实路由逻辑与类型校验、错误码、完整 payload 去重、并发容量、超时、旧源/旧 generation 丢弃、字幕操作；边界回归用确定性 adapter。 |
| 真实 RF localhost HTTP | PASS，工程回放 | 实际子进程服务器、socket、RGB → MediaPipe → 分段 AVI 上传 → RF → session 读回；重试仍为同一事件，撤销/清空通过。 |
| 真实 RF Qt | PASS，工程回放 | 实际挂载复选框驱动生产 session、捕获、分段、推理和列表；撤销后重放生成 sequence=2，再对非空历史点击清空。RGB 预处理和 RF 真实运行；活动观察复用离线真实 MediaPipe 缓存。 |
| 桌面布局 | PASS，所测尺寸 | 980×720、1280×820 已挂载控件的可见性、边界和字幕/列表无重叠测试；另有实际窗口截图。不是所有显示缩放的像素验收。 |
| 摄像头 0 | PASS，采集/观察/释放 | 3 秒探测：实际观察 48 帧、丢帧 25、正式字幕 0；capture/model/observer worker 均释放。没有保存摄像头像素、没有动作真值，识别质量未评估。 |
| 9 个原始短视频默认端点 | 未通过自动字幕链路 | ready 全部为 0，原因是 quality_lost 或 incomplete_at_eof；没有得到可提交片段，因此没有执行该路径的 RF。详细源时间表见分段文档。 |
| 真实 ONNX 分类 | PASS，CPU | 原始 Quero 视频返回 item:7，score≈0.81；自动设备明确回退 CPU 并报告原因。仅证明实际调用，不是分类准确率评估。 |
| 真实 CTC | PASS，协议调用；实时性能 FAIL | CPU 处理 8.2 秒、247 源帧（260 padded frames），输出“平方 / 我 / 自己 / 。”；未对照标注评分，不能称内容正确。约 693 秒总耗时，无法满足 2 秒目标。 |
| CUDA | 当前环境不可用 | ORT 1.23.2 可用 provider 只有 Azure、CPU；硬件为 RTX 5060 Ti，但不存在 CUDAExecutionProvider。显式 CUDA 明确报不兼容，无 GPU 推理通过结论。 |
| wheel | PASS | 隔离构建成功，检查新模块包含且从 wheel 隔离导入成功。最初 no-build-isolation 因媒体环境缺 setuptools 失败，随后使用标准隔离构建修复。 |

测试记录：`artifacts/auto-captions-smoke-20261011/tests.xml`。9 个 Torch 跳过项位于 `tests/test_tfnet_export.py` 的原生模型/导出测试；已有 ONNX CTC 权重在本次探测中真实运行，不依赖这些跳过项。

## 实测时延

工程回放使用原始 `7_Quero/INFORMANTE 2/IMG_0696.mp4` 的前缀，在 1.5 秒处追加 1.2 秒末帧保持，共 82 帧。这是**显式合成停顿**；没有当作自然手语或实时摄像头准确率样本。默认参数产生 1 段 `[0, 1.6]`，结束候选从源时间 1.5 秒至 2.1667 秒确认，约 666.7 ms。

| 路径 | 预处理 ms | 推理 ms | 服务总耗时 ms | 另记 |
| --- | ---: | ---: | ---: | --- |
| RF HTTP（合成停顿工程回放） | 1912.8 | 4.5 | 1917.4 | socket 壁钟 1937.0 ms |
| RF Qt（缓存真实活动观察） | 1122.0 | 6.1 | 1129.2 | queue=0，观察丢帧/片段均为 0 |
| ONNX RF（原始完整视频） | 2056.0 | 0.3 | 2056.3 | auto → CPU |
| CTC（原始完整视频） | 1311.6 | 691614.1 | 692938.9 | CPU，单次同步探测 |

实际 MediaPipe 离线活动观察的 82 帧壁钟为 4016 ms。该项与 Qt 缓存观察时延不同；不能用缓存回放的零丢帧推断现场观察吞吐量。上述均为单次实测，**不是 P95，也没有得到“动作结束到字幕显示≤2秒”的结论**。RF 的成本主要在关键点提取，换成 ONNX 不会自动消除提取延迟。CTC CPU 的明显慢速需单独解决，不能提高 HTTP 超时后宣称实时达标。

## 复现与本地产物

启动、分段配置、API 两语言路由及工程回放命令见[使用说明](../auto-captions-usage.md)；HTTP 字段和错误码见[API 契约](../unified-api.md)；新模型实现示例见[adapter 示例](../model-adapter-example.md)；9 个未成功原始回放与默认阈值见[分段契约](../automatic-segmentation.md)。

完整回归启用了真实保存模型的可选测试：

```powershell
$env:SLBRIDGE_LIVE_VIDEO=(Resolve-Path 'Libras-EQT-UECE (RGB)/7_Quero/INFORMANTE 2/IMG_0696.mp4').Path
$env:SLBRIDGE_LIVE_MODEL=(Resolve-Path 'artifacts/experiments/libras-rf-pilot-20261009/models/libras-rf-pilot-20261009').Path
.venv-libras/Scripts/python.exe -m pytest -q -ra --junitxml=artifacts/auto-captions-smoke-20261011/tests.xml
```

本地忽略目录中保留：

- `artifacts/auto-captions-smoke-20261011/`：fixture、实际观察、分段、HTTP/Qt 读回、server.log、截图、JUnit 与 wheel。
- `artifacts/auto-camera-probe-20261011.json`：真实摄像头无像素探测与 worker 释放证据。
- `artifacts/unified-runtime-probe-20261011.json`：实际 RF/CTC 的模型版本、处理版本、provider、typed 结果与阶段耗时。
- `artifacts/unified-provider-probe-20261011.json`：不运行慢速 CTC 的独立 RF/provider 探测。

这些产物依赖当前合法本地视频及可信模型，不随源码提交。新克隆需要自行提供模型包和输入，示例 catalog 的路径不是可下载保证。

## 待验证或未实施

1. 标注的自然停顿长流、静态词、连续相同词、遮挡与未知动作的漏段/误切/重复、内容准确率和 P95。默认启发式及 0.65 秒等待未调成经验证的产品参数。
2. 兼容 ORT GPU 环境与真实 CUDA CTC 性能；当前 GPU 硬件存在不代表 provider 可用。原生 CPU 慢速和超时后容量释放限制已公开。
3. Libras174 的 321 维 TCN 尚未接入；不能用 225 维补零/截断替代。需要独立冻结训练与在线提取协议。
4. CTC 输出仍为 gloss，RF 中文释义仍为待核验草案；没有上线 LLM 辅助、整句葡语翻译或将未审核映射标为已审核。
5. 原生 stream/frame HTTP 连接、字幕持久化/跨进程会话与全面显示缩放像素验收不在本轮完成范围。当前持续采集后调用 clip 模型。
