# 实时CSL四阶段工程实施与验证（2026-11-10）

本轮按用户确认的顺序开发：SLRT吞吐→网页增量事件→Uni-Sign句末翻译→可选LLM辅助。代码分布在既有两个检出：模型/API为 `E:/githubproject/SignLanguage-Bridge`（`codex/model-contract-auto-captions`，基线782dee3），产品为 `C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`（`codex/local-godette-v14a-demo-20261107`，基线ca20cbb）。已有动画、会议页及词库改动保留。本轮未提交、推送或合并。

使用入口与启动说明见 [实时工程说明](../realtime-csl.md)。实施计划见 [计划](../superpowers/plans/2026-11-09-realtime-csl-engineering.md)。以下数据分别描述功能、吞吐、交付时间和文字质量，不能互相替代。

## 已实现的分工

| 阶段 | 具体行为 | 主要源码 |
|---|---|---|
| SLRT吞吐 | 显式CUDA FP16、检测框复用、可调窗口步长和投票；默认离线协议不变 | `slbridge/slrt/assets.py`、`runtime.py`、`pose.py` |
| 网页增量 | 独立版本WS协议，真实帧号/时间/epoch；有界输入、事件、图像与会话；过载明确失败；取消不制造EOF | `slbridge/realtime_protocol.py`、`realtime_worker.py`、`realtime_routes.py` |
| 句末翻译 | 共享RTMPose原始133点；运动/词/停顿或手动边界；姿态缓存≤8秒；Uni-Sign只在句末执行 | `slbridge/realtime_segment.py`、`realtime_engine.py`、`realtime_service.py`、`uni_sign.py` |
| 文字辅助 | 默认可不使用；独立一运行/一等待；整体8秒截止、响应≤64KiB；只发送实际词与原译文，保留原文和人工修改 | `slbridge/llm_assistance.py` |
| 产品 | 本机轻量HTTP/WS桥；视频/摄像头/共享画面选区；双层字幕、历史核对；本轮可选辅助 | 产品检出的 `communication_live.py`、`web/communication/realtime*.{html,css,mjs}` |
| 启动 | 显式实时配置、完整相邻参数身份、预热就绪及根/嵌套配置SHA检查 | 产品 `scripts/communication_product.ps1` |

模型端保留原Uni-Sign clip和Libras RF路由。实时服务只允许一个CSL原生会话，共用现有Uni-Sign、姿态提取器和模型租约；客户端断线时仍运行的native操作不会被调用方提前释放。窗口词与原始中文独立展示，工程final/committed不代表语义确认。`native_streaming_translation=false`：原生增量的是SLRT词，Uni-Sign仍接收完成的句段姿态。

## 输入、协议与冻结身份

实测目录：`artifacts/realtime-csl-engineering-20261109/`。输入沿用上轮 `artifacts/slrt-llm-scene-validation-20261109-v1/manifest.json`：30条CSL-Daily已见test视频、28句/10签者，另有黑帧/静止人物控制。属于 `retrospective_seen_1086_scene_subset_not_independent_signers`，不属于独立新签者评测。

输入512正方形MP4V，按原FPS发送OpenCV质量90 JPEG；另显式增加1.2秒末帧重复，观察停顿及句末交付。姿态共享前端320px检测复用协议与旧Uni-Sign512px完整文件评测不同，不沿用旧成绩。`validate_realtime_pipeline.py run` 不读取参考，`evaluate` 才用P1 canonical p1-v1评测器；LLM只读prediction-only记录，不接收目标练习句或GT。每轮保留manifest、输入SHA、服务能力/配置身份、事件及阶段源码SHA。

- Uni-Sign：`uni-sign-csl-daily-pose-only-182913ca`，权重SHA256 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`，未推广S2。
- SLRT：`slrt-emnlp24-csl-daily`，checkpoint SHA256 `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d`。
- LLM：本机兼容端点、显式 `deepseek-v4.1-flash`，返回模型须匹配且完成原因须stop；密钥只在服务端内存，产物只含凭据引用。

## 吞吐和原FPS产品桥回放

最初独立加载时宿主Windows提交内存不足，`throughput-v1`无有效模型结果；通过项目启动器释放身份匹配且空闲的服务后串行实测。6正例+2控制的原生诊断中，baseline/FP16/detect5/stride2分别丢13/17/4/0帧。单独子集无丢帧仍需经过完整产品JPEG/WS链路验证，不能把加队列当吞吐优化。

下表均走8879产品桥至8765，包含30正例和2控制。首词为**从ready后采集起点至第一个partial到达测试客户端**，不是每个词相对动作完成的延迟。句末附加时间从原视频EOF计，包括声明的1.2秒重复尾帧；不是物理摄像头端到端SLA。

| 运行 | 推理stride / vote | 正例请求完成 | 黑/静止控制 | 首词P50 / P95 | 末句附加P95 | 全30句BLEU / 编辑相似度 | 主词WER |
|---|---|---:|---|---|---|---|---|
| product-ws-32-v1 | 2 / 7 | 28/30，两次overload | 无词无句 | 0.765 / 2.678s | 1.640s，仅28成功 | 21.0473 / 0.3799，含失败空输出 | 不报，有请求失败 |
| product-ws-32-v2 | 3 / 5 | 30/30 | 无词无句 | 0.782 / 2.498s | 1.461s，30成功 | 25.4540 / 0.4205 | 43.45%，S13/D60/I0，参考168词 |
| product-ws-32-v3 | 3 / 3 | 29/30，一次overload | 无词无句 | 0.679 / 1.118s，仅有词请求 | 1.548s，仅29成功 | 24.5893 / 0.4023，含失败空输出 | 不报，有请求失败 |
| product-ws-32-v4 | 4 / 3 | 30/30 | 无词无句 | 0.678 / 1.142s | 1.422s，30成功 | 25.4540 / 0.4205 | 29.76%，S20/D26/I4，参考168词 |
| product-ws-32-v5，最终词优先调度 | 4 / 3 | 30/30 | 无词无句 | 0.696 / 1.157s | 1.535s，30成功 | 25.4540 / 0.4205 | 29.76%，S20/D26/I4，参考168词 |

v3的S000295_P0008_T00在尚未开始Uni-Sign翻译前积压，最大队列已到16；保留失败，不能仅靠更快首词宣布通过。v4将窗口推理调度为7.5Hz（仍按30FPS接收和提取姿态），给GPU窗口计算留余量，并将投票改为3。没有删采集帧、伪造时间或扩训练词表。

当前显式实时示例及本机配置选择stride4/vote3，句末调度采用下述v5。这里观测到词删除减少，但有20替换和4插入；不能将29.76% WER换算成已验证70.24%准确率，也不能把同一已见数据上的调度诊断称作泛化提升。Uni-Sign中文与v2逐项相同，辅助架构没有把SLRT猜词强行覆盖中文。首词目标在这一固定批次达到；逐词动作结束延迟、物理摄像头和新签者目标尚未验收。

## 单会话连续运行

`stress_realtime_pipeline.py` 将两条固定视频循环，以30FPS单一WS连续送帧，每段加1.2秒尾帧、显式手动句界及0.5秒黑帧分隔。属于合成持续负载，不能当自然对话或真人会议评测。

第一轮 `continuous-10m-v4` 的进程资源枚举阻塞发送线程6秒以上，随后产生11.2秒落后的追赶突发，15.8秒便overload；该轮是**测试仪器无效**，没有完成10分钟。脚本改为仅采样启动器已核验的两个PID，并在发送落后250ms时中止测试、禁止突发补发旧帧；修正后的重跑记录如下。

修正仪器后的v4-r2为有效失败：27.965秒、839帧已发送/821已ack、6个循环、5句完成；生产者最大落后26.58ms，不能归因于突发发送。词队列已有约0.3秒债务时第6句仍立即运行Uni-Sign，短时占用令16帧队列过载。短片逐条完成无法证明单会话长时稳定。

据此加入v5词优先调度：保持同一native/GPU所有者、输入队列16及stride4/vote3，先处理当前帧并排空积压，再运行句末翻译；最多2个已完成句子FIFO等待，1.5秒未获空闲则明确rejected，不覆盖早先句子。自然EOF处理已接收帧后排有效任务，取消只清理等待任务，不伪造完整句。新增summary记录延后/拒答/峰值等待，包含超时任务。该修改不创建额外GPU线程、不增加输入队列。

调度增量独立审查没有Critical/Important，指出“当前已出队帧尚未处理就先翻译”及“超时不计最大等待”两项Minor，均已新增RED/GREEN回归修复。新测试覆盖有积压不翻译、上一句证据/ID不丢、等待截止、FIFO有界、idle与EOF推进、取消清理及前端保留历史。最终持续运行目录为 `continuous-10m-v5`。

v5持续测试**完成604.046秒**：单WS发送18,112帧，18,112帧ack，处理18,112帧、丢帧0；128循环/128句中文、错误0、翻译拒答0。输入队列等待P50=18.21ms、P95=217.68ms，队列峰值12帧，生产者最大迟到28.79ms；所有128个句末任务均有延后调度，等待峰值797ms、句末队列峰值1。这个结果验证该合成负载下队列能回落、前一句没有被覆盖，不能当自然会议语义/摄像头验收。

资源抽样记录于同一summary：模型进程RSS从3873.38到3889.38MiB（+16.00MiB），提交量指标vms增加158.76MiB；产品RSS增加5.66MiB。未宣称零内存增长或长期无泄漏，10分钟观察不能替代更长会议验收。测试期间未运行额外GPU基准、生成任务或更改模型/配置，实际输入队列始终16帧。

## LLM实际辅助与审查修复

`advice-smoke-v1`三次均HTTPStatusError，没有记录原始状态码，原因未确定。重跑修复后的 `advice-smoke-v2` 三次均返回risk，实际耗时2.738/3.121/3.925秒。输入仅来自v2真实预测事件。此项证明本机兼容端点、结构化校验与风险交付可用，不证明纠错收益，也不把risk当正确字幕。

最终服务 `product-advice-v4` 走真实产品WS及辅助HTTP：2正例+2控制全完成，正例的原译文先交付，随后分别收到risk，辅助耗时2.719/3.068秒；控制无词、无句、无辅助请求。两句原译文与同配置未启用辅助的v4对应记录一致。该样本与前述smoke有重叠，不相加为独立准确率样本。

独立审查 `review-findings.md` 找到两项Important并修复：页面勾选辅助未传入start；HTTPX逐操作超时缺少整体截止。修复为真实页面handler传递本次选项，以及可取消async HTTP、整体deadline、字节上限和主动关闭，禁止注入阻塞同步backend，不创建无界替换线程。RED/GREEN日志保留；测试覆盖慢字节流、超时后下一句继续、运行中关闭、同句旧修订、不同句不覆盖、异常不返回凭据及人工编辑原文保留。原生WS完成后按绑定job ID查询辅助，辅助不延长模型租约。

## 工程验证与边界

- 最终调度及Minor修复后模型/API/SLRT/Uni-Sign/新实时模块相关回归：`final-core-verified.log` **167 passed / 1 skipped**；产品识别与桥：`final-product-tests.log` **21 passed**。
- 前端实时、既有采集/选区/产品流程：`final-web-v5-tests.log` **33 passed**；启动器隔离身份场景 **9 passed**。完整仓库测试/远端CI未运行。
- 最终模型wheel的7个新生产模块、产品wheel的5个新实时静态资源均逐字节匹配工作文件，SHA保存在 `wheel-v5-content-checks.json`。此前阶段产物另存，没有覆盖旧wheel。
- 浏览器真实H264文件→取帧/JPEG→产品桥→词/原译文已跑通一次，显示首词0.78秒，原文“我们晚上饭店怎么样?”，console warn/error为空。原MP4V浏览器无法解码，另生成H264 fixture，不把重编码输入称为原视频字节相同。
- 最终v4浏览器再次播放同一H264 fixture并勾选辅助：首词显示0.68秒，独立显示“手语词为‘酒’，草稿写成‘饭店’，请核对。”，原中文保留；console warn/error为空。截图 `browser-final.jpg`、DOM与fixture SHA `browser-final.json`。这只证明可见交付和冲突提示，不以文字流畅或LLM风险提示判定动作本义。
- 最终v5部署浏览器再次验证同一文件与勾选传参：首词0.67秒、原译文与独立风险提示可见，console warn/error为空，截图/DOM为 `browser-final-v5.jpg/json`。验证后复选框恢复不勾选，保留页面可继续使用。本次不是摄像头录制。
- 单会话持续测试及最终32段结束后，原产品上传接口实际POST200，返回“我们晚上饭店怎么样?”，HTTP耗时2.462秒；实时服务仍ready且非busy，CSL/Libras均ready、generation1。见 `post-live-clip.json`。这验证共享提取器/原生租约退出后上传路径仍可使用，没有重新评估Libras内容准确率。
- 真正运行的配置内容身份验证：分别临时改根JSON和嵌套SLRT JSON，在恢复原字节的finally保护下调用Start；两次都被拒绝复用，8765/8879 PID保持不变，见 `config-digest-rejection.json`。未以停止不匹配服务来解决身份冲突。
- 服务只能通过项目启动器核对全部受影响服务身份后Stop/Start。一次直接重启命令被自动审查拦截，改用该启动器成功；无未解决的审批阻塞，无任意进程终止。

物理摄像头/共享画面权限、新签者、用户原先“我想睡觉”录制、真人CSL语义/手形审核均 **NOT_RUN**。静止控制通过仅限这两条。BLEU不是准确率，WER也不能不加定义地换算准确率；实时交付可用与识别内容可靠需分别验收。
