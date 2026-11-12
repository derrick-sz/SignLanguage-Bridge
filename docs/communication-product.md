# 面对面沟通产品使用说明

更新：2026-11-11（实时字幕、逐句核对与可复现示范）。主流程面向使用手语与文字面对面交流的双方。模型、数据集和资源审核属于设置中的开发工具，不再作为主页面任务。

2026-11-12 补充：[数字人当前缺陷、证据与改进计划](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/superpowers/plans/2026-11-12-avatar-motion-quality.md)集中记录动作数据来源、实时渲染与录像回放的区别、未通过的回认/几何候选及后续验收条件。最近优化候选尚未替换产品动作，手语语义与直接实时采集仍待验收；历史使用流程及各次测试按原日期理解。

## 本机开始使用

打开 <http://127.0.0.1:8879/communicate/>；根地址也会转到该页面。等待页首出现“已准备就绪”。当前两端配置与启动器保存在`E:/githubproject/SignLanguage-Bridge/artifacts/product-bridge-20261108/`：

```powershell
& 'E:/githubproject/SignLanguage-Bridge/artifacts/product-bridge-20261108/start-product.ps1'
```

启动器读取本机`local-profile.json`，复用身份匹配的8765识别服务和8879产品服务。未运行时隐藏启动窗口；端口被其他配置占用时拒绝操作。停止本配置的两端服务运行同目录`stop-product.ps1`。源码入口为`scripts/communication_product.ps1`，另支持`-Action StopProduct`和`-Action RestartProduct`，只停止／重启产品及其自有工作进程。不要在完整动作生成进行中重启。

2026-11-08实测源码位于动画工作树`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`；E盘主检出保留识别分支，产品通过HTTP复用它的现有v3识别协议。本机路径配置、模型、人物运行包、词动作库和完整运行日志需单独准备；换机必须替换配置，参见[动画资产与环境说明](csl-animation-workspace.md)。

### 从GitHub检出后配置

仓库提供`configs/communication-product.example.json`和`configs/communication-recognition.example.json`。复制到自己的本地配置位置，替换示例中的仓库、Python、模型、动作归档、人物、语料与词库路径；`product.bundle_id`必须填写实际词库`manifest.json`中的身份。文字规划器配置沿用`configs/animation/word-composition.example.json`，权重和输入数据需单独准备。模型环境需要项目`api`与相应识别依赖，预览环境需要`api`、NumPy和SciPy；两者可使用同一已准备好的环境。

本机中文识别已切为`recognition.csl_model=unisign_pose_stage1`，对应catalog中的CSL-Daily Uni-Sign基础版bundle；`recognition.libras_model=libras-rf`。识别环境使用含Torch、RTMPose和API依赖的`.conda-env`，`recognition.device=auto`，`startup_timeout_seconds=180`；本次实际核对模型和姿态均为CUDA。启动器按profile读取模型别名和设备，未填写新字段的旧profile仍沿用TFNet/CPU兼容值，因此迁移时必须明确填写。这里的别名是显式选择，没有自动把`dev/test`的S2升级为生产默认。切换记录及验收范围见[本轮报告](reports/communication-unisign-switch-2026-11-09.md)。

```powershell
./scripts/communication_product.ps1 -Profile 'D:/local-config/communication-product.json'
./scripts/communication_product.ps1 -Profile 'D:/local-config/communication-product.json' -Action StopProduct
```

2026-11-09的同步与复查见[同步记录](reports/github-sync-2026-11-09.md)；可在GitHub查看的小型证据位于`docs/reports/evidence/communication-product-2026-11-09/`。功能分支推送不等于合入主分支。

## 完成一次对话

实时句子可直接加入面对面对话：先完成本句并暂停或中止采集，核对并确认后点击“加入对话并回复”。首次打开表达页，后续复用原窗口；实时历史和已有回复草稿保留，导入不会自动生成动作，仍由用户填写并提交回复。超时可回实时页重试，未关闭的原对话窗口继续使用；首次打开需允许本机弹窗。来源/训练标识和识别原文随对话与TXT保留。工程接线、68 Node及32接口测试和8模块HTTP已核验，实际跨窗口操作仍待验，见[接续报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/realtime-conversation-2026-11-11.md)。

实时中国手语入口为 `/communicate/realtime.html`，也可从对话页顶部进入。选择视频、摄像头或共享画面并框选手语者后，SLRT先交付词，Uni-Sign在停顿或“结束这句（继续）”后补完整中文。“完成本句并暂停”排完已有画面并交付尾句；“中止识别”或换源取消当前未完成句。历史可核对并保留识别原文；8秒以上表达请及时分句。可选“辅助核对”默认不勾选，只发送识别文字，建议单独显示。原上传流程如下。

本机“训练示范与问题回放”提供3条已测训练短句与1条已知误追加案例：载入后可观看，再点击“识别此示范”从头识别。示范标注只作对照，目标文字不送入识别；历史保留训练、问题、本地视频或实时采集来源。首词从点击开始计时，展开“实时状态”查看准备、排队、处理、编码及采样跳帧。此入口支持重复验证，不能当成跟练或摄像头语义准确率证明；实测见[示范与延迟复核](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/realtime-demo-catalog-2026-11-11.md)。部署可在profile中显式配置`product.demo_catalog`，未配置则隐藏；更改前先按旧profile StopProduct，再按新配置Start，模型进程可保持不动。

2026-11-11后续：主字幕按本轮隔离，换源/新尝试不会回退旧译文，历史仍可核对；空帧结束明确提示重试。采集交付2秒无进展会停止，等待编码时也可取消，状态新增采集/发送/处理计数。真实静态资源和空输入协议已回读，43 Node通过；实际网页采样因浏览器控制连接超时仍未认证，见[当前恢复报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/realtime-capture-recovery-2026-11-11.md)。

实时能力需在本地profile配置 `recognition.realtime_config`，模型服务需包含新的 `csl-realtime-v1` WebSocket接口；产品桥不加载第二份模型。启动器同时验证配置内容SHA，配置变更后按同一profile Stop再Start；更换配置文件路径则先用旧profile停止。模型源码、配置和使用说明见 [实时工程说明](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/realtime-csl.md)，实测范围见 [实施报告](https://github.com/betterway614/SignLanguage-Bridge/blob/codex/realtime-model-20261112/docs/reports/realtime-csl-engineering-2026-11-10.md)。

1. 左侧选择中国手语或巴西手语Libras，点击“上传视频”，再点击“识别这段手语”。上传上限128MiB，片段不超过15秒。
2. 核对结果并直接修改文字；“查看识别原文”保留模型原始输出。点击“加入对话”才会写入本次对话，识别完成本身不代表确认。
3. 在右侧写下200字以内的中文，点击“转成手语”。已有词动作会快速组合并播放；可暂停、重播、停止、慢放或切换近景。常用表达也可直接点击。
4. 缺词或检测到已知内容遗漏时，页面提示调整、分句或尝试完整生成，不播放残缺表达，也不添加发送记录。完整生成较慢，可取消；它不保证补足语义或动作质量。
5. 下方可复制文字、重新展示历史表达，或导出可读TXT。导出保留被修改的识别原文和人工核对说明。对话只在当前页面会话保留，刷新后清除；清空后可立即撤销。

手机窄屏用“看懂手语／表达文字”切换，保留同一对话。设置中可查看能力、重新连接服务和进入开发工具。摄像头仅在用户主动开启时请求权限，不开启麦克风，预览镜像不改变送入模型的视频；单次录制12秒。本轮用户限定上传视频验证，**物理摄像头与真实浏览器录制未验收**。

## 当前能力和验证范围

| 方向 | 接入能力 | 使用限制 |
|---|---|---|
| 中国手语视频→文字 | CSL-Daily Uni-Sign基础版，v3协商后上传完整片段 | 输出整句中文候选；仍需人工核对，非原生流式、非真人泛化质量验收 |
| Libras视频→中葡释义 | MediaPipe225与12词RF试点 | 只覆盖现有类别，中葡映射为草案 |
| 中文→中国手语 | 持久T2G文字规划器与词动作库；可选SignSparK慢生成 | 覆盖和质量受模型、素材及衔接限制；缺词／已知遗漏阻断，不能据此保证全部语义 |
| Libras动作输出、完整中葡句子互译 | 尚未接入 | 页面不提供虚假的可用选项 |

识别模型不导入产品进程。产品网关只连接服务器配置的loopback HTTP地址，按次协商representation、预处理、task及模型世代；不一致的返回不会进入文字区。换源、换语种、取消、刷新和清空会阻止旧任务回写；上传临时文件在成功、失败或断开时清理。WebM通过本地FFmpeg处理为无音轨MP4后推理，不能把编码转换后的输出称为与原视频数值完全一致。

本轮真实上传、取消、缺词／遗漏、对话导出和桌面／窄屏检查见[验收记录](reports/communication-product-2026-11-08.md)。工程链路通过不代表手语语义、手形、穿模、自然度或跨语言质量通过；已知新组合句接触问题继续保留在[动画汇总](reports/animation-consolidated-status-2026-11-08.md)。

上传v3 adapter仍处理完整片段；另有`csl-realtime-v1` WebSocket已接通SLRT增量Gloss与Uni-Sign句末中文，Uni-Sign本身不是原生流式翻译模型。数据集回放的实时交付与吞吐已有工程实测，陌生签者、摄像头与真人语义质量仍待验证。双模型/LLM辅助保留各自原文和词证据，不自动覆盖人工确认。

## 完整原生词条优先

2026-11-09针对“谢谢”修订选材：单独输入时精确匹配完整单词原句，句中出现对应Gloss时也优先选完整原生词条。仅去除句末标点进行精确匹配，不把长句模糊替换成“谢谢”。源动作、指骨参数和预览FPS保持原样。

离线构建器支持`--standalone-only --label 谢谢`：验证配对LMDB与标注摘要，检查该源视频的全部标注只有一个Gloss，再导出完整原生序列，同时保留原ISO范围、完整源范围与原始blob摘要。不能先过滤“医生”等其他词，再把“谢谢医生”误报成独立词条。

可将该小库加入词库合并流程，并在`word_library`配置新合并库；可选`standalone_library`指向独立完整词库，为精确输入提供直接播放。示例中的`null`表示未配置。缺词和已有遗漏阻断保持原有行为；词条完整不等于语言语义和自然度已经认证。具体源、回归与实机画面见[本轮手势修订](reports/thanks-native-gesture-2026-11-09.md)。
