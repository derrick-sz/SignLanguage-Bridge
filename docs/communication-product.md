# 面对面沟通产品使用说明

更新：2026-11-08。主流程面向使用手语与文字面对面交流的双方。模型、数据集和资源审核属于设置中的开发工具，不再作为主页面任务。

## 本机开始使用

打开 <http://127.0.0.1:8879/communicate/>；根地址也会转到该页面。等待页首出现“已准备就绪”。当前两端配置与启动器保存在`E:/githubproject/SignLanguage-Bridge/artifacts/product-bridge-20261108/`：

```powershell
& 'E:/githubproject/SignLanguage-Bridge/artifacts/product-bridge-20261108/start-product.ps1'
```

启动器读取本机`local-profile.json`，复用身份匹配的8765识别服务和8879产品服务。未运行时隐藏启动窗口；端口被其他配置占用时拒绝操作。停止本配置的两端服务运行同目录`stop-product.ps1`。源码入口为`scripts/communication_product.ps1`，另支持`-Action StopProduct`和`-Action RestartProduct`，只停止／重启产品及其自有工作进程。不要在完整动作生成进行中重启。

2026-11-08实测源码位于动画工作树`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`；E盘主检出保留识别分支，产品通过HTTP复用它的现有v3识别协议。本机路径配置、模型、人物运行包、词动作库和完整运行日志需单独准备；换机必须替换配置，参见[动画资产与环境说明](csl-animation-workspace.md)。

### 从GitHub检出后配置

仓库提供`configs/communication-product.example.json`和`configs/communication-recognition.example.json`。复制到自己的本地配置位置，替换示例中的仓库、Python、模型、动作归档、人物、语料与词库路径；`product.bundle_id`必须填写实际词库`manifest.json`中的身份。文字规划器配置沿用`configs/animation/word-composition.example.json`，权重和输入数据需单独准备。模型环境需要项目`api`与相应识别依赖，预览环境需要`api`、NumPy和SciPy；两者可使用同一已准备好的环境。

```powershell
./scripts/communication_product.ps1 -Profile 'D:/local-config/communication-product.json'
./scripts/communication_product.ps1 -Profile 'D:/local-config/communication-product.json' -Action StopProduct
```

2026-11-09的同步与复查见[同步记录](reports/github-sync-2026-11-09.md)；可在GitHub查看的小型证据位于`docs/reports/evidence/communication-product-2026-11-09/`。功能分支推送不等于合入主分支。

## 完成一次对话

1. 左侧选择中国手语或巴西手语Libras，点击“上传视频”，再点击“识别这段手语”。上传上限128MiB，片段不超过15秒。
2. 核对结果并直接修改文字；“查看识别原文”保留模型原始输出。点击“加入对话”才会写入本次对话，识别完成本身不代表确认。
3. 在右侧写下200字以内的中文，点击“转成手语”。已有词动作会快速组合并播放；可暂停、重播、停止、慢放或切换近景。常用表达也可直接点击。
4. 缺词或检测到已知内容遗漏时，页面提示调整、分句或尝试完整生成，不播放残缺表达，也不添加发送记录。完整生成较慢，可取消；它不保证补足语义或动作质量。
5. 下方可复制文字、重新展示历史表达，或导出可读TXT。导出保留被修改的识别原文和人工核对说明。对话只在当前页面会话保留，刷新后清除；清空后可立即撤销。

手机窄屏用“看懂手语／表达文字”切换，保留同一对话。设置中可查看能力、重新连接服务和进入开发工具。摄像头仅在用户主动开启时请求权限，不开启麦克风，预览镜像不改变送入模型的视频；单次录制12秒。本轮用户限定上传视频验证，**物理摄像头与真实浏览器录制未验收**。

## 当前能力和验证范围

| 方向 | 接入能力 | 使用限制 |
|---|---|---|
| 中国手语视频→文字 | CE-CSL TFNet ONNX CTC，v3协商后上传完整片段 | 返回手语词序列，不能标为自动句子翻译；需人工核对 |
| Libras视频→中葡释义 | MediaPipe225与12词RF试点 | 只覆盖现有类别，中葡映射为草案 |
| 中文→中国手语 | 持久T2G文字规划器与词动作库；可选SignSparK慢生成 | 覆盖和质量受模型、素材及衔接限制；缺词／已知遗漏阻断，不能据此保证全部语义 |
| Libras动作输出、完整中葡句子互译 | 尚未接入 | 页面不提供虚假的可用选项 |

识别模型不导入产品进程。产品网关只连接服务器配置的loopback HTTP地址，按次协商representation、预处理、task及模型世代；不一致的返回不会进入文字区。换源、换语种、取消、刷新和清空会阻止旧任务回写；上传临时文件在成功、失败或断开时清理。WebM通过本地FFmpeg处理为无音轨MP4后推理，不能把编码转换后的输出称为与原视频数值完全一致。

本轮真实上传、取消、缺词／遗漏、对话导出和桌面／窄屏检查见[验收记录](reports/communication-product-2026-11-08.md)。工程链路通过不代表手语语义、手形、穿模、自然度或跨语言质量通过；已知新组合句接触问题继续保留在[动画汇总](reports/animation-consolidated-status-2026-11-08.md)。
