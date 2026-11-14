# 手语动画工作台使用说明

第一期提供现有动作资源的独立预览工作台，可接收中文输入、现有识别结果和桌面端候选。先匹配完整表达，再由使用者确认播放。没有覆盖的文字会被完整保留并阻止播放。

## 本机打开

当前功能分支为 `codex/csl-animation-workspace`，工作区：

`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`

服务启动后打开 [动画工作台](http://127.0.0.1:8874/animation/)。服务默认只监听 loopback；不启动自动播放。

在上述工作区打开 PowerShell，仅看动画可运行：

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -m slbridge.animation.cli --archive E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip --expected-sha256 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 --port 8874
```

要同时使用现有 Uni-Sign 识别，在本地保存 `artifacts/animation-acceptance/catalog.json`：

```json
{"schema_version":1,"models":{"csl":"E:/githubproject/SignLanguage-Bridge/models/external/uni_sign/model_bundle"}}
```

然后用已配置 CUDA 的 Python 启动：

```powershell
& E:/githubproject/SignLanguage-Bridge/.conda-env/python.exe -m slbridge.animation.cli --archive E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip --expected-sha256 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 --catalog artifacts/animation-acceptance/catalog.json --model csl --port 8874
```

同一端口只运行一个服务。动画重播不需要 GPU 推理；视频识别依旧使用原有模型环境。其他机器可安装项目的 `api` 可选依赖，并提供自己的动作 ZIP、模型目录和 Python 环境。动作资源与大模型不包含在 Python wheel 中。当前机器的 G 盘数据和权重目录无需移动。

### Linux 服务器内网访问（2026-10-30）

在 `project/git-worktree/` 中，使用已安装 API 依赖的环境启动；`--host` 可指定服务器内网地址，默认仍为 `127.0.0.1`：

```bash
.venv-animation/bin/python -m slbridge.animation.cli \
  --archive ../../assets/action_archive/signbridge-chrome-extension-motion-demo-20261022.zip \
  --expected-sha256 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 \
  --host 10.45.25.55 --port 8874
```

自己的电脑可访问 `http://10.45.25.55:8874/animation/`，前提是能连通服务器内网和该端口。三维渲染使用访问者浏览器的 WebGL 2；服务器桌面的图形加速不影响其他电脑的浏览器。此命令只加载现有动画，不登记识别模型。也可保留默认监听地址，通过 SSH 或 VS Code 转发 8874 端口访问。

## 预览步骤

1. 输入“认识你很高兴”，或在动作库中选择现成表达。
2. 点击“确认并预览”。存在多个同名资源时，先选择具体动作版本。
3. 用暂停、继续、重播、停止、速度和时间轴检查动作；正面保留整个片段的手部活动空间，侧面与近景供检查细节。
4. 在同一暂停时刻切换“原始显示／显示优化”。对照使用同一模型与动作；显示优化改变光照、衣服配色和轻度尖峰稳定，不代表语言纠错。
5. 展开“动作来源与显示处理”查看资产 ID、内容摘要和显示处理信息。

例如输入“认识你很高兴不要”，规划器不会只播放前半句。系统提示“没有完整表达资源”，整句不会播放。标点/空白归一化不删除数字、否定或小数点。

## 从识别结果进入

点击“接入识别结果”，选择已登记模型并加载（启动时已加载的单模型会自动选中），选择视频或姿态文件，再点击“识别文件”。输出只进入待核对输入框；原始文本保留。可以手动修订后重新确认，但修订内容是人的输入，不能当成模型正确输出。

Qt 桌面结果区新增“在动画工作台核对”。点击后把 CSL 文本/Gloss 候选和来源传到浏览器草稿；不会自动播放。工作台需先启动。默认地址为 `http://127.0.0.1:8874/animation/`；本地换端口时用 `SLBRIDGE_ANIMATION_URL` 配置同机地址。该入口位于通用桌面窗口，独立 SLRT 专用面板暂未增设相同按钮；其 Gloss 可先手动输入工作台。

隔离工作区启动桌面 Uni-Sign 时显式指定原模型目录：

```powershell
& E:/githubproject/SignLanguage-Bridge/.conda-env/python.exe -m slbridge.desktop_cli --unisign --unisign-model E:/githubproject/SignLanguage-Bridge/models/external/uni_sign/model_bundle
```

## 当前能力与 Blender

- 可注册 100 段现有真实动态资源：14 段参考重建、3 段候选、83 段 SMPL-X 拟合。83 段的批准只涉及动作质量，所有片段的语言审核均为未验证。
- 目前规划器只匹配已有完整表达。自由文本 CSL 语法、词间合成、空间指代、否定/疑问表情、CSL↔Libras 动作转换尚未实现。
- 本轮复用 Godette 角色，改进画面与时间控制，没有重做角色网格、手指蒙皮或面部轨，也没有宣称碰撞/接触问题已经解决。
- 当前演示不需要 Blender。更换更合适的人物、去掉影响手形观察的衣饰、修补手指绑定/蒙皮、加入面部形态或修正具体动作时，再使用 Blender 做离线资产工作；导出后仍需在 Three.js 复验。
- 没有训练 SOKE、重建全部视频或搬移 G 盘数据。后续先精选动作做 CSL 使用者理解任务，再决定角色改造和视频重建扩库规模。

[实施与验收记录](reports/csl-animation-workspace-acceptance-2026-10-28.md)列明真实请求、回归测试、已修复问题和未测边界。

