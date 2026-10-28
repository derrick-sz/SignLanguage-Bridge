# CSL 动画工作台实施与工程验收（2026-10-28）

## 结论与范围

第一期已在隔离功能分支实现：验证现有 100 段动作归档、完整表达精确规划、独立 Three.js 工作台、原始/优化显示对照、播放控制、现有识别 API 输入与桌面显式草稿入口。完成的是资源复用与可运行的工程切片，语言理解和自然 CSL 生成没有通过验收。

- 基线：`782dee3f128f727373dcdd3e4c2590d7bdc987da`。
- 实现提交：`92c8993`（registry/planner）、`1f3d739`（API/CLI）、`40773a0`（工作台/桌面）、`9d44f193587d178e6654a5bb6f072d3cfbc05744`（审查修复与回归）。最后补充 `c8aac293c60e09b118c586d34f30d31e1e2c3b3c` 清除停止后的旧轨引用；后续交付提交只补文档。
- 分支：`codex/csl-animation-workspace`。工作区：`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`。
- 原项目目录、现有权重与 G 盘数据未移动；原工作目录的本地研究笔记保留。未推送远端。
- [批准设计](../research/csl-animation-integration-and-visual-upgrade-design-2026-10-28.md)、[执行计划](../superpowers/plans/2026-10-28-csl-animation-workspace.md)、[使用说明](../csl-animation-workspace.md)。

## 实现

| 部分 | 本轮行为 | 边界 |
|---|---|---|
| MotionRegistry | 一次性只读 ZIP 快照；BUILD/资产摘要、路径/大小、55 关节拓扑和有限姿态验证 | 已验证来源包，不等于语言审核 |
| 规划器 | 完整表达匹配、同名版本选择、显式确认研究预览 | 不分词拼凑，不猜未覆盖后缀；否定和数字不丢失 |
| API | 可选挂载 `/animation/`、`/v1/animation/*`，固定资源白名单，同源模型接口 | 现有 API 在不传 registry 时维持原行为；鉴权沿用现有中间件 |
| 三维舞台 | 灯光、阴影、深色背景、衣饰色彩抑制，完整运动范围固定相机，侧面与近景 | 仍用 Godette；没有新模型/新蒙皮/面部轨 |
| 动作播放 | 原始姿态保留；目标四元数孤立尖峰稳定最多约 3°；可对照原始轨 | 没有承诺轨迹/碰撞/接触或语言纠错 |
| 时间线 | 暂停/继续、重播、停止、变速、定位；同步刷帧与取消代次门 | 未给出 30/60 FPS、显存或 P95 指标 |
| 识别入口 | 现有模型加载与 `/v3/recognitions`；输出需核对，保留原始文本 | 本轮真实跑的是 Uni-Sign；其他适配器未逐个做动画 UI 验收 |
| 桌面入口 | 通用 Qt 窗口显式打开本机工作台草稿，带 request/epoch/model 元数据 | 已做 Qt 自动测试；没有新增独立 SLRT 专用面板按钮 |

## 资源身份

归档为 `E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip`，40,752,631 bytes，SHA256：

`4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`

本轮实际验证并注册 100 段：14 reference、3 candidate、83 fitted；83 段 `motion_review=demo_motion_only`，17 段 `unverified`；100 段 `language_review=unverified`。批准绑定 approval ID、source/fitted 摘要及与上游一致的 1e-6 规范化姿态摘要。改变姿态或拟合后，旧批准不再继承。所有 100 段都做结构校验，不代表逐段真人理解、逐帧视觉或碰撞检查。

Three.js、GLTFLoader、模型和必要依赖从已验证归档固定路径提供；不会对外提供归档中其他 ASR/任意文件。最小 rig/retarget/stabilization 复用附 MIT NOTICE，角色署名为 Godette (Rigged), zahlenmaler / CC BY 4.0。

## 验证记录

| 验证 | 结果 | 证据范围 |
|---|---|---|
| Python 全量 `pytest tests -q` | **1067 passed, 25 skipped, 15 warnings**，66.91 s | `.venv-libras`；跳过不计通过；不是所有 GPU/人工验收 |
| CUDA 环境专项 | **64 passed**，1 既有弃用 warning，6.15 s | `.conda-env`，新模块+API+桌面回归 |
| Node 播放/控制/异步 UI | **10 passed** | 时钟、相机、无 RAF 控制、迟到计划/动作、编辑、原始证据 |
| 真实动作归档 | **100 注册；83 批准摘要匹配** | 新校验实际读取归档，不只 synthetic fixture |
| 真实浏览器 | **已验证** | 完整片段播放；2.4 s 定位；0.5×；暂停/继续/重播；显示对照；侧面/近景；整句未覆盖阻止 |
| 真实视频 API | **HTTP 200** | 下节限定的单个 Uni-Sign 文件请求 |
| 浏览器视频 UI | **已验证** | 选本地 MP4→识别→原始候选保留→不自动播放→整句未覆盖阻止→手动编辑后预览 |
| Qt 显式打开 | **自动测试通过** | openUrl 接收本机草稿，无自动播放；真人手工 Qt 点击验收 NOT_RUN |
| wheel | **构建并核对静态资源** | 包内包含 15 个 animation 文件；ZIP/大模型不入包 |
| `.conda-env` 依赖检查 | **pip check 无损坏依赖** | Torch 保持 `2.9.1+cu128`、CUDA 可用；补 API extras |
| 独立代码复核 | **两项 P2 修复；无遗留 P1/P2** | 复核者重跑 registry/API 21 项、Node 8 项；随后增加 2 项 UI 回归 |

独立审查没有运行 GPU 或替代人工语言验收。报告中的测试数不可称产品准确率。

## 真实识别及负对照

- 输入：`G:/Archive/Zenodo/CE-CSL/CE-CSL/video/dev/A/dev-00015.mp4`。
- 模型：原有 `models/external/uni_sign/model_bundle`，`uni_sign_pose_slt`，CUDA bfloat16，generation 1；stage1 权重摘要 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`。
- 工程 API 请求 `animation-live-20261028` 返回 HTTP 200、文本“妈妈不要多喝汤。”，客户端计时 11.4668 s。仅一条工程请求；不是准确率、P95、热态均值或摄像头延迟。模型训练域 CSL-Daily，样例来自 CE-CSL，不能外推同域成绩。
- 在浏览器再次选择上述 MP4，现有接口返回同一文本；页面显示待确认且没有自动播放。点击确认因没有整句资源而被阻止。
- “认识你很高兴不要”负对照同样被整句阻止；自动测试另覆盖未知数字、前缀否定、歧义、过期计划/动作、改姿态继承审核等。
- 手动将输入改为库内“认识你很高兴”可预览，但原始“妈妈不要多喝汤。”保留；这个操作证明编辑/证据机制，不能称为该识别句的正确手语生成。

## 问题与修复

1. 加载中 Stop 被禁用：独立 DOM stub 在计划/动作响应延迟时复现两项 RED。请求期间启用 Stop，abort 与代次校验共同阻止迟到启动；两项 GREEN。
2. 旧审核仅比 source_sha256：对合法新包改姿态/拟合仍继承批准。绑定批准 ID、fit 与规范化姿态摘要；两项 RED→GREEN，原包 83 段仍匹配。
3. 仅靠 RAF 发布定位状态：在后台或被节流时，时间轴/画面滞后。控制操作同步 render/onTick；新增无 RAF 回归。同时先读取用户定位值再暂停，避免暂停回调覆盖滑杆输入。停止时同时清除旧轨引用，防止切换显示重新显示旧片段姿态；浏览器实测 slider=2.4 s、时间=0:02/0:08。
4. 隔离工作区缺少 gitignored `configs/model_defaults.json` 导致最初全测 4 项失败。定位为本地配置缺失，复制主目录同一配置后通过；没有改模型默认值或修改历史测试掩盖失败。

## 本机证据与重跑

运行产物保留在工作区 `artifacts/animation-acceptance/`（Git 忽略）：

- `full-tests.log`：最初 4 failed/1059 passed/26 skipped，保留以追踪问题。
- `full-tests-final.log`：1067 passed/25 skipped。
- `conda-tests.log`：CUDA Python 环境 64 项回归。
- `real-recognition.json`：请求源路径、原始响应、单次计时与无质量主张。
- `optimized.jpg`、`original.jpg`：相同模型、片段与 2.4 s 的优化/原始显示对照；`recognition-blocked.jpg` 为整句未覆盖界面。
- `wheel.log`、`wheel/`：构建产物；不会承诺这是分发版。
- `catalog.json`：显式指向既有 stage1 模型的本地配置。

```powershell
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -m pytest tests -q
node --test tests/animation_player.test.mjs tests/animation_stage.test.mjs tests/animation_workspace.test.mjs
& E:/githubproject/SignLanguage-Bridge/.conda-env/python.exe -m pytest tests/test_animation_registry.py tests/test_animation_api.py tests/test_animation_handoff.py tests/test_unified_api.py tests/test_desktop.py -q
```

复核屏幕时使用可见浏览器窗口；隐藏 IAB 的截图可能滞后于当前 DOM/画布，不能把旧截图当作当前帧。

## 未实现与未测

CSL 使用者理解任务、动作自然度/画面偏好人评、100 段逐段视觉验收、面部非手控语法、连续片段合成、空间指代、IK/接触/碰撞、摄像头端到端动画、跨平台浏览器、QtWebEngine 内嵌、所有模型/语种组合与稳定 FPS/P95 均未验收。

本轮不用 Blender 也能看见界面/画面改进；新人物、手部/服饰网格、蒙皮与表情需要后续 Blender 离线制作或修补。没有新建 `.blend`、烘焙新动作或导出新角色。自由文本 SLG、SOKE 扩展和大规模视频到 SMPL-X 流水线仍是后续阶段。

