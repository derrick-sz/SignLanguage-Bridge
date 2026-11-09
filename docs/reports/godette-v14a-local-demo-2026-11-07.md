# Godette v14A Windows 本地实际演示

当前统一演示已转到 **8879**；跨会话整合、最新入口和新组合句质量限制见[当前总表](animation-consolidated-status-2026-11-08.md)。以下保留本轮/历史证据，不以旧端口状态代表当前服务。

日期：2026-11-07（Asia/Taipei）。本轮完成真实原动作 ZIP → MotionRegistry → v14A presentation → CLI → Windows Chrome WebGL 的本地运行，并连接既有全量语料库。演示服务保持运行，浏览器两个页面已保留。

后续用户反馈普遍手形退步与穿模，已进入[手形与穿模调试](godette-handshape-clearance-2026-11-07.md)。下列记录保留当时的加载/播放证据；整体动作质量仍未通过验收，当前本地代码包含后续候选修订。

## 直接使用

- v14A 新造型与完整语料：[http://127.0.0.1:8874/animation/](http://127.0.0.1:8874/animation/)。当前选中 `S000567_P0000_T00`“最近我觉得身体不舒服，你带我去医院吧。”，暂停在 1.20 秒。点击“继续”或“重播”观看；“本轮片段对照”可选择六个原有示例。
- PR 两条已生成候选：[http://127.0.0.1:8878/animation/](http://127.0.0.1:8878/animation/)。当前选中“今天我在家休息。”，暂停在 1.20 秒，已显式打开生成下颌候选。这里沿用该生成包原有的校准角色 `b42c21da…`，与 8874 的 v14A 角色分开。本轮只播放已有产物，没有重跑生成模型。

本机证据和脚本目录：`E:/githubproject/SignLanguage-Bridge/artifacts/godette-v14a-demo-20261107/`。

```powershell
# 启动或检查已运行服务；不需要激活虚拟环境
& 'E:\githubproject\SignLanguage-Bridge\artifacts\godette-v14a-demo-20261107\start-demo.ps1'
& 'E:\githubproject\SignLanguage-Bridge\artifacts\godette-v14a-demo-20261107\start-generated-preview.ps1'

# 分别停止对应演示；脚本核对 PID 的命令行及 presentation 路径
& 'E:\githubproject\SignLanguage-Bridge\artifacts\godette-v14a-demo-20261107\stop-demo.ps1'
& 'E:\githubproject\SignLanguage-Bridge\artifacts\godette-v14a-demo-20261107\stop-generated-preview.ps1'
```

服务只监听 `127.0.0.1`。后台输出在 `server.stdout.log` / `server.stderr.log` 与 `generated.stdout.log` / `generated.stderr.log`。脚本依赖下面列出的本机工作区和外部数据路径，并非可迁移安装器。

## 本轮版本与资产

| 项目 | 实际身份 |
|---|---|
| GitHub PR | [#1](https://github.com/betterway614/SignLanguage-Bridge/pull/1)，查询时为 draft/open；head `ad3806f8e0ba2227231492860cfed33eb70347f9` |
| Godette 分支 | `origin/codex/godette-v14-reference-20261107`，重新 fetch 到 `e8477dcd5c53495c0f555ae55fa8a1d97213f9a2`，已用 Git 验证包含 PR head |
| 本地候选分支 | `codex/local-godette-v14a-demo-20261107`，在 Godette 基线应用测试包候选；本轮没有提交、推送或合并 PR |
| 实际代码位置 | `C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge` |
| Python | `E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe`，3.12.7，已有 FastAPI/Uvicorn/NumPy/SciPy 环境 |
| Node | 24.19.0；测试使用原动作 ZIP 中与浏览器相同的 Three.js/GLTFLoader |
| 原动作 ZIP | `E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip`，SHA256 `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`；100 段 |
| v14A GLB | `E:/githubproject/SignLanguage-Bridge/docs/godette-v14a-windows-test-kit/assets/godette-reference-v14a.glb`，SHA256 `535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e` |
| 新 presentation | 证据目录中的 `presentation/`，由真实 100 段 catalog 构建，六 pilot 的源动作摘要均匹配 |
| 完整语料 | `G:/Archive/Zenodo/SOKE/full-library-20261107`，19,877 converted_candidate / 775 incomplete_frames / 2 missing_3d_sequence，共 20,654 索引 |
| 两句生成包 | `G:/Archive/Zenodo/SignSparK/custom-csl-20261107-run02-preview-v2`，候选 ZIP SHA256 `df2b2773d7f50dd811aead5295d67328c8e032d5e4fbce44f8d33d317eeacbff`；本轮只读产物审计 PASS |

测试包 `PACKAGE-MANIFEST.json` 中六个文件均通过实际 SHA 核对。补丁在 Windows 全局 `core.autocrlf=true` 下产生 CRLF，且 `rig.js`、`pyproject.toml` 缺少冻结候选的一个末尾空行。先在临时目录复现并定位，再从已验证 `evidence.zip` 中恢复八个候选源码的完整原字节；八个文件均与 `candidate-manifest.json` 一致。未修改任何既定摘要、模型门控或原始测试包。这个交付补丁的字节差异仍需单独修订。

## 实际浏览器检查

| 范围 | 本轮观察与证据 |
|---|---|
| 六个 pilot | “认识你很高兴”“是啊，你过得好不好？”“很好，你呢？”“工作忙吗？”“先交费，然后去取药。”“我去开会汇报抽样调查落实的情况”均加载并实际渲染。首段自然播放至 8 秒结束并保持姿态；四个参考片段核对暂停到 1.5 秒、定位结尾和 Stop 归零 |
| cap-0067 | 分别定位 4.08、4.12、4.16 秒，再倒退至 4.12 秒；暂停值与请求值一致。原始角色与 v14A 同一时间轴对照截图为 `02-v14a-comparison.jpg`。保留 `source-quaternion-temporal-v1`，不把这次取帧写成全片质量通过 |
| v14A 面部 | 在 cap-0067 4.12 秒检查眉部 0.5、眼睑 0.6、下颌 1.0 的真实变形，并恢复三项至 0。自动下颌复选框始终 disabled/unchecked；没有复用旧校准强行启用 |
| 全库 | 搜索“医院”实际返回 177 条，HTTP 核对第二页 30 条；页面选择 `S000567_P0000_T00`，按需读取一条参数、播放至结束，并定位 1.20 秒。没有加载或逐条验收全库所有动作 |
| 生成候选 | 两条均在 8878 页面实际播放；1.20 秒同帧下颌开/关可用，第二条保持打开。页面显示“自定义生成候选”和跳变线索，未提升为语言通过 |
| 控制台/HTTP | 主页面及生成页面读取的 error/warn 日志为空；HTTP 主舞台 JS/GLB、六 pilot 和单条语料均 200。服务另有浏览器 favicon 404，不影响动画资源 |
| 检查记录导出 | 已触发当前语料的导出按钮；自动化下载等待未返回，未确认文件落盘，此项保持未验收 |

时间轴及面部 range 的精确值通过本地页面 CDP 设置控件并派发标准 `input` 事件，再核对页面状态和截图；播放、确认、暂停、停止、视角、选句和下颌开关使用页面实际按钮/控件。没有替换播放器内部时钟、姿态或人物对象。

## 自动验证

- Python：`tests/test_v14a_presentation.py`、`test_animation_presentation.py`、`test_animation_registry.py`、`test_animation_api.py`、`test_animation_corpus.py`，**70 passed / 2 skipped**。两个跳过项是 Windows 当前权限不支持 symlink 创建，未提升权限；包含真实 v14A 字节环境检查。日志 `python-tests.log`；既有 Starlette/AnyIO 弃用警告保留。
- Node：v14A 身份及真实 GLTFLoader/101 级嘴部蒙皮、相等静止四元数插值、源时序、几何、播放器、工作台、对照、全库及 nonmanual Stage，**45 passed / 0 skipped**。日志 `node-tests.log`；Node 的 experimental loader 提示保留。
- 真实 HTTP 检查：`http-smoke.json`，含服务实际返回的 v14A GLB SHA、100 段数量、六 pilot 与一条全库动作。构建证据 `build-evidence.json`。
- 生成产物只读一致性审计：`generated-artifact-audit.json`，状态 PASS/complete，原始流与候选仍保留既有身份；此结果不代表本轮重新生成。

## 验收范围

本轮证明上述本机版本和资产能实际构建、启动、检索、渲染并响应已检查的控制。手语句义、手形准确性、自然度、源视频逐帧对齐、完整非手控语法、全库逐段视觉审核、任意姿态衣服/牙齿/双手碰撞保证仍为 **NOT_RUN / 未验收**。v14A 仅有手动有界口型；源下颌自动轨不支持该新身份。任意文本在线生成和包含新嘴部轨的 GLB 导出未实现。

![v14A 全量语料实际画面](E:/githubproject/SignLanguage-Bridge/artifacts/godette-v14a-demo-20261107/final-v14a-demo.jpg)

![原始角色与 v14A 同帧对照](E:/githubproject/SignLanguage-Bridge/artifacts/godette-v14a-demo-20261107/02-v14a-comparison.jpg)
