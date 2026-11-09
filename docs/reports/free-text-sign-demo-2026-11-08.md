# 自由文字生成手语演示：实现与实测

日期：2026-11-08，Asia/Taipei。工作树：`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，分支 `codex/local-godette-v14a-demo-20261107`，基础提交 `e8477dcd5c53495c0f555ae55fa8a1d97213f9a2`。本轮未提交、推送或覆盖并发人物修订。

## 当前结果

已实现 **网页自由输入中文 → 真实 SignSparK 文本条件生成 → 产物核验 → 人物播放**。本机入口为 <http://127.0.0.1:8879/generate/>，`/` 和 `/scenes/` 导向此入口；旧整句库保留在 `/scenes/library`。

此前的基本场景目录与多条完整表达接续没有满足用户“自由输入文字实现演示”的目标。此次主流程直接提交生成任务，不要求输入命中已有整句，也不通过改字幕复用其他句子的动作。

当前支持1–200字符，用户选择4/6/8/10秒，分别为120/180/240/300帧；固定50步、seed123。换行和制表符在生成侧规范为空格，原始输入保留。时长来自用户选择，尚未从语义自动预测。单GPU串行执行，等待中的任务可取消。任务失败、取消或尚未核验完成时不能播放。

**这是工程链路通过，尚未证明任意中文都能得到含义正确、手形准确、自然连贯的中国手语。** 页面保留“手语含义尚未人工核对”和动作跳变提示。当前直接使用预训练文本条件生成器；词级组合尚未成为已验收主链路。

## 两条真实库外输入

检索范围包含语料索引全部20,654记录（含不完整记录）。以下两句完全相同匹配及去标点匹配均为0。

| 输入 | 真实任务ID | 状态与播放 | 提交至完成 |
|---|---|---|---|
| 明天早上我想去图书馆学习。 | `04527b2e1ed04d7dac9956f79fabbe17` | complete；浏览器播放结束，0:04 / 0:04 | 298.782秒 |
| 今天晚上我们一起做饭。 | `10835c5c29fa43839cbe72718033c264` | complete；浏览器播放结束，0:04 / 0:04 | 320.931秒 |

两次都从浏览器提交并播放。第二次页面的控制台error/warn记录为空。截图接口返回“Unable to capture screenshot”，未获得本轮成功页面截图；因此浏览器证据为实际操作与AX状态记录，不把旧失败截图当作成功截图。页面已保留为交付入口。

两次均生成120帧、30 FPS、55关节；原始与修订姿态均为`[120,55,3]`且全部有限。body/left/right/face原始流形状分别为`[120,60]`、`[120,90]`、`[120,90]`、`[120,56]`。checkpoint三个流各859张量strict加载，missing/unexpected为空；目标动作为null、观察关键帧为0，完全屏蔽观察量的负控差异为0。

保持seed、帧数与步数相同，两种文字得到不同动作SHA与不同数值：body/left/right/face原始流最大绝对差约1.2311/0.7242/0.6675/2.9468。最终姿态也不同且非静态。这证明请求生成了不同新动作，**不构成句义准确率证据**。两条仍都有跳变警告；语言、手形和自然度验收均为`NOT_RUN`。

约5分钟是本机这两次4秒演示的提交至完成耗时，包含加载、采样、打包和观察量负控；不能当作纯模型采样延迟、摄像头端到端延迟或其他时长的性能保证。

## Windows内存问题与修复

前几次真实尝试失败，原始失败目录保留。分别出现读取checkpoint的MemoryError、初始化导入失败，以及CUDA申请16 MiB失败但仍报告约11.77 GiB空闲。对应主机提交额度已接近68,403,527,680字节上限，不能简单归因于专用显存不足。

经用户明确选择，仅关闭旧8874/8878手语预览进程。Docker和WSL未关闭，也未修改分页文件或系统配置。随后启用独立执行配置：

- `--readonly-checkpoint-mmap`：固定PyTorch2.9.1内部接口和weights-only反序列化，NumPy只读文件映射；保留完整SHA/字节数/strict校验，不写入原始权重。
- `--cpu-offload`：真实CPU FP32文本编码按完整`<Chinese> `前缀缓存，再使用Accelerate逐层搬运FP32参数。真实非持久缓冲区保留，禁止安装后整模`.to(cuda)`。
- manifest记录`checkpoint_mapping=readonly-file-backed-v1`及`execution_profile=fp32-layer-offload-cached-text-v1`。CPU与CUDA文本编码未证明逐位一致，属于单独执行配置。

两次真实任务均在此配置成功；每流PyTorch峰值allocated约377–378 MB。此值不包含驱动、其他应用或主机提交量，不能据此承诺低配置机器可运行。生成子进程结束后释放其资源，服务继续保留核验后的播放结果。

失败任务保留为failed：`2b6a5cf3c1d44c6f8e132a43e4aefa75`、`660d562dfe4849cdacb8a7c2f262b8f0`、`fe56d806d8ed4e81b179b8a2a5d07243`。早期Playwright失败记录也保留，不与后续成功记录混写。

## 实现和验证范围

新增`text_generation_jobs.py`、`text_generation_routes.py`与`web/generate*`，接入本轮独立`scene_cli.py`。服务只向客户端返回任务状态和动作URL；模型与输出路径由服务端固定。发布前核对实际receipt的输入、帧数、步数、seed、来源、零目标观察、模型/源码身份及归档SHA，再通过不可变MotionRegistry提供动作。

输入变化、取消、较晚返回的任务和动作加载均有失效保护；完成后停止仍可重播。后台只有一个生成worker，最多8个非终态任务、32条内存状态。服务重启后不恢复内存任务表，磁盘产物仍在；长文本分段、持久任务恢复及延迟优化尚未实现。

| 本轮新鲜验证 | 结果 |
|---|---|
| GPU-free服务/HTTP/场景/两个CLI定向Python回归 | 82 passed；1条Starlette第三方弃用警告 |
| 真实生成runtime中的readonly/offload/generation/empty-loading小fixture | 48 passed |
| 补齐可选Accelerate测试依赖跳过条件后，offload重跑 | 12 passed（上行子集） |
| generate与scenes控制器Node回归 | 18 passed |
| 两条真实输出摘要、shape、有限值、原始流SHA、来源及差异审计 | PASS |
| 两条新句真实GPU生成及浏览器播放至结束 | PASS |
| 独立代码及资源修复静态审查 | 无新增可行动P1/P2 |
| 语言、手形、自然度、完整表情语法、全部中文泛化、相机、全仓库/CI | NOT_RUN |

新增只读映射fixture覆盖真实torch.save、只读存储、GC生命周期、非持久缓冲区和输入拒绝；offload fixture使用真实Accelerate CPU hooks及已知线性层输出，不用随机文本embedding替代。上述定向检查不覆盖全仓库。显示人物为既有人物v14a；生成产物的旧人物下颌候选协议不能当作v14a完整口型或面部语法验收。

## 本机证据与复核

以下证据位于功能工作树`artifacts/free-text-20261108/`，为本机产物，不随Git发布：

- `novel-inputs.json`：全索引库外核验。
- `accepted/job-{1,2}.json`、`generation-{1,2}.json`、`motion-{1,2}.json`：实际服务结果与receipt副本。
- `accepted/artifact-audit.json`、`audit_free_text.py`：只读产物审计及脚本。
- `accepted/browser-cua-audit.json`：两条播放结束、控制台及截图限制记录。
- `accepted/verification.json`：定向检查命令与结果。
- `browser/browser-audit.json`：早期失败的Playwright证据，保留原状态。
- `server-offload.log`、`server-offload.err.log`：当前服务日志。

完整真实生成目录为`G:/Archive/Zenodo/SignSparK/web-generation-20261108/<job_id>/`，包含job-state、worker日志、原始流、候选ZIP与generation-manifest。复核命令：

```powershell
& 'E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe' -X utf8 artifacts/free-text-20261108/audit_free_text.py
```

实现设计与任务见[设计](../superpowers/specs/2026-11-08-free-text-sign-demo-design.md)及[计划](../superpowers/plans/2026-11-08-free-text-sign-demo.md)。使用与资产配置见[动画工作台说明](../csl-animation-workspace.md)。

## 词级组合下一阶段

维持自由输入为产品入口，将词级动作作为生成质量改进：先核对连续句数据与孤立词标注的样本身份、裁剪时间基和动作参数映射，建立可回溯的词边界；经手语审核后构建高频词/短语动作锚点，再比较纯生成与锚点约束生成的含义、手形、衔接和非手控表达。不能以中文分词、2,000个Gloss标签或数值平滑替代这一步。

已准备部分SignSparK Retrieval资产，`csl_iso.train`可解析出1,997标签的区间标注，但模型权重和LMDB仍只下载部分，未接入当前功能，也未完成逐词动作语言审核。此项不得写成现有自由输入路径的词级组合成果。
