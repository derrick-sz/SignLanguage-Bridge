# 沟通页面切换 Uni-Sign 与实时模型选择

日期：2026-11-09（Asia/Taipei）。用户在本轮明确授权模型切换。实现使用现有动画工作树；没有提交、推送、修改权重或启用LLM融合。

## 当前实际状态

- 8765中文路由已从`csl-tfnet`切换为`unisign_pose_stage1`，实际`model_id=uni-sign-csl-daily-pose-only-182913ca`、`task_type=text`。权重为既有CSL-Daily基础版，SHA256 `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c`；未把S2升级为生产默认。
- 实际翻译运行时为PyTorch CUDA／bfloat16；姿态检测和估计均包含CUDA provider、无回退、无warmup error。Libras继续使用`libras-rf-pilot-20261009`，CPU分类。
- 本机profile使用`.conda-env/python.exe`、`device=auto`、启动等待180秒。启动器读取`recognition.csl_model/libras_model/device/startup_timeout_seconds`；旧profile缺少新字段时保留TFNet/CPU兼容值，迁移必须显式填写。
- 8879通过原有v3协商接受text任务，原始结果继续为`unverified`。本轮没有修改网关、前端录制、动画或会议页面实现。

## 验证

原始证据：`E:/githubproject/SignLanguage-Bridge/artifacts/unisign-product-switch-20261109/`。

| 检查 | 本轮结果 | 证据 |
|---|---|---|
| 真实健康状态、模型身份及CUDA | 通过；Uni-Sign/text，模型及姿态CUDA，Libras ready | `health-after.json`、`negotiation-after.json` |
| 固定Pilot-50首条已裁剪CSL-Daily视频`S003002_P0000_T00.mp4` | 8765直连与8879网关原文一致；返回“他差点儿喝了酒,被罚款了。” | `direct-mp4.json`、`gateway-mp4.json` |
| 数据集视频编码为WebM再上传 | 通过；显式`transcoded=true`、输出text，本例原文也相同 | `recording-format.webm`、`gateway-webm.json` |
| Libras Sim真实视频 | 通过；返回“是 / 是的”，仍为12词分类草案 | `gateway-libras.json` |
| 模型默认策略与Uni-Sign adapter回归 | **42 passed** | `.conda-env/python.exe -m pytest -q tests/test_model_defaults.py tests/test_uni_sign_adapter.py` |
| 网关整句、契约、取消、忙状态定向回归 | **8 passed / 9 deselected** | `gateway-regression.log` |
| 启动器所有权保护 | 修复前2通过/4失败，修复后**6通过/0失败**；全部进程操作mock | `ownership-red.log`、`launcher-regression-final.json`；仓库`tests/communication_launcher.test.ps1` |
| 实际启动器复用与错误别名前缀拒绝 | 通过；正确profile复用识别55440/产品38516，错误`unisign_pose`拒绝且两端PID未变 | `launcher-negative.json`、本机`runtime-state.json` |

样本用于核验路由、格式和直连/网关一致性，**不代表语义正确或产品准确率通过**。MP4网关该次3.57秒、WebM该次4.35秒只是共享机器上的单次完整片段HTTP耗时，不是实时视频流延迟或SLA。验证脚本不读取参考答案、不把原文送入任何LLM。

## 运行与审查发现

1. 首次8766影子加载在mT5初始化时因Windows提交内存不足失败，尝试分配768,344,064字节；当时TFNet进程私有提交约7.05GB、系统提交占90%。释放身份和空闲状态均已核验的旧识别进程后，顺序加载Uni-Sign成功。没有更改pagefile或停止其他任务。模型warmup总记录约76秒，原30秒启动等待不适用本机当前负载。
2. 停止旧进程后首次立即启动被尚未消失的监听记录拒绝；确认旧进程/监听退出后重试成功，未放宽所有权检查。
3. 首次网关验证期间8879被并行工作独立重启（556→38516），产生一次连接拒绝；待服务恢复后重跑完整HTTP验证通过。本轮未发起该产品重启，不能声称整个期间产品PID保持。
4. 初次用`D:/anaconda3/python.exe`运行网关单测因旧Starlette与HTTPX的TestClient接口不兼容出现7失败/1通过；换到本次服务的`.conda-env`后定向8项通过，未升级依赖或为此修改产品逻辑。
5. 独立审查复现两个启动器问题：子串匹配会误接收别名前缀或别处出现的device值；Stop/RestartProduct可能先停产品再发现识别配置不符。已改成完整参数及相邻值匹配，并在任何停止前预核验所有受影响的服务；6个隔离回归场景由红转绿，复核无剩余可操作问题。

## 实时模型选择

在本地已有模型中，**SLRT Online更匹配逐帧增量词识别**：已有`SLRTBackend.new_stream()`、滑窗解码及partial/final生命周期；Uni-Sign是完整上下文整句翻译，反复切窗不能变成原生流式。官方方法也明确采用滑窗在线识别，并需额外Gloss-to-Text网络才能在线翻译整句：<https://github.com/FangyunWei/SLRT/tree/main/Online>。

但是本地SLRT共享v3 adapter当前仍只声明clip；网页逐帧会话尚未接通。2026-10-24历史30FPS回放曾11/12条无final，受姿态与逐帧推理吞吐/丢帧影响；见E盘主检出的`docs/reports/slrt-dataset-eval-2026-10-24.md`。该旧结果不作为本轮现状实测，本轮没有重新加载SLRT或宣称实时已达标。

建议职责：SLRT负责增量Gloss；句末Uni-Sign可独立给出整句候选，保留两路原始结果；LLM整理与仲裁仍按P-1 G2条件另行验证，不默认自动融合。

**NOT_RUN：** 物理摄像头及真实浏览器录制、用户“我想睡觉”原录制复现、新签者识别语义质量、SLRT本轮实时吞吐/首稳时延、浏览器端到端、全仓库/CI、S2对比和LLM融合。此次完成的是模型切换与HTTP工程验证。
