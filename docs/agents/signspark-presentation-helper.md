# SignSparK 生成候选的展示包接口

模块：`slbridge.animation.signspark_presentation`。不加载模型、不训练、不运行GPU、不改服务、不复制旧烘焙动画。调用者负责先完成原生 SignSparK codec，`generation_codec` 必须为 `signspark-row6d-flat-hand-v1`，jaw 已在55关节的22号，保留 `expression_channels_retained=50`。不能先加SOKE hand mean。

## 调用顺序

```python
from pathlib import Path
from slbridge.animation.signspark_presentation import (
    verify_presentation_inputs, opt_in_generated_jaw, build_avatar_presentation,
)

# 在模型构造/推理之前预检真实人物、配置和模板；不创建输出。
inputs = verify_presentation_inputs(
    smplx_model=Path("G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz"),
)

# 在写clip JSON、生成归档和计算最终摘要之前调用；返回deepcopy，不改输入。
clip = opt_in_generated_jaw(
    clip,
    smplx_model=Path("G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz"),
)

# 调用者把最终clips写入候选archive，创建MotionRegistry并取得最终assets。
# 必须使用最终archive中的资源摘要，而非opt-in前的clip摘要。
manifest = build_avatar_presentation(
    run_output / "presentation",              # 必须是尚不存在的新目录
    assets=registry.catalog()["assets"],      # 可省略，此时无pilots
)
```

二者均可显式传 `source_presentation=Path(...)`，但无忽略摘要或信任任意摘要参数。默认路径是已校验的 `G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/presentation`；换路径时仍须匹配完全相同的身份。源文件原地不写入。

## 实际校验与结果

- 来源manifest SHA256固定为 `a9fa82f145861c731d6aadc63957df549e7008fb701cb282a5f8dcd7143d80c8`，同时认证所复用的 `rig_profile` 与 `attribution`。重新制作角色/配置须重新校准，不能仅改manifest的自报摘要。
- 人物必须是 `avatar-clean.glb`，SHA256 `b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c`；opt-in另实读模板并流式校验SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`。
- `opt_in_generated_jaw` 先验证生成协议、有限55关节姿态和源身份，再返回独立clip：新增 `template_sha256`、`nonmanual_candidate=JAW_CANDIDATE`、`presentation_status="research_candidate"`、`export_supported=false`；动作/语言/表情审核重置unverified，human_approved/training_ready为false。
- `poses`、`refined_poses`及已有expression数组不作任何数值变换。`unmapped_target_channels`把 `jaw` 改为仍未使用的 `jaw_swing`；`expression`必须继续存在，50维表情只是保留，不直接驱动唇骨。实际嘴部候选仍由前端的有界X开口投影生成，语法与口型未认证。
- `build_avatar_presentation`只读取源manifest和avatar，完全不读取/复制旧pilots及其导出文件。输出 `files` 只有人物GLB；新pilots只从调用者给出的新assets取 id/sha256/text，无 `export_file`，不继承审核。
- 所有来源校验先于发布。数据写到同级唯一临时目录，用真实 `PresentationBundle` 验证后重命名发布；新目录只有 `manifest.json` 和 `avatar-clean.glb`。已有输出拒绝覆盖；坏pilot或摘要不符不发布成最终目录。
- Manifest处理版本为 `signspark-avatar-jaw-candidate-v1`。它是人物展示包，**没有含生成动作或新嘴部轨的可下载GLB**。实际动作仍由候选archive与播放器驱动。

## 验证

`tests/test_signspark_presentation.py`：13项（含1项真实素材集成），常规12项使用夹具，仅在测试内注入可信固定摘要。生产接口不提供摘要绕过。覆盖人物替换、来源配置替换、连同自报SHA一起替换、模板不符、错误codec/表情维度、旧输出保护、坏pilot不发布、无旧烘焙、输入不变、joint22及flat-hand保持、PresentationBundle消费。

```powershell
$env:SIGNBRIDGE_REAL_PRESENTATION_TEST='1'
& E:/githubproject/SignLanguage-Bridge/.venv-libras/Scripts/python.exe -B -m pytest tests/test_signspark_presentation.py tests/test_animation_presentation.py tests/test_animation_nonmanual.py -q
```

2026-11-07实测 **29 passed**，包括真实已校验人物/模板，1条既有Starlette弃用警告。未设置环境变量时真实素材用例跳过，其他夹具测试正常运行。父任务已整合worker preflight、clip显式opt-in及最终avatar-only发布，集成41tests通过；增量审查的jaw_swing标记已修复，变更后14tests通过。此结果不是真实SignSparK GPU生成或生成动作的浏览器验收。
