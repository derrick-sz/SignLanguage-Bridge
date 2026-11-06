# 下颌数据驱动候选与几何校准（2026-11-07）

已把全库条目中真实的 SMPL-X joint 22 接到经过校准的 Godette 制作角色，默认播放有界嘴部候选，可同帧关闭对照。输入姿态保留，不按文本、音素或随机值张嘴；expr10/50 未直接映射到唇骨。原100条无该显式协议，自动画面与烘焙文件不变。已校准角色的手动下颌也改为正确开口方向，并联动下唇。

这是一维开口的研究候选，不是完整嘴形、语言级非手控表达或逐字口型同步。原片时间对齐、手语含义和视觉接受度仍未验证。本子任务没有启动/停止服务、操作用户当前tab、训练、重烘全库或提交代码。

## 实际几何证据

- 源：`G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz`，SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`。
- 目标：`G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/presentation/avatar-clean.glb`，SHA256 `b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c`。
- 可复现脚本：[calibrate_jaw.py](../../scripts/animation/calibrate_jaw.py)。数值报告：[calibration.json](../../artifacts/animation-upgrade/nonmanual/calibration.json)。脚本只读取源文件，输出独立报告；源 LBS 包含该旋转的 pose corrective，目标 LBS 使用真实 inverse bind、全部3个材质 primitive、15,153个顶点（含材质接缝重复）。

源模板中央内唇的 landmark45/49，在 neutral 时 signed Y gap 为 -0.0000174（两点距离约0.000402），jaw +X 0.08/0.16 rad 后变为0.007306/0.014453；-X导致上下交叉。因此本模板 +X 为开口，不能照搬负号。

目标 `Jaw_4` 有205个权重大于0.001的顶点；中央上/下唇骨分别有18/17个。`Jaw_4`、`Lips_Lower_13` 均是 `Head_129` 的孩子，**下唇不是下颌的子节点**。原 `rotateX(-amount*0.16)` 把下颌往上转；单独转 Jaw 对下唇只产生小幅部分变形。新校准让下唇骨组及其下牙子树绕同一下颌支点运动。

| 目标 +X / 下唇联动 | 中央唇加权质心 signed Y 间距 | 下唇质心 Y 位移 | 上唇质心位移 | 全网格最大位移 |
|---|---:|---:|---:|---:|
| 0 | 0.013249 | 约0 | 0 | 约0 |
| 0.08 rad | 0.021186 | -0.007937 | 0 | 0.008514 |
| 0.16 rad | 0.028929 | -0.015681 | 0 | 0.017015 |

单位为模型坐标单位。目标量是**受权重顶点的质心间距**，与源内唇 landmark 距离分开报告；不是唇缝最小距离、自然度或碰撞检测结果。扫描的所有顶点均有限；0–0.16 rad 为本轮保守显示范围，不据此声称任意表情组合无穿插。

## 输入与处理边界

`smplx-jaw-godette-calibrated-v1` 锁定上述两个摘要，只用于带明确 `clip.nonmanual_candidate` 的未烘焙条目。`CorpusStore.PLAYBACK_VERSION` 改为 `soke-full-lazy-gesture-jaw-v2`，派生身份包括下颌协议，旧缓存不会复用新动作。

候选从原始 joint22 的轴角旋转求绕 X 的 twist；其他方向不伪装为开口。源旋转范数超过0.35 rad时不直接驱动，负向归零，最大0.16 rad，三帧中值抑制孤立噪声，再限制相邻速度0.6 rad/s。整段预编译后按同一时钟线性采样，反向定位无状态累积。阈值、平滑和速度约束可能改变原始运动，应与源片继续审核。

全库按ID顺序前40段的便利抽样共2,163帧，逐文件SHA与数据库一致。jaw X中位数0.04085、95分位0.09792 rad，但最大3.12419 rad；相邻轴角差最大3.14740 rad。8帧超过0.35 rad，保留在源姿态中并记入派生诊断。候选最大0.16 rad，最大相邻变化0.020000000000000004 rad（30FPS舍入）；没有推广为全19,877段嘴部验收。证据：[sample-candidate-audit.json](../../artifacts/animation-upgrade/nonmanual/sample-candidate-audit.json)，源抽样及文件摘要在校准报告中。

手动下颌覆盖自动值，二者不相加；调眉/眼不会顺带覆盖下颌。恢复按钮释放手动覆盖，返回自动候选或中立。未知目标摘要、未知profile、模板摘要不匹配、已有烘焙导出或缺失显式协议时，不套用自动校准。新生成的 SignSparK 片段本子任务未擅自 opt-in。

含候选嘴部的 GLB 导出暂未实现，协议与诊断明确 `export_supported=false`。全库条目继续没有GLB下载入口；旧100条的烘焙下载不含手动表情检查。导出的review JSON包括启用/覆盖状态、有效角度、来源绑定和异常诊断。

## 验证与浏览器交接

- TDD先观察了缺模块、尖峰直接放大、缺校准rig接口、缺API协议以及眉/眼误传jawOpen的失败，再逐项修复。
- 动画 Node **38 passed、0 skipped**，包括显式真实GLB/Three路径下的实际骨架开口、20次重复应用不漂移及复位测试；正常无外部资产环境会跳过这一单独的可选真实资源用例。日志：[node-tests.log](../../artifacts/animation-upgrade/nonmanual/node-tests.log)。有一条Node对既有 `.js` ES模块自动识别的警告。
- Python相关API/全库/制作包/registry **54 passed**，1条既有Starlette弃用警告。日志：[python-tests.log](../../artifacts/animation-upgrade/nonmanual/python-tests.log)。先前误用系统Python的Starlette/httpx版本不兼容不计为功能失败，正式结果使用项目 `.venv-libras`。
- 真实100条制作包 + 全库的独立TestClient探针：`S000000_P0000_T00`正常规划并按摘要取回新协议，静态模块200，旧100条无nonmanual profile；未启动监听服务。[api-probe.json](../../artifacts/animation-upgrade/nonmanual/api-probe.json)。
- 当前运行的旧8876进程不会自动加载Python静态allowlist变更；父任务应先启动新版本8877验证，再统一切换正式入口。不能把旧进程刷新失败误当成候选数值失败。

浏览器验收步骤（本子任务 **NOT_RUN**）：

1. 新服务选择“全量语料”，按ID搜索 `S000000_P0008_T00`，确认播放，暂停定位 **1.43333秒（第43帧，零基）**。该帧原jaw `[0.17409775,0.00277764,-0.02892332]`，范数0.17651<0.35，相邻帧有效；候选0.16 rad。证据：[browser-probe-frame.json](../../artifacts/animation-upgrade/nonmanual/browser-probe-frame.json)。
2. 近景/原始修订同帧对照，打开面部检查，关闭/开启“源数据嘴部候选”，核对下唇和下牙下降，时间不变；观察是否产生明显网格穿插。
3. 调眉/眼应保留自动嘴部；手动下颌0应闭回中立、0.75应为0.12 rad，恢复按钮回到自动。检查review JSON的manual_override和角度；无新GLB下载宣称。
4. 停止后切换显示不能复活动作；选择旧100条，候选开关不可用、自动嘴部仍中立。窄屏确认新增控件可用且不改变画面取景。

只有实际完成上述浏览器检查后，才能另记视觉证据；程序测试和蒙皮数值不能替代语言、美术或原片时间对齐验收。

## 父任务实际浏览器核验

2026-11-07 已在更新后的正式本地入口 `http://127.0.0.1:8876/animation/` 完成单条功能核验；服务使用原制作包与全量语料，未重烘旧100条。临时8877服务核验后已停止，8876保留供查看。

`S000000_P0008_T00` 在1.43秒侧面观察，开启候选时下颌/下唇下降，关闭时回到中立；该帧未见明显嘴部网格穿插，但不推广为全库碰撞或美术验收。浏览器导出的5份review均为同一时刻：自动值0.1598766893 rad，调眉0.2/眼睑0.05后仍为同一自动值，手动下颌0覆盖为0 rad，手动0.75覆盖为0.12 rad，恢复按钮释放覆盖并回到自动值。停止后嘴部候选控件禁用；旧100条“谢谢你，医生。”播放时没有自动候选，开关禁用。最后恢复全库“你们好！”自动候选供查看。浏览器警告与错误日志0条。

证据：[browser-verification.json](../../artifacts/animation-upgrade/nonmanual/browser-verification.json)、[自动嘴部画面](../../artifacts/animation-upgrade/nonmanual/preview-jaw-candidate.png)、[同帧关闭画面](../../artifacts/animation-upgrade/nonmanual/preview-jaw-disabled.png)，以及同目录`review-*.json`。本轮未单独进行窄屏验收；语言、完整嘴形、原片时间对齐及全库穿插验收仍为NOT_RUN，含此嘴部轨的GLB导出仍不支持。
