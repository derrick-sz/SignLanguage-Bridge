# 动画动作与人物画面升级（2026-11-06）

后续用户反馈仍有抽搐，已对全部 100 段补充时间连续性诊断，见[残留抽搐与语料扩充核查](animation-jitter-and-corpus-expansion-2026-11-06.md)。本报告的测试、导出一致性和播放结果不构成“动作无抽搐”验收。

## 范围与身份

用户要求先制定计划再执行，完成第一轮可运行工程改进。工作区为 C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge，分支 codex/csl-animation-workspace，修改前 HEAD 为 86e0d45cd371d2db6db2a0c522021c37277d9828。保留原有未提交资料；未提交、推送、合并或改变模型默认值。

- [设计](../superpowers/specs/2026-11-06-animation-quality-upgrade-design.md)
- [实施计划](../superpowers/plans/2026-11-06-animation-quality-upgrade.md)
- [操作说明](../csl-animation-workspace.md)
- 本机工程预览：[升级工作台](http://127.0.0.1:8875/animation/)（独立动画服务，无识别模型加载）。

## 已实现

1. 保留原动作，以源姿态 FK 和目标肢体比例建立手腕目标，用双节 IK 修正位置，保持骨长与手腕世界方向。手指仅进行原有的最多约 3° 尖峰修正。原版与派生轨独立，不覆盖源 JSON。
2. 用 Blender 4.5.9 制作原授权 Godette 的改进版：调整手部和衣服材质，移除背包、领带及前侧背带等独立附件，共 673 个顶点。保留原人物和署名；没有重新雕刻成人比例或重做完整手指网格。
3. 角色映射和有限轴修正通过制作包 rig_profile 配置；人物文件、配置和动作源摘要分别校验，不继承原资产质量批准。
4. 原始/制作修正版使用同一个播放时钟和视角进行分屏；取景包含所有末节手指与深度，窄画幅上下对照。可显示经归一化的源骨架，并遮住字幕。
5. 增加眉部、实际眼睑变形骨骼、下颌的手工绑定检查。缺少源轨时保持中立，新片段清空手工值，没有从中文推断语法表情。
6. 五段源资源完成真实离线烘焙，可从工作台下载 GLB；当前检查可导出 JSON，含动作、人物、处理版本、时间和未审核状态。

## 五段固定试点

来自原有固定 ZIP，不是新录制视频或新增认证词库。参考组共享源 bundle 摘要，必须同时记录 asset_id。

| asset_id | 显示文本 | 帧数 / FPS | 源资源 SHA256 |
|---|---|---|---|
| new-1918b4f42eac7d78-cap-0009 | 认识你很高兴 | 200 / 25 | 02e3cbfaceca600f560b83ccfa5787590ab19ca19a4429fe7336db65e91547af |
| pilot-greeting-answer-002 | 是啊，你过得好不好？ | 174 / 25 | 0087d5024d320ea2564640f538e4cc7a8ff06f36898f9875c32d9add94ed8a6f |
| pilot-greeting-how-004 | 很好，你呢？ | 153 / 25 | 同上，与 asset_id 共同识别 |
| pilot-greeting-busy-005 | 工作忙吗？ | 148 / 25 | 同上，与 asset_id 共同识别 |
| pilot-hospital-pay-013 | 先交费，然后去取药。 | 163 / 25 | 同上，与 asset_id 共同识别 |

## 实际验证

| 检查 | 本轮结果 | 范围 |
|---|---|---|
| Node 动画测试 | 21 passed | FK/IK、面部轨边界、深度/对照取景、加载中操作及既有停止/定位/过期请求回归 |
| Python 全量 | 1079 passed / 25 skipped / 15 warnings，71.00 s | .venv-libras 全量实际运行；跳过不计通过，warnings 是已有依赖弃用提示 |
| 制作包专项 | 12 passed | 路径/摘要/源绑定/快照/可配置骨架、片段内部身份和审核状态及路由 |
| Blender 制作 | 完成 1 个派生人物与 5 个动画 GLB | 实际运行官方便携 Blender |
| 五段几何不变量 | 838 帧；骨长变化 0；手指平滑修正不超过约 3°；不可达目标 0 | 派生目标与约束检查，不是对真人手形或含义的准确率 |
| 导出一致性 | 838/838 帧检查；最大关节位置差 2.6052075537554223e-06 模型坐标单位 | GLB 与同一目标角色运行轨一致，不等于语言正确 |
| 浏览器 | 五段均播放至“已结束”；同步对照、源骨架、面部试调、0.5x/1x、2.4 s 定位、停止和 360 px 窄屏通过 | browser-pilots.json；窄屏上下对照，scrollWidth 345 px；实际下载检查 JSON 与动画 GLB |
| 原 ZIP | SHA256 仍为 4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5 | 原始资源未修改 |

手腕相对派生目标的平均偏差由约 0.01165–0.01966 降至约 6.67e-8–1.02e-7 模型坐标单位。目标由源 FK、比例和肩部锚点计算，不能当作真人真值误差、语义准确率或语言提升幅度。

浏览器对未覆盖的完整输入“认识你很高兴不要”显示“没有完整表达资源；未省略或猜测输入内容。”并保持 0:00 / 0:00。实际下载的 GLB 与制作包摘要一致，内部记录动作 ID、源摘要、人物摘要和处理版本；检查 JSON 保留 language_review=NOT_RUN。停止后切换人物和对照没有恢复旧片段。上述属于工程交互核验，没有人工语言理解评分。

## 发现并修复的问题

- 原 GLB 默认节点姿态与 skin inverse-bind 对应的绑定姿态不同。首次按默认节点矩阵烘焙，导出关节产生约 0.6 模型单位偏移。修复为先制作目标角色，再针对目标编译，以 inverse-bind 计算轴转换，逐帧验证导出。Blender 从第 1 帧开始还会偏移 0.04 s，现已统一从第 0 帧烘焙。
- glTF 四元数有 float32 舍入；未归一化便调用 angleTo 会把约 3° 误记成 3.00083°。检查先归一化，没有放宽修正幅度。
- 原眼睑编辑器控制骨骼无有效蒙皮权重，转动它们不产生可见变形。本轮按实际加权变形骨骼实现眼睑检查；骨骼名或编辑器控制器存在不等于 GLB 可驱动。
- 独立审查发现并修复 3 项 P2 问题：人物异步加载期间操作会访问未初始化场景；未知动作缺少源摘要或交换导出文件可能绕过身份检查；制作包逐段审核状态可能继承且暴露额外元数据。均先复现失败再修复，通过真实 Stage 回归、强制源身份和 GLB 内部绑定、目录字段白名单与逐段审核状态重置覆盖。

## 本机产物与复核

artifacts/animation-upgrade/ 为忽略目录，二进制未加入 Git：

- presentation/manifest.json、avatar-clean.glb、五个以 asset_id 命名的动画 GLB。
- presentation/geometry-report.json、export-verification.json、production-report.json。
- node-tests.log、python-full.log、build.log、export-verification.log。
- browser-pilots.json、comparison.png、source-skeleton.png、narrow-comparison.png、face-controls.png。
- downloaded-review.json、downloaded-animation.glb：浏览器真实下载的核验副本。
- tools/blender-source.txt：官方 URL、校验源和 SHA256。

![原始与制作修正版同时间对照](../../artifacts/animation-upgrade/comparison.png)

Blender 为官方 download.blender.org 的 4.5.9 Windows 便携 ZIP，发布摘要为 41da973b9bf95bb312cbeff4d1982feb13259b43c821686b9bafea4dfe5477cf。没有全局安装或改变现有 Python/Torch 模型环境。

## 边界与实施选择

当前完成角色制作、位置约束、面部控制与导出一致性。原归档不含对应真人源视频，未做真人三层同步；没有逐段人工修指形、通用接触/碰撞保证、语义表情轨、自由文本连续 CSL、真人理解测试或审美批准。手腕目标一致不等于手掌/指尖接触正确。物理摄像头、稳定 FPS/P95 和 GPU 识别链路仍为 NOT_RUN。

采用现有授权人物派生版是本轮实施选择；若其比例或造型仍不符合展示要求，下一轮可沿同一人物配置和导出接口接入新角色，新人物不继承旧批准。
