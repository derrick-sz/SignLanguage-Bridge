# Godette v13 GLB 导出与验证

## 交付与身份

主文件：`godette-reference-v13.glb`

SHA256：`2d89e2e33271d2e06c5a834a56675663aa5551814e2c98ef86fc084ca0c151a8`

冻结原生模型 SHA256：`f03f7ebecb892ac2b01a0768d3c140802cf06e207c4bad33bd0aa4f0b92822b4`

以已验证 v12 GLB 为保真基底，逐字段导出冻结 v13 的变更。未改写冻结 Blend、已有 v12 文件或项目源码，未写入 GitHub。不是从展示站姿副本导出。

## 导出保真：通过

- 74 个蒙皮网格、78 个 primitive、227 骨、0 动画。
- 227 骨名称、顺序、层级和原生 rest 不变，GLB inverse-bind matrices 逐字节保持。
- 16 个中性唇骨节点与 v12 完全一致。仅 13 个下身默认 pose 节点改变，对应冻结原生模型有意清零的 Hip、Leg、Knee、Foot、Toe 通道。没有把新的 pose 当作新的绑定 rest。
- 改动 7 个衣装网格的 POSITION/NORMAL，以及连续裤装的 JOINTS/WEIGHTS。裤装保留 224 个不大于 0.0001 的正权重，每点最多 4 个影响，float32 归一化最大改变量 5.96e-8。其余权重与所有 UV、索引、材质定义、贴图及嵌入图片字节保持。
- 所有 78 个 primitive、300,447 个导出顶点与冻结原生默认蒙皮的最大世界坐标误差 1.218 微米，小于 3 微米阈值。
- 真实 Blender 4.3.2 GLB 回导后，74 网格与原生评估表面双向最近位置最大误差 1.468 微米，小于 4 微米阈值。允许 glTF 法线/UV 引起的位置重复点。
- 裤体单连通、三个边界环为 140/64/64，内部非流形边及退化三角均为 0。两个已存在的封闭鞋底仍为单连通闭壳，Euler=2，正有向体积。
- 错误候选摘要与覆写已有输出均被拒绝，测试后模型摘要不变。

`previews/` 中正面、侧面、脸部均为实际导入本 GLB 后渲染。灯光与相机来自审查场景，模型在导入前清空再从 GLB 载入；图片已目视检查。导入器创建的隐藏骨骼显示辅助体不计入 74 个模型网格。

## 口型验证：隔离兼容探针通过，项目尚未接入

在独立的 test-only runtime 副本中，用真实 v13 SHA 替换 v12 的身份常量。实际 GLTFLoader 与 Three r169/r170 各跑 101 档协调手动口型、倒序重复、表达式复位和 rig 复位，均通过。8,308 个保留皮肤三角形没有相对法线翻转；最小面积比 0.8709，最大面积比 2.8191，最大边长比 1.6557；下牙下降 3.107 mm，上牙、非脸网格和复位误差均为 0。

这不代表现有项目已经支持 v13：现有项目尚未识别 v13 的协调口型身份，自动 jaw gate 仍关闭。但它对未知模型保留通用手动 jaw-only 回退，实际 `jawOpen=1` 会只旋转 `Jaw_4` 约 -0.16 rad。因此不能把 `manualMouthAvailable=false` 解释成嘴部完全不会动。正式接入必须显式加入 v13 的身份与协调口型保护，不可伪报成 v12 SHA。详见 `runtime_probe/production-gate.json`，该项目接入检查明确为未通过。

Three.js 探针使用真实几何与 CPU 蒙皮，纹理解码被 stub；不声称浏览器/WebGL/GPU 视觉验收。

## 六段动作：未重跑

当前工作区只有历史六段脚本及报告，没有六个原始动作 JSON。缺失清单及期望摘要见 `six-clip-availability.json`。历史结果没有计入本轮通过项。v13 有意改变 13 个下身默认 pose，未来恢复动作后应按新基线解释差异，不能直接沿用旧版“全骨世界矩阵零差”断言。

本轮不声称完成六段动作、任意动作无碰撞、手语语义质量、手机性能、真实服务发布或全库验收。原生裤体压力测试见原模型交付目录，本报告不把其结果冒充 GLB 运行时测试。

## 文件

- `export-contract.json`：冻结输入/输出摘要、实际修改范围、位置映射及 pose 分解误差。
- `artifact-check.json`：独立实际 GLB 字节与结构检查、裤/鞋底拓扑。
- `deformation-numerics.json`：全部 primitive 蒙皮数值。
- `roundtrip-validation.json`：真实 Blender 回导误差。
- `runtime_probe/three169/`、`runtime_probe/three170/`：本轮实际口型数值与运行日志。
- `runtime_probe/production-gate.json`：现有项目对 v13 的真实门控及回退行为。
- `negative-gate-tests.json`：摘要与覆写拒绝测试。
- `DELIVERY-MANIFEST.json`：交付文件大小与 SHA256。

`preparation/`、NPZ、运行时节点快照和失败探针日志为可追溯的本地中间资料。请按交付清单打包，勿把 test-only runtime 当作已集成项目代码发布。
