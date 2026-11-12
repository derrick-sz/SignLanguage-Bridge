# “谢谢”完整原生动作修订

日期：2026-11-09，Asia/Taipei。针对用户截图中“谢谢”只播放0.3秒、右手挡住鼻部的问题，已在本机产品替换选材并实际播放。此次修订解决来源片段过短和选材优先级问题；手语语义、自然度和通用穿模仍未通过人工验收。

## 原因与来源

旧词库先按标注时长接近中位数筛选候选，单独输入时取首候选，句中再按接缝角度选材。现有8份“谢谢”候选只有5或9帧，漏掉了完整单词原句。

| 项目 | 修订前 | 修订后 |
|---|---|---|
| 实际独立输入选材 | `S000122_P0000_T00`，源原句“谢谢医生。” | `S000003_P0000_T00`，源原句“谢谢！”、唯一Gloss为“谢谢” |
| 使用范围 | 原生配对LMDB的`[4,13)`，9帧 | 完整原生`[0,30)`；保留ISO标注`[4,17)` |
| 产品播放时长 | 30FPS下0.3秒 | 同一30FPS下1秒；慢放开关仍为原状态 |
| 单独输入策略 | 文字规划后选首候选 | 精确匹配完整原生词条，`selection_mode=exact_native_utterance` |
| 句中策略 | 接缝角度优先 | 有完整词条时先限定到完整候选，再按接缝角度选择 |

另外纳入同一原句的`P0004`（32帧）、`P0008`（35帧），没有人工补造指骨动作。完整序列保留起收势；完整不等于已经认证为适合所有句子上下文。

官方[固定版本的检索裁剪实现](https://github.com/JianHe0628/SignSparK_Retrieval/blob/b1ed92705cafd2c9ff0c72a89df14b0a9ee62258/signspark_retrieval/dictionary.py)也是直接对配对LMDB取`start:end`。本轮没有证据支持给ISO索引乘时间系数，因此未凭感觉改时间基准。没有对同ID的SOKE参数套用这些区间。

原生blob中的文字/Gloss仅用NPY头和`pickletools`指令检查提取字符串作来源核对，不执行object pickle；构建和运行仍只读取数值动作字段。源动作和人物没有被覆盖。

## 实现和数据约束

- 构建器新增`--standalone-only --label 谢谢`。先统计同一视频全部有效标注，并排除含无效标注的视频，再筛标签；不能把“医生”过滤后误称独立“谢谢”。仅train源可构建完整词条，仍验证整个LMDB和配对标注摘要。
- 完整词条记录原ISO区间、完整源区间、唯一标注数量和原blob摘要，合并器及运行时验证这些绑定。篡改文件、缺标记、范围不完整或骨架不一致均拒绝。
- 可选`standalone_library`只对完整文本精确匹配，允许去除句末`。！？.!?`；不把长句模糊替换为“谢谢”。句中完整素材优先保留规划Gloss顺序和全部源帧。
- 没有修改GLB、蒙皮、指骨参数、预览FPS或源拟合；没有通用手指平滑覆盖原始局部旋转。
- 合并前后均为1,990标签／14,317候选。仅“谢谢”候选集合变化，另1,989标签候选相同，最初5,916单位全部保留。“你好”“请等一下”的实际输出参数与旧库完全相同。

| 来源身份 | SHA256 / bundle |
|---|---|
| train LMDB完整`data.mdb`，3,327,815,680字节 | `7dbc5dc932c6e34ea1d7d47976e8db8fb90ba4c2aff3eb05a6991c7008366222` |
| 配对`csl_iso.train` | `92cabbe0fcc97770abd491802be00cb35883a2237cabb7f0c5de847845355f2c` |
| 新选`S000003_P0000_T00`原始blob | `2ae15a6cb957066a0bf9750ac903a688c5f23dd7b6d99cd3de5745d06f4324e4` |
| 该完整导出单位NPZ | `19464f2a05d8914a9203e918a2c92235413bcce7baf4f534158bd0b4b49fff05` |
| 完整“谢谢”小库 | `csl-words-08cb8863983b420e92ba682a` |
| 产品合并库 | `csl-product-ae27c39e228ec7375afb2081` |
| 冻结Godette v14A | `535fc9ab96ec3d837e2568817abc41d593fac1c4fcad8565b45b29c5f4472f5e` |

产品合并库继续标记`product-demo`、`evaluation_eligible=false`，既有混合split边界保留，不把产品检索当独立测试集成绩。

## 实际验证

| 验证 | 结果与边界 |
|---|---|
| RED→GREEN | 先出现缺少完整词条接口及句中误选短片段的失败，再实现；日志保存在本机产物目录 |
| 原生构建/合并/规划回归 | **111 passed**，含临时LMDB样例的数值保真测试，无跳过；实际数据另有完整构建审计 |
| 组合、API、场景、沟通回归 | **124 passed**，1条既有Starlette弃用警告；与上组有重叠，不相加 |
| Node/Three/产品控制器 | **119 passed，9 skipped，0 failed**；跳过项包括旧v11/v12/v13口部及未配置相应源动作的显示修订检查，不计通过；本轮“谢谢”几何另有专门实测 |
| wheel构建 | 成功；安装资源目录与关键模块核对见本轮机器可读审计 |
| 真HTTP“谢谢” | `ready`，30帧、精确原生匹配、规划耗时字段0；非T2G推理正确率 |
| 真HTTP“我想去成都。” | `blocked`，缺词不部分播放 |
| 真HTTP“明天早上我们一起去医院看朋友。” | `needs_review`，已有遗漏保护保留 |
| 原产品页 | “谢谢！”实际播放结束，`0:01 / 0:01`，右手位于胸前、脸部无遮挡；保留原35条对话并新增本次1条，未刷新或清空 |
| 新测试页“你好，谢谢” | 实际播放72帧，规划Gloss为`你 好 你 谢谢`，末词选完整35帧`P0008`。这验证句中选材，不证明该规划语义正确 |
| “谢谢你”实际规划 | 模型只返回`谢谢`，30帧；该简化仍需语言核对，没有因动作替换而宣称整句语义通过 |
| 冻结v14A几何检查 | 30帧、900个指骨局部键不变，源输入不变；233个插值时刻、14,446个头部核心关节/线段采样无命中，额外避让偏移/旋转均为0 |

最后一项是固定保守头部核心检查，**不是**完整皮肤、服装、指间、手间碰撞认证，也不验证语言自然度。本轮未测物理摄像头、麦克风、新一轮完整GPU慢生成、全仓库测试或远端CI；既有“喝水”等组合的质量限制继续保留。

修订前：

![修订前谢谢短片段](evidence/thanks-native-2026-11-09/thanks-before.jpg)

修订后，原产品页实际播放结束：

![修订后完整谢谢](evidence/thanks-native-2026-11-09/thanks-after.jpg)

人物使用既有Godette v14A，授权及署名沿用[人物说明](../../slbridge/animation/godette_v14a_attribution.md)。截图是本地产品渲染，未作图像修饰。

## 本机复现与版本边界

源码工作树：`C:/Users/ADMIN/.codex/worktrees/csl-animation-workspace/SignLanguage-Bridge`，分支`codex/local-godette-v14a-demo-20261107`，基线HEAD `ca20cbb94acf6cc26c02e07db78ced165d41393c`。本轮修订仍为本地未提交改动，未新增推送或合并；此前GitHub Draft PR #2保持独立版本身份。

实际服务入口：<http://127.0.0.1:8879/communicate/>。复用原识别服务8765，只重启产品8879及其所属规划器。新外部数据目录为：

```text
G:/Archive/Zenodo/SignSparK_Retrieval/standalone-thanks-20261109
G:/Archive/Zenodo/SignSparK_Retrieval/word-library-v3-native-thanks-20261109
```

本机启动配置`E:/githubproject/SignLanguage-Bridge/artifacts/product-bridge-20261108/local-profile.json`已指向新库。修订前后配置和所有日志、原生来源核对、几何探针、真HTTP及浏览器响应保存在`E:/githubproject/SignLanguage-Bridge/artifacts/thanks-gesture-20261109/`；其中`profile-before.json`与`profile-after.json`保留回退依据。旧词库及原LMDB未覆盖，外部动作、模型和视频不纳入Git。

精简审计：[verification.json](evidence/thanks-native-2026-11-09/verification.json)。
