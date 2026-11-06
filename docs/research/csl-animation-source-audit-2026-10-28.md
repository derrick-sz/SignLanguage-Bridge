# SignBridge 与 VLibras 一手来源核验

核验日期：2026-10-28（Asia/Taipei）。方式：只读 GitHub API、固定提交源码、公开资产元数据及官方文档；未安装或执行上游程序，未做浏览器播放、动作视觉验收或手语者理解测试。

## 结论

“独立规划器 + 可追溯动作库 + 虚拟人播放器”的工程方向成立。SignBridge 有可复用播放器、动作导入校验、人工启用、SMPL-X 姿态重定向与公开动作快照，但公开证据不足以把它称为“已审核标准中国手语词典”。VLibras 可借鉴其翻译、词典、播放与社区审核的分层；其葡语/Libras 资源不能直接成为 CSL 资源。[S1][S2][S3][V1][V2]

## 1. 冻结上游版本

SignBridge 仓库为 `feifei9126/signbridge`；核验时默认分支 `codex/t08-avatar-demo`，提交 `3503ff7802bc991ccf5e0753c2fe41b79689b860`，提交时间 2026-10-27。用户列举的六个文件均存在，但实际仓库相对路径有 `signbridge/` 前缀。[S1]

| 源码 | 核验结果 | 不能由此推出 |
|---|---|---|
| `signbridge/src/avatar/sign-language-data.js` | `SIGNS` 有 52 个内置条目；`id/gloss/texts/tags/frames/meta`，存在最长别名匹配、单字回退与功能词跳过 | 已具备 CSL 句法或所有词条获语言审核 |
| `signbridge/src/avatar/motion-asset.js` | 只验证 `signbridge-motion` v1、`humanoid-local-v1`、弧度、骨骼绑定及数值/大小/时长等；导入需要 `custom_` ID | 原生通吃任意 SMPL-X、glTF 或跨骨骼文件 |
| `signbridge/src/avatar/motion-library.js` | 新导入条目默认禁用；启用确认绑定序列化资产内容；内容改变不沿用旧批准 | 人工启用等于专业语言认证 |
| `signbridge/src/avatar/motion-player.js` | 支持 quaternion-track、最多 40 段 motion-sequence 顺序展开、骨架插值、短 blend、排队去重和过期丢弃 | 具备手语协同发音、接触约束、句法正确的连续动作合成 |
| `signbridge/src/avatar/humanoid-rig.js` | 统一骨骼访问与四元数 slerp | 已实现全身动力学、碰撞或语义质量门 |
| `signbridge/src/demo/demo-retarget.js` | `createT08Retarget` 将 axis-angle 姿态转换到当前人物骨骼 | 完整无损 SMPL-X/表情/全身重定向 |
| `signbridge/src/demo/demo-avatar.js` | 读取并校验 T08/候选/实验资源、编译并缓存姿态，支持来源分流和播放控制 | 任意文字在线生成新动作 |

以上均直接核对固定版本源码：[S4]–[S10]。`sign-language-data.js` 的字段是 `meta`，不宜写成已经具有完整来源、许可、版本、审核状态的统一 `metadata` 协议。

## 2. 公开动作库究竟提供什么

1. 2026-10-27 公开快照共 **100 个片段 = 14 个 T08 参考 + 3 个 T08 候选 + 83 个 SMPL-X 拟合片段**，不能写成 100 个独立标准词条。普通 `npm run build` 为词库版，公开动作版走 `npm run build:motion`。[S2][S3]
2. 对公开 `experimental-index.json` 直接解析：83 个 clip ID、83 个不同 `text`、36 个不同哈希 `source_id`。它们混合单词与整句，例如“通知”和“我去开会汇报抽样调查落实的情况”。条目没有 `gloss` 或 `sign_id` 字段，不能由中文标签计算标准 Gloss 覆盖数。[S11]
3. 索引没有足以独立认证每段语种、教材身份或公共数据集归属的完整原视频；来源以摘要引用，公开包不含原视频。上游文档把研究目标及素材称为中国手语教学/句子资料，但这不替代逐段语言审核。不能把这 83 段归因为 SignAvatars、CE-CSL 或某个经认证 CSL 词典。[S2][S11][S12]
4. 83 段的审批记录为 `scope=motion_quality_only`，依据是项目用户在对话中的明确批准声明；记录绑定每段的资产、源视频、拟合与姿态摘要。未见据此可证明的具名手语专家、独立双人语义盲评或“标准中国手语”认证。[S13]
5. `mirror_status=unknown`、`signer_id=unconfirmed`、`training_data_member=false` 等继续保留；审批不确认中文含义、词界、镜像方向、人物身份、划分和训练资格。manifest 明确 `semantic_accuracy_certified=false`，`training_ready/model_training_ready/deployment_ready=false`。[S2][S11][S13][S14]
6. T08 资源记录 `runtime=precomputed_cpu_token_decode`、`text_to_sign_trained=false`、`heldout_accuracy_available=false`，14 参考来自两个哈希源视频；这属于动作编码/解码资源，不是已经训练好的中文到新句手语生成器。[S15]

因此宜复用其**资产验证、播放器及审核流程**，重新定义本项目可播放资源的语言审核门。不能直接把“中文 text 完整匹配”当成“CSL 规划完成”。[S4][S6][S11]

## 3. SMPL-X 重定向和连续播放的实际边界

`demo-retarget.js` 绑定头颈、肩臂与手腕共 10 个骨骼，双手五指各三节共 30 个骨骼，总计 40 个；双手通道存在，不等于手形/掌向/接触准确。源码明确让 jaw 维持静止；T08 元数据明确 eyes 静止、facial coefficients 不重定向，root/lower-body/torso 在其表示范围外静止。实验索引也声明 root、下肢和脊柱静止。因此现有播放器不能直接承担完整非手控语法表达。[S8][S11][S15]

`motion-player.js` 的 sequence 是展开片段；quaternion-track 分支在同一片段的相邻采样间插值，未见词间速度连续、手形保持、接触/碰撞、空间指代与表情轨的语言约束。静态帧分支短时间 blend 也不能证明这些性质。第一版可保留明确边界和停顿；应先证明单动作可理解，再评估两动作过渡，不应以“更平滑”自动晋级。[S7]

上游六批检查记录包括全帧数值/结构/时序筛查和采样人物截图，并明确未声称逐段人工观看全部视频或完成语义批准。公开源码与已有测试记录只能支持其各自程序范围；本次未运行其测试，也不复述上游测试为本项目 PASS。[S16]

## 4. VLibras 可借鉴的具体部分

巴西政府官方定义为葡语数字内容到 Libras 的自动翻译套件；官方 API 目录分别列出“葡语→gloss”“gloss→视频”“按 platform/sign 获取动画”“区域词典”和“词典 trie”。目录较旧，本文只用作架构证据，没有验证这些示例端点当前可直接调用。[V1][V3]

WikiLibras 是协作构建平台：贡献者提议、动画制作者制作、专家评估、语言人员修订翻译，获专家批准的动作才进入词典。官方页面称库中有 21,000+ 个 Libras signs，数量不能当作 CSL 覆盖或用户理解正确率。[V2][V4]

不能把当前 VLibras 简化为“只有逐词替换”。核验的 Text Core（`550b5eb215ec080dd9a50055f3f92de3b4bbdf8b`）依赖 `vlibras-translate` 与 `vlibras-deeplearning`，部署配置有 `ENABLE_DL_TRANSLATION`、`ENABLE_REFINEMENT` 及 LLM 相关选项；本文未执行其翻译引擎，不能断定官方线上服务启用了哪种配置。[V5]

建议借鉴：目标语规划与动作检索分离、区域变体、专门动画制作工具、专家批准和可追溯版本。CSL 与 Libras 应维护独立规划器和表达资源；跨语言可共享概念查询和技术播放器，不共享语言正确性结论。[V1]–[V5]

## 5. 授权需按层核对

- SignBridge 项目自有代码为 MIT；公开动作 README 特别声明第三方 SMPL-X/SMPLer-X/SOKE、人物模型、源视频及衍生权利不会因公开分发自动重新许可为 MIT。[S3][S17]
- 核验的 VLibras Text Core、Translator API、Webjs Player 源码仓库标明 LGPLv3；不能由代码许可证直接推定全部动画、人物形象、字典或在线服务拥有相同使用权。[V5][V6][V7]
- VLibras 当前官方使用条款对在线工具、人物和翻译用途另有规定，代码复用、服务调用、提取字典、重新分发动画应分别核查。本文仅记录一手条款，不作法律结论。[V8]

## 6. 对技术路线的修订建议

1. 将第一阶段目标写为“有限领域、有限词汇/短语、经过目标手语使用者审核的动画检索播放”。先做 20–50 个可审核单元和少量固定整句，数量仅为建议的工程试点规模。
2. 词典设计至少拆开语言审核、动作质量、来源授权和部署批准，审核绑定具体动作版本与目标人物重定向结果。`human_approved` 单一布尔值不足以支持自动发布。
3. 将 `gloss` 看作特定标注规范中的标签；映射到 `sign_id` 时记录 `gloss_namespace`、语种、变体和来源，不默认 SLRT 标签天然等于词典键。
4. 第一版优先允许完整审核短语命中；未知词、否定/疑问或空间关系没有实现时返回“缺资源/需审核”，不要靠功能词删除或单字回退制造完整句子的假象。
5. 先复用 40 骨骼播放器做可追溯播放，另列非手控轨、接触约束和连续编排作为后续模块；保留中立姿态和字幕兜底。
6. 将视频重建定位为“动作候选生产”，而非标准动作自动生产。重建、人工修正、目标人物重定向和语言可理解性审核逐层验收。

以上为研究建议，不是本项目已接入或已通过验收的能力。

## 来源

- [S1：SignBridge 固定提交](https://github.com/feifei9126/signbridge/commit/3503ff7802bc991ccf5e0753c2fe41b79689b860)
- [S2：公开动作发布范围与验收说明](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/public-motion-release-20261027.md)
- [S3：公开动作资产 README](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/README.md)
- [S4：sign-language-data.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/sign-language-data.js)
- [S5：motion-asset.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-asset.js)
- [S6：motion-library.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-library.js)
- [S7：motion-player.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-player.js)
- [S8：demo-retarget.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/demo/demo-retarget.js)
- [S9：humanoid-rig.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/humanoid-rig.js)
- [S10：demo-avatar.js](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/demo/demo-avatar.js)
- [S11：83 段实验动作索引](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/experimental-index.json)
- [S12：数据与候选来源说明](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/sign-language-dataset.md)
- [S13：83 段 motion quality 审批原始记录](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/experimental/motion-approval.json)
- [S14：公开 manifest](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/manifest.json)
- [S15：T08 动作 JSON 元数据](https://raw.githubusercontent.com/feifei9126/signbridge/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/t08-motion.json)
- [S16：六批动作质量检查说明](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/six-batch-motion-audit-20261016.md)
- [S17：SignBridge MIT LICENSE](https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/LICENSE)
- [V1：巴西政府 VLibras 官方说明](https://www.gov.br/governodigital/pt-br/acessibilidade-e-usuario/vlibras/vlibras)
- [V2：WikiLibras 官方协作平台](https://wiki-dth.vlibras.lavid.ufpb.br/)
- [V3：巴西政府 VLibras API 目录](https://www.gov.br/conecta/catalogo/apis/vlibras)
- [V4：WikiLibras 官方开发说明](https://vlibras.gov.br/files/Doc_VLibras_Wikilibras.pdf)
- [V5：VLibras Text Core 固定版本 README](https://github.com/spbgovbr-vlibras/vlibras-translator-text-core/blob/550b5eb215ec080dd9a50055f3f92de3b4bbdf8b/README.md)
- [V6：VLibras Translator API 固定版本 README](https://github.com/spbgovbr-vlibras/vlibras-translator-api/blob/5a00fd2bccb77ffa442eec4dbd6c8b036178729a/README.md)
- [V7：VLibras Player Webjs 固定版本 README](https://github.com/spbgovbr-vlibras/vlibras-player-webjs/blob/5389767f8dcb915e6cd1a768095bfefa825c3d14/README.md)
- [V8：VLibras 官方使用条款](https://www.vlibras.gov.br/privacy.html)

[S1]: https://github.com/feifei9126/signbridge/commit/3503ff7802bc991ccf5e0753c2fe41b79689b860
[S2]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/public-motion-release-20261027.md
[S3]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/README.md
[S4]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/sign-language-data.js
[S5]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-asset.js
[S6]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-library.js
[S7]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/motion-player.js
[S8]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/demo/demo-retarget.js
[S9]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/avatar/humanoid-rig.js
[S10]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/src/demo/demo-avatar.js
[S11]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/experimental-index.json
[S12]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/sign-language-dataset.md
[S13]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/experimental/motion-approval.json
[S14]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/manifest.json
[S15]: https://raw.githubusercontent.com/feifei9126/signbridge/3503ff7802bc991ccf5e0753c2fe41b79689b860/signbridge/assets/public-motion-v1/t08-motion.json
[S16]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/docs/six-batch-motion-audit-20261016.md
[S17]: https://github.com/feifei9126/signbridge/blob/3503ff7802bc991ccf5e0753c2fe41b79689b860/LICENSE
[V1]: https://www.gov.br/governodigital/pt-br/acessibilidade-e-usuario/vlibras/vlibras
[V2]: https://wiki-dth.vlibras.lavid.ufpb.br/
[V3]: https://www.gov.br/conecta/catalogo/apis/vlibras
[V4]: https://vlibras.gov.br/files/Doc_VLibras_Wikilibras.pdf
[V5]: https://github.com/spbgovbr-vlibras/vlibras-translator-text-core/blob/550b5eb215ec080dd9a50055f3f92de3b4bbdf8b/README.md
[V6]: https://github.com/spbgovbr-vlibras/vlibras-translator-api/blob/5a00fd2bccb77ffa442eec4dbd6c8b036178729a/README.md
[V7]: https://github.com/spbgovbr-vlibras/vlibras-player-webjs/blob/5389767f8dcb915e6cd1a768095bfefa825c3d14/README.md
[V8]: https://www.vlibras.gov.br/privacy.html
