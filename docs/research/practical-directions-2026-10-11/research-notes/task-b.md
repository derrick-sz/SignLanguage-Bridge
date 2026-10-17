---
task_id: b
role: Product Landscape Analyst
objective: 核查现有一手产品与部署模式，辨别可借鉴应用和集成边界
status: complete
confidence: medium
sources_found: 11
acceptance_met: yes
---

## Sources

所有链接于 2026-10-11 核查。企业页面仅证明其公开产品说明，不证明独立准确率、用户效果、盈利或项目适配。

- [B1] Hand Talk App | https://www.handtalk.me/en/app/ | VENDOR / FIRST-PARTY | 页面未标发布日期，现行产品页 | 对语言方向和学习功能的权威性高，效果证据弱。
- [B2] VLibras — Governo Digital | https://www.gov.br/governodigital/pt-br/acessibilidade-e-usuario/vlibras/vlibras | GOVERNMENT / PRIMARY | 页面未标发布日期，现行政府入口 | 官方定义、部署形式、使用边界可信；站点数量未独立审计，不纳入结论。
- [B3] VLibras Terms of Use and Privacy | https://www.vlibras.gov.br/privacy.html | OFFICIAL TERMS / PRIMARY | 更新 2026-08-27 | 对其托管服务、头像、输出的当前使用条件权威，不能推定覆盖独立代码许可证的全部范围。
- [B4] VLIBRAS — Catálogo de APIs governamentais | https://www.gov.br/conecta/catalogo/apis/vlibras | GOVERNMENT API CATALOG / PRIMARY | 条目标示版本 1.0，2018-10-25 | 可证实公开描述的接口方向；版本旧，网页端点为占位形式，未验证运行或 SLA。
- [B5] VLibras translator-text-core LICENSE | https://github.com/spbgovbr-vlibras/vlibras-translator-text-core/blob/dev/LICENSE | PROJECT SOURCE / PRIMARY | 现行 dev 文件，最后变更日未核实 | 明确这一组件的 LGPLv3；不延伸为头像、字典、托管服务统一许可证。
- [B6] Signapse Transport Translations | https://www.signapse.ai/transport-translations | VENDOR / FIRST-PARTY | 页面未标发布日期，现行产品页 | 明确语言、领域、集成方式；宣传延迟/准确率未独立验证。
- [B7] SignStream API | https://www.signapse.ai/signstream-api | VENDOR / FIRST-PARTY | 页面未标发布日期，现行产品页 | 明确文本到手语视频、API 和无人工审核边界；不证明输出质量。
- [B8] SignStudio Platform | https://www.signapse.ai/signstudio-platform | VENDOR / FIRST-PARTY | 页面未标发布日期，现行产品页 | 明确离线内容制作和质量检查/人工选项；同页语言细节存在冲突。
- [B9] South Western Railway Customer Report, March 2025 | https://www.southwesternrailway.com/-/media/files/other/about-us/customer-report/customer-report-march-2025.pdf | DEPLOYER REPORT / PRIMARY | 2025-03，涵盖 2024-07 至 2025-02 | 部署方确认已接入 One Scan，独立于供应商宣传；没有 BSL 单独任务成功率或对照评估。
- [B10] Signapse Translation Process | https://www.signapse.ai/post/signapses-sign-language-translation-using-generative-ai-achieving-photo-realism-and-accuracy | VENDOR TECHNICAL DESCRIPTION / FIRST-PARTY | 2023-05-10 | 历史产品流程的一手说明；不能当作所有现行产品的架构。
- [B11] Hand Talk Plugin Installation | https://docs.handtalk.me/en/docs/introducao/instalacao/ | VENDOR DOCUMENTATION / PRIMARY | 当前 URL 页面标 v12.0.0；主入口标 v13.0.0 | 明确公开 JS 文档与有效 token 要求；版本信息须在实际接入时重新确认。

## Findings (facts only)

- Hand Talk App 产品页提供葡语→Libras、英语→ASL 的内容转换；词典有主题/词检索、离线访问、可调播放速度；Hands Up 提供学习模块，称内容由聋人教育者、语言学者与 CODA 制作。[B1]
- Hand Talk 插件文档要求网站脚本和有效访问 token；获得 token 需联系其业务人员。公开集成说明不等于开放源码或无需商业安排。[B11]
- VLibras 官方定位是葡语数字内容→Libras，免费、开源工具覆盖浏览器、网站 widget、桌面与移动端；明确不能替代人工译员。[B2]
- VLibras 最新条款把工具、头像、品牌与生成译文的使用限定在数字无障碍范围；商业/广告/选举活动使用需要负责方事先明确授权；条款列出教育、司法、影视等限制。[B3]
- VLibras 2018 API 目录描述文本→gloss、gloss→视频、词典与地区变体接口；本次没有发送请求，也没有确认端点、认证、SLA 或当前兼容性。[B4]
- VLibras translator-text-core 代码组件 LICENSE 是 GNU LGPL Version 3；这与托管服务条款属于不同材料，尚未完成组件、头像、词典资产的逐项授权审查。[B5][B3]
- Signapse 交通产品公开支持英语文本→BSL/ASL，可接车站/机场信息屏与 API；场景是发车、延误、扰动原因、站台变化等信息播报。[B6]
- SWR 2025-03 客户报告 PDF 第 3 页确认 2024 年 9 月把 AI 生成的 BSL 接入 One Scan；第 6 页说明车次时间、扰动、下一站等译文在乘客设备播放。报告没有给出该功能单独的理解准确率。[B9]
- SignStream 是英语短句→BSL/ASL 视频/输出流的 API 和聊天工具；官方 FAQ 明确无人工语法检查或质保，可能有语法错误，建议用于非正式消息或概念测试，称不适合正式、法律或官方用途。[B7]
- SignStudio 面向已有视频、文档、音频或 YouTube 内容的后期制作，公开提供 AI 加质量检查或人工翻译，并有企业 API。[B8]
- Signapse 2023 技术描述给出的交通流程是维护预录译文、译员选择片段、AI 衔接和业务数据更新。这是历史说明，不能外推为现行 SignStream 或所有自然语言生成的流程。[B10]

## Analysis (your synthesis)

1. **最有借鉴价值的是任务与信息结构。** 交通播报有明确的业务事实来源、有限消息类型和可核对的时间/地点槽位。对于 SignLanguage-Bridge，可先研究“服务请求识别→用户确认→员工回复选择→审核手语视频播放”，或“业务通知→审核双语说明/手语视频”；这并不要求先解决任意自然手语到任意语言的全自动双向会话。[B6][B9][B10]
2. **学习助手与沟通助手的验收目标应分开。** 学习工具可以提供参考视频、回放、主题词表、教师审核与练习记录；自动识别的动作相似度只能形成待验证反馈，不能直接当作语言正确性或熟练度分数。现有产品公开功能证实这种产品类别存在，不证实本项目已有自动教学效果。[B1]
3. **文本→手语内容可作为互补方向。** 经本地教师/译员审核的 CSL 与 Libras 参考视频库和固定回复板，有望补足听人到聋人的回复路径。不能直接把葡语词逐个拼接成符合 Libras 语法的句子；建议先整句/意图审核模板，数字、日期、地点也需可验证呈现。[B8][B10]
4. **内容制作工具可能比全天候自动翻译更早交付价值。** 检索参考片段、草稿字幕、时间轴编辑、审核记录、变体管理和导出，是可由人工把关的工具闭环。与 SignStudio 的公开工作流相符，但本项目是否更省时需测量每分钟素材的人工作业时间与审核后错误率。[B8]
5. **不要从竞品语言覆盖推断本项目可直接接入。** Signapse 当前页面只列 BSL/ASL；Hand Talk 列 Libras/ASL；本次没有找到这些页面明确的 CSL 支持。它们主要展示手语输出方向，不是摄像头手语识别端点。[B1][B6][B7]
6. **兼顾真实产品与中巴竞赛展示的任务链（研究建议，未验收）：** 选一个低风险校园/展会接待场景，采用已审核的 CSL/Libras 少量意图；用户打手语→显示候选→本人确认→呈现对应的中葡文本/回复视频→对方选择回复→完成一个明确任务。展示一次正常流程和一次未知/歧义→澄清/改选流程。使用预先审核的跨语意图表；如果没有合格 Libras/CSL 视频与本地审核，就明确展示文字路径，不把模型识别结果直接包装成双向手语翻译能力。[B1][B8][B10]
7. **内容审校工作台的比赛展示：** 导入一段授权素材→模型草稿→定位低置信度片段→专家修改并关联参考视频→保存审定版本→导出字幕/审核记录。它能展示模型、字幕时间轴与人工审定如何一起完成工作。实用性用人工节省时间、审后错误数和返工率评估；这些效果必须另测。[B8]

## Support snippets / paraphrases for top claims

- Claim: 交通通知是已有实际部署的受约束应用，未证明自由双向翻译。
  Source: [B9] PDF 第 3、6 页。
  Support: 部署方描述 One Scan 从列车运行信息生成 BSL 并在乘客设备播放，消息围绕车次、扰动与后续站点；没有乘客手语输入流程或自然会话准确率。
- Claim: 无审核的即时 API 与有质量控制的内容制作应分层。
  Source: [B7] FAQ；[B8] 产品说明和上传/语言选择流程。
  Support: SignStream 明示 AI 输出没有人工语法检查，限制正式用途；SignStudio 介绍专家翻译和质量检查选项。
- Claim: 免费开源宣传不能直接等于托管服务与头像可随意商用。
  Source: [B3] General Conditions / Restrictions / Intellectual Property；[B5] LICENSE。
  Support: 当前服务条款要求商业用途事先授权，限制头像与译文用途；代码组件另有 LGPLv3。两者并列核查，不能用一种替代另一种。
- Claim: 词典、练习和重播已是公开产品形态。
  Source: [B1] 产品功能。
  Support: App 提供主题词典、调速、离线访问和学习模块，适合研究参考视频与练习支持；这些特性并不构成对动作识别评分的证据。

## Conflicts / unresolved issues

- SignStudio 同一页的概述、Benefits 和 FAQ 称 AI BSL/ASL；Features 又写 AI BSL 带质量检查、ASL 即将提供。因此只采纳“提供质量检查/人工选项的内容制作模式”，不确定其 ASL AI 套餐当前覆盖。[B8]
- Hand Talk 文档主入口显示 v13，Installation URL 内容仍标 v12；集成版本需实际验证，本次只确认 token 机制和文档存在。[B11]
- VLibras 服务条款和组件 LGPLv3 不能合并为单一法律结论。确有集成可能，但外部服务、头像、品牌、训练资料、独立代码分别是什么授权尚待审查。[B3][B5]
- 供应商称实时/高准确率没有本次独立用户研究支持。SWR 确认部署存在，也未提供 BSL 单项效果评估。[B6][B7][B9]

## Leads discovered

- 本地手语教师/译员审核的整句/意图视频库，可同时用于小词表双向回复板、工作人员培训与语义数据标注。
- SignStudio 类型的后期内容工作台，关注审核成本、版本差异与导出，而非仅优化自动推理耗时。
- VLibras 的地区变体接口提示本地词典/模型应记录地域与审核来源；目录很旧，实际端点需后续工程核验。[B4]

## Gaps

- SignAll：两次限定域名检索未返回可用的现行官方产品页，https://www.signall.us/ 获取错误。本次无法确认其当前 Learn 产品、购买、API、运营或许可；不把历史产品存在写成当前可采用方案。
- 未实测任何竞品 App/API，未购买、注册、请求 token 或联系厂商；未确认接入成本、离线模型权利、服务 SLA、价格或效果。
- 未找到上述材料中明确 CSL 现行支持；这只是所读公开页的覆盖范围，不声称全市场没有 CSL 产品。
- 个人常用语回复板是本线程提出的产品假设，不是从这些页面证明了中巴用户偏好。需要本地参与者验证是否优于写字、文本/图片或人工支持。
- 核查工作量：9 个搜索查询（含 3 个许可/部署/SignAll 追查），阅读 14 个可用独立页面/文件 URL，另 1 个官网访问失败；注册接受 11 项，其他为导航/功能说明。
