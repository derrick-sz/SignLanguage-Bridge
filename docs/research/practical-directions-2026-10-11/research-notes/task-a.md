---
task_id: a
role: Deaf user needs researcher
objective: Find original user and representative-organization evidence for practical SignLanguage-Bridge applications
status: complete
confidence: medium
sources_found: 4
acceptance_met: yes
searched_at: 2026-10-11
---

## Sources

- [A1] U.S. Deaf Community Perspectives on Automatic Sign Language Translation | https://danibragg.com/papers/Automatic_Sign_Language_Translation_Perspectives.pdf | Nina Tran, Richard E. Ladner, Danielle Bragg | ASSETS 2023, DOI 10.1145/3597638.3614507 | ORIGINAL USER SURVEY / author-hosted conference paper | Aut:9 Rec:6 Rel:9 Dep:7 = 7.75. Original methods/results, but small U.S. sample, stated preferences rather than observed deployment. Dates: October 2023. Read PDF pp. 2–5, especially methods, scenario ratings, concerns and conclusion.
- [A2] 5th WFD Conference – Artificial Intelligence and Sign Languages | https://wfdeaf.org/artificial-intelligence-and-sign-languages/ | World Federation of the Deaf, reporting Maartje De Meulder's workshop | OFFICIAL REPRESENTATIVE ORGANIZATION / press release | Aut:9 Rec:9 Rel:8 Dep:5 = 7.75. Strong for organizational priorities, brief workshop report rather than empirical effectiveness evidence or a universal ban on AI. Date: 2025-08-12; fetched current page on 2026-10-11.
- [A3] Reimagining Sign Language Technologies: Analyzing Translation Work of Chinese Deaf Online Content Creators | https://arxiv.org/html/2602.10235v1 | Xinru Tang, Anne Marie Piper | CHI 2026, DOI 10.1145/3772318.3790624 | ORIGINAL QUALITATIVE STUDY / public author manuscript | Aut:9 Rec:9 Rel:10 Dep:9 = 9.25. Direct Chinese creator interviews and transparent methods/limitations; small, selected creator sample and interviews from 2022–2023. Author manuscript date 2026-02-10; conference April 2026. Read sections 4.2–4.3, 6.2–6.3.
- [A4] The hearing impaired communicate through sign language: the complexity of gaining access to healthcare services | https://scielo.br/j/csp/a/f7xLDvF3cHXKJhzJBLTndLh/abstract/?ilang=en&lang=en | Rafael Rodrigues de Moraes, Laura Sabrinny de Sá Pereira, Priscila Maria Stolses Bergamo Francisco, Giovana Astolfi Pico, Nubia Garcia Vianna | Cadernos de Saúde Pública 41(9), 2025, DOI 10.1590/0102-311XPT00211524 | ORIGINAL CROSS-SECTIONAL STUDY / journal site | Aut:9 Rec:8 Rel:8 Dep:7 = 8.0. Direct Brazilian Libras respondents, accessible questionnaire; one metropolitan region, self-reports, no technology intervention. Read abstract and Table 1, not full Portuguese article. PubMed mirror: https://pubmed.ncbi.nlm.nih.gov/41059818/ .

Scores are this thread's screening judgments, not objective quality measurements. All four fetched; search also surfaced papers not promoted into accepted evidence.

## Findings (facts only)

- 美国调查为 32 位自认 DHH、具 ASL 能力的参与者，采用电子邮件/社交媒体招募；81% 将 ASL 列为主要语言。商户场景平均使用意愿最高，其后是自助设备与非正式学习；医疗场景对准确度/速度等要求较高，使用意愿分歧也大。[A1]
- 同一调查的手机应用选择人数为 29/32；受访者担心缺失 ASL 语法、不同用户表现差异、社区参与不足及隐私等问题。论文明确要求研究美国之外的观点。[A1]
- WFD 2025 会议报告强调聋人社区参与 AI 讨论，评估数据所有权、社区参与、透明度和真实使用表现。[A2]
- 中国研究访谈了 13 位聋人内容创作者，实际访谈时间为 2022 年 12 月至 2023 年 5 月；使用目的/滚雪球抽样，参与者熟练使用中文与手语，多来自较发达地区并有高等教育背景。[A3]
- 中国研究观察到签语、字幕、图像等混合表达，作者提出多候选译法和协作修改方向；没有测量观众理解效果，作者也说明自身手语能力有限。[A3]
- 巴西 Campinas 都市区横断研究有 316 位成年 Libras 使用者，问卷可用手语访问。公立服务使用者中的 64.3%、私立服务使用者中的 67.6% 认为与医务人员沟通较差；不同分母为 255 与 226，不能将两组百分比视为全部 316 人。[A4]
- 巴西研究表 1 同时列出手语不通时被要求葡语书写、口语表达，以及亲属/翻译协助等情况；研究未评估自动翻译工具。[A4]

## Analysis (our synthesis)

- **优先验证内容创作者审校工作台**：上传自己的手语视频，切片、回看、候选字幕、人工修改和确认、保存术语/变体、导出字幕/翻译稿。其价值可以通过总编辑时间、漏改错误、观众理解任务测量，而不要求模型立即做到全自动自由交流。中国直接研究给这一方向提供了比纯美国偏好更相关的起点。[A3] 当前项目适配需由 C 线程核实；这是产品假设，尚未证明有付费需求。
- **条件优先验证商户/校园服务台的有限任务闭环**：先挑选查询地点、领取物品、提出重复/帮助等可纠错任务，以审核短语、图片、触屏和本人确认完成任务。ASL 调查提示值得探索，但中巴用户及现场工作人员仍需本地任务观察；不可用美国样本宣布中巴需求成立。[A1]
- **个人常用短语/表达辅助**可作为共享能力：允许保存本人审核过的表达、选择语言/变体、预览后确认；保留手语视频和图片入口。不能默认所有聋人都愿意或能够用中文/葡语文本修正模型。用户选择输入/输出方式是待测试设计原则，而非已验证偏好。[A3][A4]
- **学习/录制反馈应聚焦提示与老师审核**：参考动作回看、局部手形/轨迹比较可提供练习线索，不能仅凭“与一个样本相似”评定所有正确手语。多变体及社区审核适合与内容/词典工具共享。[A3]
- **医疗方向宜作为后续独立任务研究**：巴西研究显示障碍，并不证明算法可有效解决；先研究导航、预约信息和请求翻译等可纠错入口。自由病史、诊断、剂量和知情同意需要另设真实任务评估及专业参与，不能把“需求严重”直接等同于“适合首发自动通译”。[A4][A1]
- 产品设计需把实际使用者的决策权做具体：哪些视频留存/可删除、是否参与训练、谁审核表达、错误如何修正。WFD 报告支持这些问题的必要性，不提供某个产品必然成功的证据。[A2]

## Support snippets / paraphrases for top claims

- Claim: 商户和自助设备是可测试切入点，不能外推中巴偏好。
  Source: [A1], PDF pp. 2–3 and conclusion pp. 4–5.
  Support: 论文显示商户使用意愿均值最高，并明确后续需研究美国之外的 DHH 观点；调查只询问观点，未做真实部署实验。
- Claim: 中国聋人内容创作工具是值得验证的独立方向。
  Source: [A3], sections 4.2–4.3 and 6.2–6.3.
  Support: 作者研究双语聋人创作者混合媒体的翻译实践，讨论多个候选译法与协作修改；并承认没有调查观众接收效果。
- Claim: 巴西医疗沟通存在实际障碍，不能据此宣称自动翻译有效。
  Source: [A4], abstract and Table 1.
  Support: 可用 Libras 的问卷记录一个都市区的自报沟通困难；既没有部署翻译系统，也没有测量系统改善效果。
- Claim: 社区参与、数据权与真实使用表现需进入试点评价。
  Source: [A2], main press-release paragraphs.
  Support: WFD 会议报告将这些问题列入评估 AI 项目的清单，要求聋人社区参与决策。

## Conflicts / unresolved issues

- 没有直接相互冲突的实测效果结果。调查意愿、创作实践和组织立场属于不同证据，不能合并为同一种产品需求率。
- 医疗需求大但期望高；“优先首发”不能仅根据困难严重程度决定。[A1][A4]
- 不使用中国论文背景中“90% 看不懂电视手语”的数字：这是引用其他调查，不是本研究 13 人直接结果；本线程未核查原调查。
- 本线程未找到足够的一手中国/巴西**零售服务台 AI 部署**对照证据，故仅形成试点假设。

## Leads discovered

- ASL STEM Wiki（EMNLP 2024）及 ASL Wiki（ASSETS 2022）：内容协作、术语和视频资源工具的原始研究，适合进一步查任务效率与署名/数据治理。
- 中国 CHI 2023 Community-Driven Information Accessibility: Online Sign Language Content Creation within d/Deaf Communities，DOI 10.1145/3544548.3581286：12 人访谈+视频分析，未在本线程正式纳入，以避免同作者/相近对象被误算为独立复现。
- “We do use it, but not how hearing people think”：DHH 使用文本 LLM 的研究，可用于后续个人表达辅助设计；本线程未完成全文核查。

## Gaps

- 中巴地区具体任务的频率、现有替代方式、参与者获得难度、支付主体与付费意愿未证实。
- 尚缺项目真实用户的识字程度、偏好输入法/语言/地方变体、对确认界面的理解与信任校准数据。
- Creator 方向需同时招募创作者与目标观众验证，不只满意度；样本需包含不同教育程度、区域和签语习惯。
- 最能改变优先级的证据是：同一真实任务下，相比图片/预录短语/文字输入，模型减少了多少步骤和编辑时间，是否引入未被发现的误义，以及用户是否自主愿意再次使用。

## Search and access log

- 3 batched web search calls, 8 query strings: original ASLT user perspectives; WFD AI statements; Brazilian Libras communication studies; Chinese Deaf communication technology. Followed by targeted opens/finds of four accepted originals.
- PMC returned CAPTCHA and one author PDF produced internal error; read the same Brazilian journal source and the Chinese paper's arXiv author HTML instead. No unsupported claim recovered from blocked pages.
- Current sources verified on 2026-10-11; study data dates explicitly distinguished from publication dates.
