# 认可来源注册表

核查日期：2026-10-11。只收录本报告实际使用的一手来源。编号与 draft.md 一致。研究笔记保留更详细的支持位置、评分及局限；本表的质量判断为人工筛选意见，不是量化真值。

[1] Chinese Deaf Online Content Creators | https://arxiv.org/html/2602.10235v1 | ORIGINAL_QUALITATIVE_STUDY | 2026-02-10; interviews 2022-2023
- 支持范围：A3; 13 selected Chinese creators; design proposals, not product efficacy.
- 质量与边界：Strong relevance and transparent limitations; not population representative.

[2] U.S. Deaf Community Perspectives on Automatic Sign Language Translation | https://danibragg.com/papers/Automatic_Sign_Language_Translation_Perspectives.pdf | ORIGINAL_SURVEY | 2023-10
- 支持范围：A1; 32 ASL users; U.S. stated preferences, not deployment.
- 质量与边界：Original methods; cannot extrapolate to CSL/Libras.

[3] WFD AI and Sign Languages | https://wfdeaf.org/artificial-intelligence-and-sign-languages/ | REPRESENTATIVE_ORGANIZATION_REPORT | 2025-08-12
- 支持范围：A2; participation, ownership, transparency and real-world performance.
- 质量与边界：Organizational position, not technical efficacy study.

[4] SWR Customer Report March 2025 | https://www.southwesternrailway.com/-/media/files/other/about-us/customer-report/customer-report-march-2025.pdf | DEPLOYER_REPORT | 2025-03
- 支持范围：B9; One Scan BSL deployment; pdf metadata title inaccurate relative to dated cover/content.
- 质量与边界：Deployment corroboration; no isolated BSL user efficacy measure.

[5] SignStudio | https://www.signapse.ai/signstudio-platform | VENDOR_FIRST_PARTY | undated; checked 2026-10-11
- 支持范围：B8; content production, quality check/human options; AI ASL package text conflicts.
- 质量与边界：Feature claims only; avoid advertised quality/precise coverage.

[6] SignStream API | https://www.signapse.ai/signstream-api | VENDOR_FIRST_PARTY | undated; checked 2026-10-11
- 支持范围：B7; text to BSL/ASL and no human grammatical review.
- 质量与边界：Good for stated workflow/limitations; not independent accuracy evidence.

[7] Hand Talk App | https://www.handtalk.me/en/app/ | VENDOR_FIRST_PARTY | undated; checked 2026-10-11
- 支持范围：B1; Portuguese-Libras and English-ASL, dictionary/speed/offline/learning.
- 质量与边界：Feature claims only; no camera recognition/CSL inference.

[8] Automatic Annotation Elaboration as Feedback to Sign Language Learners | https://aclanthology.org/2024.law-1.5.pdf | ORIGINAL_EXPERIMENT | 2024-03
- 支持范围：D1; expert annotations, 100 sentences, five learner evaluators; observed omissions/hallucinations.
- 质量与边界：Explicit small scope; no direct camera error recognition or learning gain.

[9] ELAN | https://archive.mpi.nl/tla/elan | OFFICIAL_SOFTWARE_DOC | undated; checked 2026-10-11
- 支持范围：D2; tiers/time alignment/EAF/GPL3.
- 质量与边界：Primary feature documentation, not demand evidence.

[10] FluentSigners-50 | https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0273649 | ORIGINAL_DATASET_STUDY | 2022-09-12
- 支持范围：D3; KRSL independent-signers/unseen-sentences protocols.
- 质量与边界：Supports evaluation design only; not transferable accuracy.

[11] Towards Online CSLR and Translation | https://aclanthology.org/2024.emnlp-main.619/ | ORIGINAL_MODEL_STUDY | 2024-11
- 支持范围：D4; dictionary-trained isolated model and sliding windows.
- 质量与边界：Online methodology; no local camera acceptance.

[12] Access to healthcare services | https://scielo.br/j/csp/a/f7xLDvF3cHXKJhzJBLTndLh/abstract/?ilang=en&lang=en | ORIGINAL_CROSS_SECTIONAL_STUDY | 2025
- 支持范围：A4; Brazilian Libras respondents; abstract/table1; no automatic-translation intervention.
- 质量与边界：Demand/obstacle evidence in one region; no technology effect.

[13] VLibras Terms of Use | https://www.vlibras.gov.br/privacy.html | OFFICIAL_TERMS | 2026-08-27
- 支持范围：B3; scope restrictions and commercial authorization, source/assets/services distinct.
- 质量与边界：Current stated terms; not a full legal opinion.

[14] VLibras translator-text-core LICENSE | https://github.com/spbgovbr-vlibras/vlibras-translator-text-core/blob/dev/LICENSE | PRIMARY_CODE_LICENSE | current dev; checked 2026-10-11
- 支持范围：B5; LGPLv3 for this component only.
- 质量与边界：Specific component license; not all hosted services/assets.

## 本地来源

[L1] 场景候选审查与 CSV：线程 C1。本轮重算来源文字分组，未重看全部视频。
[L2] 共同语义与字幕资源 JSON：线程 C3/C4。负责人及 C 线程核对当前文件；不是已审核双语资产。
[L3] caption_events.py / api.py / desktop.py：线程 C7/C8。现存代码与缺失编辑功能，未新增 UI 验收。
[L4] GPU / auto-captions 验证报告：线程 C9/C10。使用明确的样例计时边界，不扩大为长流效果。
[L5] session.py、CTC接口报告与当前模型manifest：线程 C6/C14及独立评审/负责人原文件检查。clip 路由、15秒/600帧输入上限与 gloss 语义/时间限制。
[L6] 新特征审计与 contracts.py：线程 C2/C5/C13。训练工作树与部署代码区别。
[L7] LLM中葡表达草案：线程 C11。设计建议，未实现。

本地文件的绝对路径和行定位在 draft.md 文末及 task-c.md。数字引用脚本仅检查上述14个网络来源；本地 L 编号由人工/独立评审检查。

## 未认可为结论依据的线索

SignAll 现行官网访问失败；Corpus/Signbank详细内容读取不足；旧供应商技术流程、旧VLibras API目录仅保留在线程笔记，不支持当前运行性。未引用宣传用户数、下载量、准确率或跨国市场规模。

