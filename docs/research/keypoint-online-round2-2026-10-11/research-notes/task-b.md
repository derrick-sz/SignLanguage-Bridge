---
task_id: b
role: Online temporal recognition analyst
objective: Verify what online training and stateful temporal inference require beyond a causal architecture name
status: complete
confidence: high for source mechanisms; medium for proposed project adaptation
sources_found: 4
acceptance_met: yes
retrieved: 2026-10-11
---

## Sources

[B1] Towards Online Continuous Sign Language Recognition and Translation | https://aclanthology.org/2024.emnlp-main.619.pdf | PRIMARY / EMNLP 2024, pp.11050–11067, November 12–16, 2024 | Aut:10 Rec:7 Rel:10 Dep:10 = 9.25. Read §§3.1–3.4, 4.1, 4.3, A.1–A.3 and Algorithm 2; mechanism support is stronger than project performance transfer.

[B2] Bai/Kolter/Koltun author implementation, TCN/TCN/tcn.py | https://github.com/locuslab/TCN/blob/8845f88f31def1e7ffccb8811ea966e9f58d9695/TCN/tcn.py | PRIMARY / AUTHOR SOURCE | last file commit 2018-04-16T21:26:54Z, obtained from GitHub API; inspected current master file and corresponding file history | Aut:10 Rec:5 Rel:10 Dep:10 = 8.75. Current master viewed at https://github.com/locuslab/TCN/blob/master/TCN/tcn.py .

[B3] torch.nn.GRU | https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html | OFFICIAL DOCUMENTATION / current main branch, retrieved 2026-10-11; this does not identify the project's installed PyTorch version | Aut:10 Rec:10 Rel:9 Dep:8 = 9.25.

[B4] EMNLP 2024 author evaluation implementation, Online/CSLR/prediction_slide.py | https://raw.githubusercontent.com/FangyunWei/SLRT/main/Online/CSLR/prediction_slide.py | PRIMARY / AUTHOR SOURCE / retrieved 2026-10-11 | Aut:10 Rec:8 Rel:10 Dep:10 = 9.5. Related entry-point instructions: https://raw.githubusercontent.com/FangyunWei/SLRT/main/Online/CSLR/README.md . The main-branch snapshot is mutable; file commit metadata lookup failed with a transport EOF, so no immutable SHA is claimed.

Scores are investigator judgments of source usefulness, not measured confidence probabilities. Four core sources retained; the newer-skeleton search did not produce an additional fully verified online mechanism worth enlarging the main recommendation.

## Search log

Six query strings, all executed 2026-10-11:

1. `"Towards Online Continuous Sign Language Recognition and Translation" window background dictionary`
2. `"skeleton" "online sign language recognition" causal streaming`
3. `locuslab TCN tcn.py Chomp1d causal two convolutions residual block`
4. `"Skeleton-based Online Sign Language Recognition using Monotonic Attention" pdf`
5. `2025 2026 "online" "sign language recognition" "skeleton" causal`
6. `site:scitepress.org "108994" "chunk" "encoder"`

Follow-up: read the EMNLP PDF, current author code, TCN source and GRU API; author GitHub tree API resolved a failing GitHub HTML page. The Japanese skeleton paper's publisher full PDF repeatedly returned 403/internal errors; only its indexed abstract was available. No third-party summary or benchmark score was substituted for the unavailable full mechanism.

## Findings (facts only)

- EMNLP 2024 用预训练 CSLR 和真实 gloss 序列强制对齐，得到词典伪边界；围绕片段生成窗口增强并加入背景，训练 ISLR 分类与 saliency loss。背景是词间过渡，不是任意未知手语。[B1, §3.1–3.2/A.1]
- 其识别器是 RGB 与 HRNet 63 点热图双流 S3D，而非纯坐标 TCN；在线步骤为滑窗分类、投票、去背景和消重。Algorithm 2 在背景后允许再次提交同词。W=16、S=1、B=7 属于该文设置。[B1, §4.1/A.2–A.3]
- 文中的 AL 是窗口相关的理论等待，WPL 是 V100 上窗口计算；二者不证明本项目摄像头、关键点提取、最终字幕确认的总耗时。翻译扩展另接训练过的 gloss-to-text 网络。[B1, §3.4/4.3]
- `prediction_slide.py` 的 `sliding_windows()` 接受完整 video/keypoint tensor，复制左右端帧、一次分配所有窗口；`evaluation_slide()` 批量调用模型，再对预测序列左右填充、投票、消重并去 blank。它提供数据集评估，不提供持续摄像头回调或逐层计算状态接口。[B4, lines 70–115, 206–250, 312–413]
- 原作者 TCN 每个 `TemporalBlock` 有两次相同 dilation 的卷积，配 Chomp、ReLU、Dropout，残差最多经 1×1 投影；第 l 块 dilation=2^l、stride=1。`forward(x)` 处理所给整段 tensor，没有 state 输入输出或卷积缓存。[B2]
- 单向 GRU 的递推依赖当前输入和前一步状态；API 接受 `h_0`、返回 `h_n`。省略 `h_0` 默认归零，`bidirectional` 默认 False；多层隐藏状态包括全部层。[B3]

## Analysis (project proposals and derivations)

### 1. 在线处理、因果性、状态复用与字幕提交是四个条件

由 [B1] 与 [B4] 推导：固定短窗口可以在视频尚未结束时运行，但仍重算重叠窗口；它既不等同逐帧缓存，也不等同零等待。公开脚本的窗口围绕其输出位置填充，部署若按窗口中心标时间，必须计入右半窗口已被观察的等待。可以把到达帧组织为尾窗，按最新观测时刻输出，仍需重新验证训练分布和提交延迟。

“词间背景”只处理训练覆盖的过渡；词表外动作、普通交谈动作和跟踪失败需要另外验证拒识。不能把未完成且可继续辨认的动作都标成背景。建议状态分清：无目标活动、正在观察但证据不足、词表外/无法可靠辨认、可确认候选。

架构不能解除语义不可辨认性。若两个词共享动作前缀，任何只读过去帧的模型都要等到分叉线索出现；允许早期候选改动，但确认字幕另有稳定性与纠错政策。

### 2. 因果 TCN 的感受野和缓存必须按实际卷积逐层计算

以下是从 [B2] 结构推导，不是论文给出的项目配置。假设 L 个残差块、每块两次核宽 k 的卷积、d_l=2^l、stride=1，无额外时间池化：

`R = 1 + sum_l 2(k-1)d_l = 1 + 2(k-1)(2^L-1)`。

若采用不同核宽或 dilation，应逐卷积求和，不能只数残差块。1×1 残差投影不增加时间范围。例：L=4、k=3 得 R=61 帧；若是 30Hz 等间隔输入，覆盖过去 60/30=2 秒。这个例子只是容量说明，不能解释成固定 2 秒输出等待，也不是推荐参数。

每一次卷积需缓存它自己的输入激活，保留 `(k-1)d_l` 个历史时间位置。第一卷积输入通道 C_in，第二为 C_out，则第 l 块的一种直接缓存方案约需 `(k-1)d_l(C_in+C_out)` 个标量。不能只缓存原始关键点，再声称所有中间层避免重算。按块拼入历史、只输出新时刻、滚动更新每层 ring buffer，才是所建议的状态增量实现。

R 是历史依赖，不是未来前视。流开始、reset 或长缺口后的历史不足，需要规定零/中性填充、热身标识和提交资格。所有内核、转置、复制和调度仍可能成为开销，不能从常数状态大小推断实测毫秒数。

### 3. 一个容易漏掉的分布差异：warm state 不等于每窗 cold reset

推导：持续编码 z_t 时，窗头的 z_t 仍依赖窗外过去；每个 W 帧窗口归零重编码，窗头依赖的是填充。因此“缓存编码 + 最近 W 帧池化”不必等于“W 帧窗口 reset + 编码 + 池化”。训练必须选择一种语义：持续上下文训练，或窗口训练时包含左侧 burn-in 上下文且只监督目标尾窗；否则缓存提速可能改变原模型输出。

单向 GRU 将上一块 `h_n` 作为下一块 `h_0` 可以保留前史；每窗省略 state 则是另一模型使用方式。当前人物、镜头或 source epoch 变化后保留旧 state 也会混入无关信息，应在协议里列出 reset 原因。不能用固定 clip 学到的末端隐状态直接宣称自然连续流已可靠。

### 4. 完整片段分类头可以廉价更新，但早期结果仍未获得意义保证

若编码器确为因果、归一化不依赖未来、eval 关闭随机 Dropout，过去的 z_t 不随新帧变化。对定义明确的有效帧权重 w_t，可以维护 `s_t=s_(t-1)+w_t*z_t`、`n_t=n_(t-1)+w_t`；`s_t/n_t` 与该前缀的同定义 masked mean 相等。此时不必每次重算所有历史编码或池化。

但只用完整片段标签训练的 softmax 头，没有被验证为“前缀已可提交词义”的概率。对正在开始、两个词共用前缀、片段中间随机起点、重叠两个动作、回到中性和无目标动作，应单独训练/评估。不能将所有短前缀硬贴最终类别再以高 softmax 当作可提前辨认。

这是一个分层验证问题：先证明全前缀与增量计算在相同 reset/上下文条件下数值等价；再证明未知/背景/未完成不会误提交；最后在独立签者自然流验证早期候选与最终提交。数值等价只回答实现，没有回答识别正确。

### 5. 时间统计、缺失值和 mask 可以破坏“看起来因果”的网络

以下是数学与接口设计检查项：若归一化统计跨整个 T 求均值/方差，过去输出会随追加帧变化；即使卷积左填充，整体仍不满足前缀不变性。per-frame 仅跨特征通道的归一化可保留该性质；训练时跨批次/时间的统计与推理冻结统计须分别审计。LayerNorm 的 normalized_shape 若包含 T，与仅包含 C 的语义不同。

序列 padding mask、每个点的观测 mask、活动阶段不是同一量。仅在最后池化排除 padding，不能消除先前编码中 sentinel 或错误归一化对特征的影响。不能将旧 -999 哨兵直接参与角度、速度或标准化；中性填充值加点观测/缺失年龄等特征的方案需在同一提取协议上重新训练。

因果速度只能用当前与过去有效观测及实际 Δt；若跟踪输入因忙碌丢帧，不能仍假设每 step 等于 1/FPS。长时间失踪、重新捕获左右手以及相机切换应测试 state 和归一化基准恢复。手部缺失不自动意味着整帧无效或没有手语。

### 6. 最小模型比较与补数据路线

建议比较规模先限制为 A 线程的多模板/子序列 DTW、一个小因果 TCN、一个单向 GRU。每者使用同版本关键点、同词表、同独立签者切分、同输出资格和同未知集；同时记录抽取、排队、模型、提交、渲染耗时。T/FPS、核宽/层数/隐藏维度应由实际动作持续时间、混淆和测量预算选择，暂不固定论文数字。

数据分两阶段：

- 完成动作的孤立词/审核短句标签可以先支持受限分类器和模板比较。item/表达 ID 可以是分类目标；这并不变成逐词 gloss 序列。
- 持续流需新录自然起止、等待/过渡、同词重复、词表外动作、半动作及相邻词。记录动作边界和可区分证据点，训练与部署一致的尾窗或有 burn-in 的状态序列；需要早期输出时，加入前缀和阶段监督/选择性预测。对语义仍不确定的前缀，用标注与输出资格控制，不把等待状态伪装成背景。若未来做 CSLR/CTC，还需对应语言的真实 gloss 序列及必要的对齐审核。[B1 只支持已具备 gloss 标签的词典构建机制]

每种新模型的反证实验：改变未来帧不影响过去编码；整段前缀/单帧/多种 chunk 分割在 eval 下等价；明确 cold/warm 模式不混比；新增窗口不可错误重复提交，分开的同词动作不可被永久去重；unknown 与提取失败不会输出稳定已知词。确认策略要测错误率随等待/覆盖率的变化，不能以“连续 K 次相同 argmax”单独完成验收。

升级条件属于分析建议：若错误集中在关节关系而观测质量足够，先做空间特征消融，证明空间图编码有增益再引入 GCN；若错误集中在遮挡、精细手形/接触、口形等当前表示缺失的线索，先补对应观测，再做有预算的手部/脸部 RGB 融合。不是因为别的数据集离线分数高便添加多流大模型。

## Support snippets / paraphrases for top claims

- Claim: 不可将 RGB 双流在线滑窗文献写成已验证的骨架坐标缓存网络。
  Source: B1 A.3；B4 `sliding_windows()` 与 `evaluation_slide()`。
  Support: 论文列出两套 S3D 与 63 点热图；发布入口先生成整段的窗口 batch，再逐批 forward。
- Claim: 作者 TCN 的感受野应包含每块两次卷积，缓存不是其原有接口。
  Source: B2 `TemporalBlock` / `TemporalConvNet`。
  Support: conv1、conv2 均用相同 dilation；forward 只收 x，类中无缓存参数。
- Claim: GRU 不显式携带状态，重复窗口调用会默认从零开始。
  Source: B3 Inputs/Outputs。
  Support: h_0 缺省零，h_n 返回末状态；串联新块必须显式传递全部层 state。

## Conflicts / unresolved issues

- B1 的 online 指滑窗到达后识别；B4 是按数据集整段张量评估的实现。它们描述不同层次的能力。论文机制可设计为接收流，但公开评估入口没有证明摄像头长时持续性或系统端到端提交时间。
- background 类与 unknown 拒识、正常等待与未完成词、同词重复与重复窗口没有可互换的定义。需要明确项目的事件标注与输出契约。
- 前缀均值数值等价依赖 encoder、normalization、mask、reset、时间采样都一致；这些假设应由 C 线程当前仓库审计验证，不能从文件名 causal 推出。

## Leads discovered

- Takayama et al., *Skeleton-based Online Sign Language Recognition using Monotonic Attention*, VISAPP 2022, DOI 10.5220/0010899400003124. Publisher URL: https://www.scitepress.org/Papers/2022/108994/108994.pdf ; indexed abstract describes STGCN+RNN and HMA/MoChA/MILK on JSL. Full-text access failed, so chunk size, encoder future access, cache handling and camera latency were not used as accepted claims. Follow later if moving from bounded isolated-word recognition to learned continuous sequence emission.
- 2025 LGF-SLR search hit describes isolated benchmark datasets; its publication status phrase “early online” is not evidence of streaming inference. No online-state recommendation or accuracy transfer was taken from it.

## Gaps

- No current-project causal cache, uni-GRU, camera stream, independent-signing-person accuracy, calibrated OOV threshold or action-to-confirmed-caption time was measured in this research thread.
- The data needed to supervise learnable early recognition/commit is not established merely by whole-clip labels. Existing natural language text is not automatically gloss.
- Whether a small skeleton temporal head is faster on CPU or GPU depends on measured transfer/call overhead as well as compute; GPU's prior whole-RGB-model benefit does not settle this choice.
- Strongest uncertainty reduction: freeze the extractor/timestamps/masks, record representative independent-person natural streams, and compare the three models with identical rejection and commit policy. Report event precision/recall, false commits per unit time, repeat preservation, coverage, action-to-confirmed-output delay, and each pipeline stage's timing rather than a single offline accuracy.
