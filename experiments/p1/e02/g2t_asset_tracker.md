# G2-B 外部资产跟踪：SLRT CSL-Daily G2T/SLT 权重是否已发布

- 任务：task-13（G2-B 外部 BLOCKED 跟踪）
- 检查日期：**2026-10-27**（本机网络实测，非凭记忆）
- 执行约束：只用 web_search/web_fetch + 本机 HTTP HEAD/GET 与只读文件检查；**未下载任何权重、未跑 GPU/LLM、未改动 slbridge/ 与既有产物、未 commit**
- 结论一句话：**CSL-Daily G2T（gloss→text）权重未在可追溯渠道发布 → B 分支仍为外部 BLOCKED。**
  另有两个相关发现：① CSL-Daily **SingleStream-SLT（S2T，video→sentence）checkpoint 实际上已在作者 Google Drive 替换包中**（本地也已有，SHA256 见 §5）——但它不是 B 分支需要的 gloss→text；
  ② 官方文档列出的 **TwoStream-SLT CSL-Daily 三个翻译网络 checkpoint 链接（SharePoint）全部 404**，作者 2026-08 的 Google Drive 替换包中也没有这三项。

## 1. G2-B 到底缺什么（与"已有什么"的映射）

| G2-B 所需 | 作用 | 当前状态 |
|---|---|---|
| CSL-Daily **G2T**（Gloss→Text）权重 | 把 SLRT 预测的 gloss 序列转成中文句子（B 分支核心） | **NOT_FOUND**（未发布；见 §3 #8/#9/#12） |
| 作者剪枝 **mBART_zh**（含 gloss2ids/gloss_embeddings/old2new_vocab） | G2T 初始化/词表；官方 `pretrained_models/mBart_zh` | **NOT_AVAILABLE**（官方链接 404；替换 Drive 里 `mBart_zh_t2g` 是 T2G 反方向） |
| CSL-Daily **TwoStream-SLT** 翻译网络（video/keypoint/joint） | 官方 SLT 上限参照（不是 B 的 gloss→text 输入） | **NOT_AVAILABLE**（文档链接 404；替换 Drive 无） |
| CSL-Daily **SingleStream-SLT（S2T）** | 官方 SLT 基线（video→sentence） | **FOUND**（作者 Drive `ckpts.zip` 内含；本地已有；不是 B 所需） |
| Phoenix-2014T G2T | 参考实现/调试 | FOUND（作者 Drive `ckpts.zip` 内含，**数据集不对**，不能当 CSL-Daily 用；见 §5） |
| SLRT CSLR cslr_best.ckpt | gloss 识别（G2 的输入来源） | 本地已有（R12），非本任务缺口 |

## 2. 结论状态

| 状态 | 内容 |
|---|---|
| **B 分支** | **外部 BLOCKED（结论不变）**：CSL-Daily G2T 权重与作者 mBART_zh 均无公开可下载来源 |
| 可联系渠道 | ① 在 `FangyunWei/SLRT` 新建 issue 或回复 #107/#106（@2000ZRL）；② 论文通讯作者邮箱（需从论文 PDF/author 主页人工核对，本文不猜） |
| 替代路线（不依赖外部发布） | ① 用已到手的 CSL-Daily 句级标注 + base mBART（+ R12 已生成的 gloss 侧资产）**自行训练 G2T**（需 GPU 预算与训练/评测协议；本轮不跑）；② 确定性 Gloss 后处理（规则/模板）替代学习式 G2T；③ 人工语义评审（Round-1 包已就绪） |
| 复查触发 | 作者新 issue 回复/新 commit/push、Google Drive 文件夹新增文件、HF 新模型；建议每次外部实验前复查一次 |

## 3. 检查清单（逐条链接）

> HTTP 用本机 `Invoke-WebRequest -Method Head` / API GET 实测（2026-10-27）；`404` 表示链接不可达（SharePoint 原 OneDrive 已被作者确认清空）。

| # | 检查项 | 链接 / 来源 | 检查方式与日期 | HTTP/可达性 | 结论 | 下一步 |
|---|---|---|---|---|---|---|
| 1 | 作者 GitHub 仓库主页 / 许可证 / 活跃度 | https://github.com/FangyunWei/SLRT | GitHub API GET，2026-10-27 | API 200；`license=null`；`pushed_at=2025-02-21`；open issues 62 | 可达但**无 LICENSE**；近一年半无代码更新 | 若找到权重需向作者确认许可 |
| 2 | GitHub Releases（是否有发布权重） | https://api.github.com/repos/FangyunWei/SLRT/releases | API GET，2026-10-27 | 200，`[]`（空） | 无 release 资产 | — |
| 3 | 主 README（是否给模型链接） | https://github.com/FangyunWei/SLRT/blob/main/README.md | 原文抓取，2026-10-27 | 200 | 只列论文+代码目录，**无权重下载入口** | — |
| 4 | TwoStream-SLT 文档（CSL-Daily SLT 权重入口） | https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/TwoStream-SLT.md | 原文抓取 + 链接 HEAD，2026-10-27 | 文档 200；所列 SharePoint 链接 HEAD → **404** | 文档**确实列有 CSL-Daily SLT**（video/keypoint/joint，B4 25.79）与 Phoenix；但**链接全部 404** | 等作者重传；或联系作者 |
| 5 | SingleStream-SLT 文档（含各阶段 ckpt 文件夹） | https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/docs/SingleStream-SLT.md | 同上，2026-10-27 | 文档 200；2 个 ckpt 链接 + "各阶段 ckpt"文件夹均 **404** | 官方曾提供 CSL-Daily SingleStream-SLT（B4 24.09）与各阶段 ckpt；现**旧链接 404** | 作者 Drive 替换包只含 SingleStream-S2T（见 #10/#15），**无 G2T 阶段 ckpt** |
| 6 | pretrained_models（mBART_zh / s3ds_glosscls_ckpt） | TwoStreamNetwork/README.md 内 SharePoint 链接 | HEAD，2026-10-27 | **404** | 作者剪枝 mBART_zh 无法从官方文档获取 | 需作者重传；或使用 R12 全词表 fallback（非等价） |
| 7 | Online/SLT README（在线 G2T 权重入口） | https://github.com/FangyunWei/SLRT/blob/main/Online/SLT/README.md | 原文抓取，2026-10-27 | 200 | **没有任何 checkpoint 下载链接**；只给训练/测试命令 | `Online/SLT/configs/g2t_wait2_csl.yaml` 只引用 `../../pretrained_models/mBart_zh/`（训练初始化），无 ckpt 路径 |
| 8 | Online/SLT 的 CSL-Daily G2T 配置 | https://raw.githubusercontent.com/FangyunWei/SLRT/main/Online/SLT/configs/g2t_wait2_csl.yaml（及 `ts-16_g2t_wait2_csl.yaml`） | 原文抓取，2026-10-27 | 200 | 配置存在（G2T、wait-k=2、char 级、mBART_zh） | 只有代码/配置，**无发布权重**；需自行训练或等作者发布 |
| 9 | Issue #17「csl-daily_g2t ckpt seems missing」（CSL-Daily G2T 缺失的最早记录） | https://github.com/FangyunWei/SLRT/issues/17 | API GET + comments，2026-10-27 | 200；3 条评论，**无作者回复**；2023-06 关闭 | 历史证据：**CSL-Daily G2T ckpt 确实缺失**；提问者最终"自己训练"（2024-07 评论） | 说明这是长期未发布资产，不是本轮网络问题 |
| 10 | Issue #107「Broken SharePoint links…」+ 作者回复 | https://github.com/FangyunWei/SLRT/issues/107 | API GET + comments，2026-10-27 | 200；作者（collaborator `2000ZRL`）2026-08-28 回复 | 作者：**HKUST OneDrive 已被清空**，把"some copies"上传到自己的 Google Drive（链接见 #11） | 以 #11 实际文件列表为准 |
| 11 | 作者替换 Google Drive 文件夹（2026-08 更新） | https://drive.google.com/drive/folders/1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa | HEAD + `embeddedfolderview` 列表，2026-10-27 | HEAD **200**；列表页 200 | 顶层仅 6 个文件：`ckpts.zip`、`data.zip`、`EMNLP24_Online.zip`、`mBart_de_t2g.zip`、`mBart_zh_t2g.zip`、`TA.zip`。**无 SLT/G2T 命名的权重包**；`mBart_zh_t2g` 是 Text2Gloss（反方向，R12 已核） | 已足够确认替换包不含 CSL-Daily G2T；如作者后续新增文件需复查 |
| 12 | Issue #54「domain-aware ckpts for Phoenix2014T and CSL daily」 | https://github.com/FangyunWei/SLRT/issues/54 | API GET，2026-10-27 | 200；**0 条评论，2024-04 起 open** | 明确请求 CSL-Daily/Phoenix SLT ckpt，**作者未回应** | 可在此 issue 追加请求（不代发） |
| 13 | Issue #106「404 FILE NOT FOUND」 | https://github.com/FangyunWei/SLRT/issues/106 | API GET + comments，2026-10-27 | 200；作者 2026-07-08 同上 Drive 回复 | 与 #107 同结论：旧链接失效，替换包有限 | — |
| 14 | Issue #96「CSL-Daily keypoints/gloss_logits 在哪」 | https://github.com/FangyunWei/SLRT/issues/96 | API GET + comments，2026-10-27 | 200；1 条评论（他人），**无作者回复** | 在线 SLT 所需 gloss_logits 也未发布 | 说明在线 G2T 链路依赖的中间产物同样缺 |
| 15 | 作者 Drive 包在本地的实际内容（只读核对） | `F:\model_weigth\ckpts\TwoStreamSLT\**` | 只读 `Get-ChildItem`，2026-10-27 | 本地可读 | 含：`SingleStream/{csl-daily_s2g, csl-daily_s2t, phoenix-2014t_g2t, phoenix-2014t_s2g, phoenix-2014t_s2t}`、`TwoStream/{csl-daily_video, csl-daily_keypoint, csl-daily_joint, csl-daily_s2g, phoenix-*}`；**没有任何 `csl-daily_g2t`**；`SingleStream/phoenix-2014t_g2t` 是 Phoenix 而非 CSL-Daily | **CSL-Daily G2T 不存在于作者替换包** | 与 #11 列表一致 |
| 16 | HuggingFace 模型检索（官方或第三方是否转载） | `huggingface.co/api/models?search=...`：`SLRT` / `TwoStream` / `csl-daily` / `sign language translation` / `mBART zh` | API GET，2026-10-27 | 200 | **无** SLRT 官方/转载的 CSL-Daily G2T/SLT 权重（检索到的 `waternew/slrtp*` 与 `rao0/SignRAG-...` 均与本资产无关） | 若有新上传需复查；不要下载第三方非官方权重 |
| 17 | 配置引用的 `SignLanguageTranslation_Baseline` 仓库（G2T 初始化来源 `gloss2text_cp_2000_bleu10`） | `TwoStreamNetwork/experiments/configs/TwoStream/csl-daily_s2t_ensemble.yaml` 内路径；GitHub 搜索 | 原文抓取 + GitHub search API，2026-10-27 | 配置文件 200；`api.github.com/search/repositories?q=SignLanguageTranslation_Baseline` → **0 结果**；`ChenYutongTHU/...`、`FangyunWei/...` → **404** | 该 G2T 初始化权重来源仓库**不公开/已删除**；issue #21 也显示用户找不到 `gloss2text_cp_2000_bleu10`（作者未给出下载链接） | 需作者提供；或自行用 base mBART 训练 |
| 18 | 论文（EMNLP 2024, arXiv 2401.05336）对资产可用性的表述 | https://ar5iv.labs.arxiv.org/html/2401.05336 | 抓取正文，2026-10-27 | 200 | 论文写 "Code and models are available at https://github.com/FangyunWei/SLRT" | 官方声明存在，但**具体 CSL-Daily G2T/SLT 资产链接已失效或从未发布**；引用时需注明 |

## 4. 关键证据摘录（原文）

- 作者 2026-08-28 在 issue #107 的回复：
  > "The onedrive of my phd university has been out of dated, and all data on it were purged. I have uploaded some copies to my own google drive. Please check whether there are the files you need …"
  （Google Drive 链接：`https://drive.google.com/drive/folders/1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa`）
- 该 Drive 文件夹 2026-10-27 可见文件名（完整、无省略）：`ckpts.zip`、`data.zip`、`EMNLP24_Online.zip`、`mBart_de_t2g.zip`、`mBart_zh_t2g.zip`、`TA.zip`。
- Issue #17（2023-06-10，标题 *csl-daily_g2t ckpt seems missing*）→ 作者从未回复；2024-07 另一用户回复 "Hi, I have trained it myself."
- issue #54（2024-04，请求 CSL-Daily/Phoenix domain-aware ckpt）→ 0 评论、open 至今。
- `Online/SLT/configs/g2t_wait2_csl.yaml` 只配置 `pretrained_model_name_or_path: ../../pretrained_models/mBart_zh/`（初始化），没有 `from_ckpt` 指向发布 G2T 权重。

## 5. FOUND 的资产详情（注意：不是 B 分支所需的 G2T）

| 资产 | 数据集/任务 | 本地路径（来自作者 Drive `ckpts.zip`） | 大小 (B) | SHA256 | 许可 |
|---|---|---|---|---|---|
| SingleStream-SLT（**S2T**：video→sentence） | **CSL-Daily** / SLT 基线（论文 B4 24.09） | `F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2t\ckpts\best.ckpt` | 1,503,041,809 | `b6bdf7c2d8babe4d8ced724783f966f101d27468fd0494be864e73e03e8bd66c` | **仓库无 LICENSE（license=null）**；用于研究需向作者确认 |
| SingleStream-G2T（Gloss→Text） | **Phoenix-2014T** / G2T | `F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\phoenix-2014t_g2t\ckpts\best.ckpt` | 1,434,230,775 | `51e5de5c18f7dad651e6a636a1f439856e8041dbb1fc0c17838b9f0c62cedc7e` | 同上；**数据集不对，不能用于 CSL-Daily B** |

- SHA256 获取方式：`hashlib.sha256` 分块读取本地已存在文件（未下载；命令见本文件 §8 复核命令）。
- 预期入口：SingleStream-SLT 需 `TwoStreamNetwork` 的 SingleStream 评测流程（`prediction.py --config experiments/configs/SingleStream/csl-daily_s2t.yaml`），输入是 CSL-Daily **视频/预提取特征**，不是 SLRT gloss 序列 → **不能直接充当 B**。
- 若未来找到 **CSL-Daily G2T**，预期入口为以下之一（按官方代码）：
  1. `TwoStreamNetwork/prediction.py --config experiments/configs/SingleStream/csl-daily_g2t.yaml`（离线 G2T，需 `pretrained_models/mBart_zh`）；
  2. `Online/SLT/prediction.py --config Online/SLT/configs/g2t_wait2_csl.yaml`（在线 wait-k G2T，输入为 online CSLR 的 gloss 预测文件 `csl_pred.dev/.test`）。
- **最小验证步骤（仅在获得确凿链接后执行，本轮不跑）**：
  1. 下载到独立目录（如 `F:\model_weigth\uploads\`），先记录文件大小/SHA256，与作者给出值核对；
  2. 只读 `torch.load(..., map_location="cpu")` 检查 state_dict 键数与关键张量（翻译网络 ~数百键；gloss embedding 行数应为 2,000 + 特殊 token；文本词表剪枝映射 `old2new_vocab.pkl` 应与 mBART_zh 匹配）；
  3. 用官方 config 在 3–5 条 pilot 样本上跑 gloss→text，**GT 只在离线评测侧 join**，记录输出与版本；
  4. 与全词表 fallback / 确定性 join 的基线对比，确认不是"随机权重继续跑"（检查 loss/输出非退化）；
  5. 任何大文件运行前先单独确认许可与算力预算（GPU 由 E01/E04 协调）。

## 6. 替代路线（不依赖作者发布）

1. **自行训练 CSL-Daily G2T**：数据已到手（`F:\model_weigth\data\csl-daily` 的 `csl.train/dev/test`；R12 已生成 CSL gloss 侧资产、base mBART 与全词表 fallback）→ 可训练 G2T，但需 GPU 预算 + 训练/评测协议，且**不能称"作者发布权重复现"**。
2. **确定性 Gloss 后处理**：用规则/模板把 SLRT gloss 转成句子（确定性强、可审计，质量上限有限）。
3. **人工语义评审**：Round-1 评审包（`experiments/p1/e06/human_review/`）已就绪，可直接支撑"严重语义错误"证据，不依赖 G2T 权重。

## 7. 联系渠道（模板；**本轮未发送**）

- 渠道 A：GitHub issue（仓库 `FangyunWei/SLRT`，可新开或回复 #107/#106，@2000ZRL 为作者 collaborator）。
  Issue 模板：
  > 标题：Request for CSL-Daily G2T / TwoStream-SLT checkpoints after the 2026-08 Google Drive update
  > 正文：We use the official Online/SLT G2T configs (`g2t_wait2_csl.yaml`, `ts-16_g2t_wait2_csl.yaml`) and would like to reproduce the CSL-Daily SLT path. The SharePoint links in `TwoStreamNetwork/docs/TwoStream-SLT.md` and the "checkpoints trained by each stage" folder now return 404, and the 2026-08 Google Drive replacement contains only `ckpts.zip`/`data.zip`/`EMNLP24_Online.zip`/`mBart_de_t2g.zip`/`mBart_zh_t2g.zip`/`TA.zip`. Could you please confirm whether a CSL-Daily **G2T (gloss→text)** checkpoint and/or the TwoStream-SLT CSL-Daily (video/keypoint/joint) translation-network checkpoints are available? If the author-pruned mBART_zh is required, please also share it (or confirm that the full-vocab fallback is acceptable). We will record hash/license and run only on the official protocol.
- 渠道 B：论文通讯作者邮箱（EMNLP 2024 / arXiv 2401.05336 作者页；需人工从论文/主页核对地址后再发，本文不提供未核对邮箱）。
  Email 模板要点同 A，附 GitHub issue 链接。

## 8. 复核命令（本机可重复）

```powershell
# 1) 官方文档链接可达性（HEAD；2026-10-27 全部 404）
$urls = @(
  "https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EmSUuTojKAZIpy90aA75s00BBOrlZyhkvFBNsbibtgx5mg?e=0MPPEn",
  "https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EpyVs_YNq2NKrxn0FPJJWF4BtS7O1wTrOEa2ZvMwT2OU-g?e=u0sQab",
  "https://hkustconnect-my.sharepoint.com/:f:/g/personal/rzuo_connect_ust_hk/EolDU7j15xROncGg8QqLpkABn9mFEfriS0owcyr048nyXg?e=jHdxlg"
)
foreach ($u in $urls) { try { (Invoke-WebRequest -Uri $u -Method Head -TimeoutSec 20 -UseBasicParsing).StatusCode } catch { $_.Exception.Message } }

# 2) 作者替换 Drive 文件夹顶层列表
(Invoke-WebRequest "https://drive.google.com/embeddedfolderview?id=1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa#list" -UseBasicParsing).Content

# 3) 本地已有 SLT/S2T 资产核对（只读，不下载）
$env:PYTHONPATH='E:\githubproject\SignLanguage-Bridge'
.\.conda-env\python.exe -c "import hashlib,os; p=r'F:\model_weigth\ckpts\TwoStreamSLT\SingleStream\csl-daily_s2t\ckpts\best.ckpt'; h=hashlib.sha256(); f=open(p,'rb'); [h.update(c) for c in iter(lambda: f.read(8*1024*1024), b'')]; f.close(); print(os.path.getsize(p), h.hexdigest())"
```

## 9. 边界与免责

1. 本文件是**外部资产核查记录**，不是权重使用许可；`SLRT` 仓库 `license=null`，任何使用/再分发前需作者书面确认。
2. 未下载任何权重、未运行 GPU/LLM、未修改 `slbridge/` 或既有实验产物；B 分支仍按 R11/R12/R18 与 E02 报告标记 **BLOCKED**，不得因本文件出现"FOUND"字样而误读为 G2T 可用。
3. 链接可达性受本机网络与站点状态影响；日期 2026-10-27 的结果只代表当日。
4. HTTP 404 是 Sharepoint 原 OneDrive 清空的直接表现；作者已确认（#107/#106），非本机网络问题。
5. 若未来获权重：先登记来源/日期/大小/SHA256/许可，再按 §5 最小验证步骤执行；不得跳过许可与算力协调。

## 10. 下一步（建议）

- [ ] 由 Lead/用户决定是否按 §7 模板向作者发 issue（本轮未发）。
- [ ] 若两周内无作者回应：将 B 分支维持 BLOCKED，优先"自行训练 G2T（替代路线 1）"或确定性后处理（路线 2），人评（路线 3）不依赖该资产。
- [ ] 每次外部实验前复查：GitHub issue #107/#106 新回复、仓库新 push、Drive 文件夹是否新增文件、HF 是否有官方上传。
