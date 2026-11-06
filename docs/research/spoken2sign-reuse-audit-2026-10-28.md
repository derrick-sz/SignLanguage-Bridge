# Spoken2Sign 免训练复用核验（2026-10-28）

## 后续获取结果（2026-10-28）

**`E:/githubproject/SignLanguage-Bridge/models` 专项盘点（2026-10-28）**：`models/external/SLRT_Online/pretrained_models/mBart/pytorch_model.bin` 和 `mBART_zh_fullvocab/pytorch_model.bin` 存在、均为 2,444,517,405 字节且 `os.path.samefile` 为真；两个配置 `vocab_size=250027`。作者 `mBart_zh_t2g.zip` 的 config 为 6308，不能因文件同名而视为等价或直接替换。本次只读文件名、配置和物理文件身份，未加载权重或做兼容测试。另有 HRNet WholeBody（254,976,311 B）、Faster R-CNN（165,671,301 B）、RTMW/YOLOX ONNX、Uni-Sign CSL-Daily 句子权重、SLRT CSL-Daily Gloss 权重，可分别服务 2D 姿态/人体检测和识别，不能直接替代 SMPL-X 拟合模型或生成器。对 `models` 递归按 smplx、smpler、hamer、mano、vposer、connector、gloss2items 文件名与常见权重/数据扩展名筛选，未找到目标权重；源码中的配置/测试数据不等于已取得对应模型，非全机器不存在的证明。上游 Spoken2Sign 源码另带 `vposer/snapshots/TR00_E096.pt`（2,702,962 B），仅已确认文件存在，未验证完整依赖、授权或模型加载。

**用户指正后的更正（2026-10-28）**：本机 `E:/githubproject/SignLanguage-Bridge/artifacts/slrt-gdrive-20261025/mBart_zh_t2g/` 已有 `gloss_embeddings.bin`（1,059,020,103 B，SHA256 `df3530c07ae5f300c90e732cc91cf1ad2ba4c98d6800312cf7bd199dd156b319`）和 `text_embeddings.bin`（25,838,315 B，SHA256 `d287f6c6886a0ca1ee33481a62b1cb02a93f7e249ebb4a1e9d257e88a1a2d10c`）。两者文件大小及 CRC32 与作者 ZIP 的中央目录一致；CRC32 不是抗篡改证明。此前仅扫描 F/G 目录遗漏这两项，对“本机未发现 T2G 文件”的宽泛表述应以本更正为准。该目录只含这两个文件，未见 `pytorch_model.bin`、配置、分词器和任务 `best.ckpt`；配置/分词器现已从作者 ZIP 取到 G 盘，完整 ZIP 继续下载。是否已训练完成仍需核验独立任务 ckpt；无模型加载或推理证据。

- **已获取源码**：官方 SLRT 的 Spoken2Sign 稀疏检出位于 `G:/Archive/Zenodo/Spoken2Sign/SLRT-source/Spoken2Sign`，上游 HEAD `38a4f7b00da7a858d59b7fabe5093876a84db8e0`。仅源码获取和静态阅读；未运行。仓库所列 `smplifyx/` 路径在该提交不存在，相关脚本位于 Spoken2Sign 根目录，运行方式需适配及许可核验。
- **作者包已确定存在，完整包下载中**：作者在 [issue #107](https://github.com/FangyunWei/SLRT/issues/107) 提供的 [替代 Drive](https://drive.google.com/drive/folders/1U-BK7R-fMLmkSDq2M7GpHT9J4aKK9TFa) 中 `mBart_zh_t2g.zip`，文件 ID `1IWPRmZaLfnP3PZaKDYYovjDe5qBXeBWY`，远端长度 2,522,783,797 字节。范围请求读取 ZIP 中央目录确认 8 项：config、两个词表 pkl、两个 sentencepiece、pytorch_model.bin、gloss_embeddings.bin、text_embeddings.bin。
- **已取得五个小文件**：`G:/Archive/Zenodo/Spoken2Sign/mBart_zh_t2g_verified_small/` 下的 config、两个词表及两个 sentencepiece；通过 ZIP CRC 校验并计算本地 SHA256，远端没有提供已核对的官方 SHA。config vocab_size=6308。未加载 pickle 或运行模型。
- **任务检查点仍未确认**：上游 `text2gloss/prediction.py` 除基础模型外查找 `{dataset}_best.ckpt`；如果没有会记录缺失然后继续执行。作者 mBart 包无独立 ckpt，不能以能够加载基础模型冒充完成 Text2Gloss 任务推理。
- **大文件可续传**：部分文件为 `G:/Archive/Zenodo/Spoken2Sign/mBart_zh_t2g.zip.part`；后台脚本 `artifacts/animation-acceptance/download_spoken2sign_t2g.py` 使用 HTTP Range 续传，完成后检查长度、SHA256 和 ZIP CRC，再改名为 `mBart_zh_t2g.zip`。日志与 `download-status.json` 在同一 G 盘目录，`complete` 才代表本机完整下载；自算 SHA 不是官方来源认证。状态初期可能尚未写入，直接检查 `.part` 大小与进程。下载失败时保留部分文件。
- **3D 词典/connector 仍未取得**：旧 SharePoint 所属 OneDrive 被作者确认清空；作者替代目录本次未列出这两个名称。上游 [issue #104](https://github.com/FangyunWei/SLRT/issues/104) 有人请求 Connector 文件但未见回复；不能断言作者没有其他可用发布。未经明确授权不向作者发送消息。
- **SMPL-X/MANO**：应从[官方门户](https://smpl-x.is.tue.mpg.de/)查看许可及获取条件；本轮未代用户注册或接受许可，未取得模型。原视频拟合、真人质量复核和加载测试均 NOT_RUN。

## 结论

架构方向成立：离线用真人视频制作 CSL 动作资产，在线做中文规划、资产检索和播放。中方 CSL 与巴方 Libras 独立；应用接口及资产格式协商统一。本次只核验材料，不训练模型，也未实现或运行 Spoken2Sign。

| 项目 | 结果和证据范围 |
|---|---|
| 上游方法 | [论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/6499_ECCV_2024_paper.php)与[官方 README](https://raw.githubusercontent.com/FangyunWei/SLRT/main/Spoken2Sign/README.md)确认 Text2Gloss、3D 词典、Sign Connector 与渲染；源码里有预训练模型依赖，不是开箱即用。 |
| 本机 Spoken2Sign 源码 | `E:/githubproject/SignLanguage-Bridge/models/external/SLRT_Online` 下本次只看到 `Online/CSLR`、`pretrained_models`；在该目录及 `models/external` 定向按脚本名扫描未找到 `motion_gen.py`/`gen_segment.py`。不能推断整台机器绝对不存在。 |
| CSL-Daily 原视频 | `G:/Archive/Zenodo/csl-daliy` 当前存在；历史 R21 清点 19,528 个 MP4、P1 1,086 条有同名文件；未逐文件核对迁移内容哈希。 |
| 句级文本和 Gloss | `F:/model_weigth/data/csl-daily/csl-daily.{train,dev,test}` 与 `csl-daily-keypoints.pkl` 本次确认文件存在；2026-10-25 记录句级标注有 `name/gloss/text/num_frames`。需在新流程逐条确认 ID 与视频匹配及合法许可。句级 Gloss 不等于词级精确帧边界。 |
| TwoStream-S2G 权重 | 本次 `F:/model_weigth/ckpts/TwoStreamSLT/TwoStream/csl-daily_s2g/ckpts/best.ckpt` 存在，504,360,673 B；可研究已有权重的强制对齐。未在 Spoken2Sign 分段流程运行。 |
| Text2Gloss | 既往 R12 记录作者 Drive 替代包 `mBart_zh_t2g.zip` 包含词表配置等文本侧资产；不等于已确认完整 Text2Gloss 任务检查点。本次在 F:/model_weigth 定向搜索未发现 t2g 目录/权重；未下载、未加载、未推理。 |
| 现成 3D 词典 | 官方 README 有下载入口；本次在 G:/Archive/Zenodo 与 F:/model_weigth 按文件名定向搜索未发现可确认的 3D 词典。官方链接在本次网页工具不可访问，原因未知，不证明源数据不存在。 |
| Sign Connector | 上游 `motion_gen.py` 读取 `data/connector.pth`；在上述本地搜索范围未找到可确认权重。没有运行现成 connector；用户不希望训练新模型。 |
| 现有 100 段 | 项目工作台可注册已有参考/候选/拟合片段，83 条为动作质量批准、语言未经核验；不能当作上游 Spoken2Sign 词级 3D 词典。 |

## 上游实现需适配

[motion_gen.py](https://raw.githubusercontent.com/FangyunWei/SLRT/main/Spoken2Sign/motion_gen.py) 直接读取 connector、词典及预测结果，源码示例路径包括 Phoenix 数据；不能假定直接支持当前 CSL 路径和在线任意输入。它在词典缺词时 `continue` 跳过，产品接入必须返回缺失项并阻止静默丢失。上游示例的 Blender 渲染与本项目 Three.js 播放是不同执行路径，需定义动作格式、坐标系、骨架和角色重定向并验证。

该源文件顶部含 MPG 权利及使用许可声明；SMPL-X、MANO、模型权重、数据和衍生资产也须分别核查许可，不能仅凭仓库可见即假定可复制或商用。

## 不训练的实际顺序

1. 选 3–5 个完整句子，核对句子、视频 ID、内容摘要、关键点和许可；完整句级资产优先。
2. 查找或从官方渠道取得预训练 Text2Gloss 任务检查点、3D 词典、connector、SMPL-X/MANO 所需资源；分别检查文件身份、任务、模型配置、词表及许可。普通 mBART 或词表不等于任务权重。
3. 先只读验证资产结构，再在独立环境做单条加载和推理；资源缺失就标注 BLOCKED，不通过训练替代。
4. 若现成 3D 词典可用，验证 SMPL-X 参数到当前播放器的转换、重定向和动作/语言质量；若不可用，继续已有真人视频到三维动作资产的离线方案，先核实三维拟合工具和权重可运行。
5. 使用已有 TwoStream-S2G 对齐只能产生候选时间边界；审查手形和共发音后才加入词典。中文到 Gloss 和词间连接必须处理未知词、否定、时间和非手控信息。
6. 双方先约定语种、输入、结果状态、版本、来源/授权及时间线；共享角色骨架/动画协议是否需要统一由实际部署确定。

本次未运行模型、拟合、分段、动画或人工语言评测；没有宣称现有检查点可加载、资源可下载、质量达标或产品集成完成。
