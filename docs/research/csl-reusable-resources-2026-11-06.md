# 中国手语三维词典可复用资源核查（2026-11-06）

有可复用资源。当前角色、播放器和现有 100 段动作可继续使用；新增动作优先验证 SOKE 发布的 CSL-Daily 三维拟合数据，缺失或质量不足的片段才从视频重建。公开可下载、格式可读取、能驱动当前角色、词义正确和可对外发布是不同条件。

## 已核到的资源与可用范围

| 资源 | 可复用内容 | 当前状态与剩余工作 |
|---|---|---|
| 本机现有动作包与动画工作台 | 100 段动作、Godette 角色、播放/对照/审核/GLB 导出；已有 6 段制作候选 | 工程基础已可用。100 段中包含完整短句，不是 100 个已审核独立词；词义、来源授权和新修订的语言审核仍分别记录 |
| [SOKE CSL-Daily 数据](https://2000zrl.github.io/soke/) | 已拟合的 SMPL-X 身体、左右手、面部等参数，以及 train/val/test 标注 | 公共目录和下载已实查，已抽查一个动作帧文件；尚未完成整段读取、骨架转换、当前角色播放与语言审核 |
| 本机 CSL-Daily / CE-CSL | 原视频、句级标注和已有二维关键点，可补词/核对原片 | 现有 19,528 + 5,988 个 MP4；二维关键点不是三维动作。可用作 SOKE 来源对照和缺项重建输入 |
| [PHOSA / MVSign](https://naaapi.github.io/PHOSA/) | 多视角中国手语，身体、手部、面部的 SMPL-X 拟合参数与三维点 | [数据卡](https://huggingface.co/datasets/naaaaapi/MVSign)受限访问，需要申请/同意条件；标注 CC BY-NC 4.0。页面显示总量约 1.83 TB，宜选择标注及小子集；代码仍标 coming soon。本轮未登录、提交申请或接受条款 |
| [Spoken2Sign](https://github.com/FangyunWei/SLRT/tree/main/Spoken2Sign) | 词典/动作衔接架构、处理代码，以及本机已有的中文 T2G 资源 | 本机有完整大小的 `mBart_zh_t2g.zip`；其不是动作词库。本轮未确认旧三维词典下载仍可用，也未确认本机具备完整动作词典与连接器权重 |
| [SignAvatars](https://github.com/ZhengdiYu/SignAvatars) / VLibras | SMPL-X/MANO 处理与可视化、播放器和词典组织方法 | SignAvatars 标注需申请，不能将其全部样本算作中文词汇；VLibras 动作属于巴西手语，不能直接当中国手语词条 |

已有完整本地动作包：`E:/githubproject/signbridge/docs/assets/signbridge-chrome-extension-motion-demo-20261022.zip`，40,752,631 B，SHA256 `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5`。二维数据与文本模型只作为相关资源，不列作已完成三维词典。

## SOKE：本轮实际验证了什么

入口来自 [SOKE 官方项目页](https://2000zrl.github.io/soke/)和[官方 README](https://github.com/2000ZRL/SOKE#data)，不是仅依据论文存在作推断。公共 [CSL-Daily-Fittings 目录](https://www.dropbox.com/scl/fo/3lwcdq60q8yg3foj6ulh0/AOw8pBv6LEk0JR0KvgVR-m4?rlkey=9ewwux5vnslljv88hbb5ibhds&dl=0)实际显示：

- `csl-daily_pose.zip`：HTTP 范围响应确认总长 **2,886,912,220 B（约 2.69 GiB）**。
- `csl_clean.train`、`csl_clean.val`、`csl_clean.test`。
- `csl_mean.pt`、`csl_std.pt`。本轮未反序列化这两个文件。

已以访客下载训练索引 `csl_clean.train`：645,053 B，SHA256 `87f47e53f6a1fe2a86e3e5fcc3444829286139ff5ea6c03b9d766919bbef279c`。仅解压 gzip 后用 `pickletools` 检查操作码与字符串，未执行 pickle 反序列化。可见字段包含 name、text、gloss、num_frames、signer；取到 **18,401 个唯一训练样本 ID**，其中 **17,438 个**能匹配本机 CSL-Daily MP4 文件名。此处是文件名匹配，尚未核对原片 SHA、裁剪、帧号和内容。

未下载完整三维 ZIP。通过 HTTP Range 读取 ZIP 尾部、512 KiB 目录片段和一个小成员，确认：

- ZIP64 元数据声明 **2,481,971 个目录项**，包括目录和文件，不能解释成同等数量的视频、句子或词条。
- 目录片段实际解析 4,643 项，其中 4,603 个 `.pkl`；可见结构为 `csl-daily_pose/<sample_id>/<frame_number>.pkl`。目录顺序不是数字帧序，导入时需按帧号排序。
- 抽查成员 `csl-daily_pose/S006865_P0000_T00/000003.pkl`，解压后 1,219 B，CRC 校验通过；SHA256 `d66e6cd7929b0ca837e5551e571bc6a7f2b9ba9c2c44cf5ff27bad2a2058023e`。
- 仅通过 pickle 操作码看到了 `smplx_root_pose`、`smplx_body_pose`、`smplx_lhand_pose`、`smplx_rhand_pose`、`smplx_jaw_pose`、`smplx_shape`、`smplx_expr`、`cam_trans`。未验证这些数组的数值、单位、旋转表示或时序质量。

因此已证明它包含实际三维拟合参数文件，不能称为整包完整性通过、与本项目骨架兼容通过或词义审核通过。其大量逐帧小文件也意味着压缩包大小不能直接当作 Windows 全量解压磁盘与耗时预算；宜按选中的样本顺序读取并合并为序列，而非先铺开全部小文件。

## 复用条件与实施顺序

[SOKE 仓库许可](https://github.com/2000ZRL/SOKE/blob/main/license.txt)标为 **CC BY-NC-ND 4.0**。Dropbox 数据包、CSL-Daily 原始数据、人体模板与角色还需分别核对适用条件，不能由“公开可下载”推出商业使用或改编词典公开分发已获授权。许可状态应随每个来源保留；本轮没有替用户接受受限数据条款。

1. 保留当前角色、播放器、审核界面和导出工具，先使用已具备明确来源的动作。
2. 对 SOKE 选择少量有本地原片的代表序列，核对帧范围、帧率、参数表示和人体模板条件，完成 SMPL-X 到当前角色的映射；再判断是否能省去重建。
3. 将可用整句动作与词级边界分开审核，审核通过后关联词条；句级 text/gloss 标注本身不证明逐词时间边界。
4. 对缺失、遮挡严重或手形/接触不合格的词，才使用原片重建或补录。MVSign 可作为获得访问后的高细节候选来源，不安排全量下载。

这次核查支持优先采用[资源预算](../reports/csl-dictionary-resource-estimate-2026-11-06.md)的 A 路线，但没有新的全流程计时或人工效率实测，不据此直接缩短原 5–10 工作日规划。

## 证据与验证边界

本轮文件位于忽略目录 `artifacts/animation-upgrade/reuse-check/`：`soke-folder-verified.txt`、`soke-folder.png`、`soke-visible-links.json`、`browser-download.json`、`soke-train-inspection.json`、`soke-train-id-match.json`、`soke-zip-range.json`、`soke-archive-fragment.json`、`pose-sample.pkl`。最初的整份 ZIP 目录读取因超过预设下载上限停止，记录保留在 `soke-archive-sample.json`；随后仅抽取目录片段和一个小成员，没有整包下载。

`soke-archive-fragment.json`仅记录目录抽查与单成员 CRC，不是全库通过清单。当前角色重定向、完整序列有限值/抽搐检查、词级切分、手语专家审核和本机新增资源性能均 **NOT_RUN**。
