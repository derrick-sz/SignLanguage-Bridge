# Libras RGB 本地数据审计说明

日期：2026-10-09（Asia/Shanghai）。本报告为开发规划的证据附件，非模型评估结果。

## 检查范围与方法

- 枚举 `Libras-EQT-UECE (RGB)` 全部目录／文件：只发现 5,347 个 MP4；统计全部文件大小，逐文件流式计算 SHA-256。
- 按一级目录开头的数字提取 source_item_id，保留目录完整原文；二级目录中的 INFORMANTE 为条目内局部编号。
- 按文件内容 hash 检查完全重复，未执行感知近似去重；不同 hash 不足以证明不是同一录制的转码或剪辑。
- 对每个有视频条目按相对路径排序取首条，共 178 条；额外为条目 1、2、92、116、152 的每个本地 INFORMANTE 取首条，去重后 198 条。
- 使用 ffprobe 读取上述 198 条的媒体容器／视频流元数据，未逐帧解码全部视频。未出现 probe 错误不等于全部视频完整可解码。
- 用 ffmpeg 解码 5 条代表视频的 20%／50%／80% 时刻，共 15 帧，制作并目视检查 contact sheet；未用图像推断全局人物身份或认定手语翻译。
- 用 NumPy、`allow_pickle=False` 读取两个旧序列目录全部 160 个 NPY，检查 shape/dtype/有限值与 -999。
- 对四份 Python 和两份参考 Markdown 计算 SHA-256；本轮结束时复查 Python 哈希未改变。

## 文件

| 文件 | 内容 |
|---|---|
| [summary.json](summary.json) | 总量、空目录、异常数量、重复组、样本媒体分布、旧 NPY 统计、源码与参考文档 hash |
| [item_inventory.csv](item_inventory.csv) | 178 条目原编号／标签及各局部 INFORMANTE 数量；UTF-8 BOM |
| [video_hashes.csv](video_hashes.csv) | 5,347 个相对视频路径、字节数、SHA-256；UTF-8 BOM |
| [exact_duplicate_groups.csv](exact_duplicate_groups.csv) | 76 对字节相同的视频路径及是否跨条目 |
| [sample_video_probes.json](sample_video_probes.json) | 198 条抽样视频的 ffprobe 结果 |
| [sample_contact_sheet.jpg](sample_contact_sheet.jpg) | 15 帧可视化样本；行标题仅作路径定位，不作为语言学判读 |

重点：198 与 199 的 30 条视频逐一完全相同；22 与 31 也有 30 对相同视频；151 与 165 有 12 对。另有 4 对为条目内部重复。总计 76 对，72 对跨条目，5,271 种不同文件内容。应在建训练集前处理标签冲突及重复组隔离，保留原始来源记录。

## 尚未完成的检查

没有安装项目依赖、执行应用、运行 MediaPipe、训练／反序列化模型、访问摄像头、逐视频完整解码、验证全部语义标注或确认全局签者身份。没有取得 T 包的官方 split，也没有通过原压缩包 checksum 确认本地发布版本。外部资料与本地证据的对应边界见 [开发规划](../../Libras识别Demo_后续开发规划_2026-10-09.md)。
