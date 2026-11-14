# SSH 传输验收记录（2026-10-29）

目标服务器：`dglguser@10.45.25.55`

目标目录：`/media/dglguser/datadisk/workspace/whh/SignLanguage-Bridge`

远端主机：`dglguser-05`

结论：传输完整性通过。远端总计约 131 GiB，剩余 1,014 GiB。本记录只验证文件传输、数量与摘要，不代表模型加载、推理、训练或语言质量验收。

## 远端目录

```text
SignLanguage-Bridge/
  project/source/          # 可运行源码归档
  project/git-worktree/    # 从 Git bundle 克隆的功能分支
  data/csl-daily/
  data/ce-csl/
  data/other/
  data/libras/
  models/uni-sign/
  models/slrt-online/
  models/emnlp24-online/
  models/spoken2sign/
  models/legacy-checkpoints/
  assets/action_archive/
  evidence/
  docs/
  manifests/
```

## Git

- 分支：`codex/csl-animation-workspace`
- HEAD：`86e0d45cd371d2db6db2a0c522021c37277d9828`
- Git bundle SHA256：`0af48bbcdb72fd04ca4193fa311c2f28b4e195d32788805eb3c972cd55cc9215`
- 远端 Git 工作树干净。

## 数量与容量

| 资源 | 远端结果 |
|---|---:|
| CSL-Daily 视频 | 19,528 文件，26,959,704,328 bytes |
| CSL-Daily 标注/关键点 | 11 个规范文件、5 个兼容符号链接 |
| CE-CSL | 5,991 文件 |
| 其他研究数据 | 38 文件 |
| Uni-Sign | 48 文件 |
| SLRT Online | 33 文件 |
| EMNLP24 Online | 14 文件 |
| Spoken2Sign | 155 文件 |
| 历史检查点 | 67 文件 |
| Libras | 10,684 文件，包括 processed ZIP、README 和 10,682 个 landmark 文件 |

CSL-Daily 视频的文件数与本地完全一致，文件长度总和也完全一致。三个关键点别名和三组句级标注别名在远端以符号链接恢复，避免重复上传。

## SHA256 对照

以下摘要均由本地与远端分别计算，结果一致：

| 文件 | SHA256 |
|---|---|
| 源码归档 | `2bcc06ba97ec72201b9e7e71815367d3363857a37aa7792df7cd265155062a1b` |
| 100 段动作 ZIP | `4d98380ac0fe08db90b042df63528ae9be5c96b93f492625ec27a015ea5edae5` |
| CSL-Daily keypoints | `23f8ab1063cfcfb882cc912f5ca918e77ee1dcc8d8ad2fac28c195b627107461` |
| CSL-Daily pose ZIP | `8b0dfc4c77f594e3a7ff39f434554fd5faed64621c32f32a8add05c7ba008835` |
| Uni-Sign CSL sentence model | `73bea87a9603f719b4f4a481a3a1a5ed24e7e5f7b81afd14ef498c40469c1a6c` |
| SLRT CSL-Daily CSLR | `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d` |
| EMNLP24 CSL-Daily CSLR | `e9fdd3a66d9667e7685c2b1437c67b8cbc9e45d2562730f04c0226717b7cbd4d` |
| CE-CSL TFNet | `160d35a42d9f8f52a090d0131586df1631d2fcf8c9f41ee94729cae1fb6b0386` |
| 中文 mBART T2G ZIP | `b49e4c62b2b5461a304d68ae9d6de0ad7b77d8e76a41f5c291dd903408185fcb` |
| 德文 mBART T2G ZIP | `f490a41ce8544dd9b73adb9d863f20160da63e4038dbae32c9aebea58d4d14cd` |
| Libras processed ZIP | `53a7c6ea943ed9319ad49c370755e5c68a82be675200ba6ab3a06449ddfb5f9e` |
| 小传输包 | `8a7437a02c57e77b56ea721cbf24c0e4c64cd5f5f6ffe79960475bd168650c9b` |

## 为避免重复而保留的本地资源

- Libras 展开目录约 34.17 GiB，与 `Libras-EQT-UECE-T.zip` 重复，未再上传；服务器保留 ZIP 和 landmarks。
- `artifacts/slrt-gdrive-20261025/mBart_zh_t2g` 的两个嵌入文件与完整中文 mBART ZIP 内部文件重复，未单独上传。
- CSL-Daily 的三个 keypoints 硬链接和三组句级标注硬链接，只在远端创建符号链接。

## 仍未取得的生成链资产

- Spoken2Sign 现成 3D 词典。
- Sign Connector `connector.pth`。
- Text2Gloss 独立任务 checkpoint。
- 完整且获许可的 SMPL-X/MANO 模型包。

## 未测试

- 尚未在远端创建 Python 环境、安装依赖或加载模型。
- 尚未执行识别、Text2Gloss、动作生成、三维重定向或训练。
- 尚未进行数据集解压完整性、真人语言理解和动画视觉质量验收。
