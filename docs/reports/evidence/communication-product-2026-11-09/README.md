# 产品与动画同步验证证据

此目录保存可随Git查看的小型证据；外部模型、词库、视频和完整运行日志不包含在内。人物为Godette by zahlenmaler（CC BY 4.0），本项目已修改外观与服装，来源和改动见[资产归因](../../../../assets/godette-v14-preview/ATTRIBUTION.md)。截图中的识别视频用于本地工程接入记录，原视频未加入本目录。

文本保留原始输出内容，换行统一为LF；源码摘要也按LF规范化，避免Windows检出换行差异。截图字节保持原样。

| 文件 | 含义 |
|---|---|
| `verification.json` | 2026-11-09定向复查结果、固定基线、源码和证据摘要；不把重叠用例相加 |
| `python-preview.txt` / `python-generation-runtime.txt` / `node.txt` | 新跑检查的原始输出，包括跳过原因 |
| `browser-audit-final.json` | 2026-11-08真实上传、人工核对、回复、阻断、取消、窄屏和对话恢复的DOM记录 |
| `http-audit-final.json` | 2026-11-08身份、输入拒绝和上传WebM转码适配；不是物理摄像头 |
| `launcher-negative-control.json` | 错误配置被拒绝且监听PID保持 |
| `retarget-integration-20261108.json` | 原句和新组合的源指骨与避让检查，保留新组合失败 |
| `product-final.jpg` / `product-mobile-*.jpg` | 2026-11-08实机产品和窄屏截图 |

物理摄像头、实际浏览器录制、真人语义/手形/自然度、Libras动作输出、完整中葡句子互译、全仓库与本轮完整模型GPU不属于通过项。源代码测试和工程可播放不能覆盖这些边界。
