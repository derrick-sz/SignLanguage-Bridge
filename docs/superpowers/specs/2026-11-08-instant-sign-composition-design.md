# 自由输入的快速词级动作组合

用户已批准按前轮建议推进开发。主目标是自由文字进入真实词级组合，而不是完整原句查询或每句从头神经生成。复用现有动画功能工作树，不覆盖并发人物修订，不提交／推送。

## 数据与离线建库

使用作者固定revision的CSL-Daily ISO词边界及配套SignSparK train LMDB，不把边界套入SOKE。原始文件完整SHA验证后只读打开；ISO禁止class/persistent反序列化，LMDB仅读取NPZ数值字段、allow_pickle=False。body取最后60列，左右手各90列，face56列，使用现有codec、`left_hand_convention='smplx-left'`，不加SOKE均值。

词库`manifest.json`含schema=`signbridge-word-library-v1`、bundle_id、source身份、skeleton（parents/restJoints）、units列表。每个unit含id、label、relative path、sha256、frames、fps=30、source（video_file/start/end/split）。每词最多3个候选；词内帧不修改，语言审核均NOT_RUN。NPZ只存float32 poses[T,55,3]及expression[T,50]。原始输入与词库不改写。30FPS为显式产品预览时钟；作者LMDB schema没有时间戳且展示脚本默认20FPS，不能称原始采集FPS已核验。

## 文字规划

独立持久JSONL worker加载固定作者mBART T2G，进程保留模型。输入`{id,text}`，输出`{id,glosses:[str],model_identity,elapsed_ms}`；ready/error协议不含本机路径。禁止最近词替换。1–200字符，控制字符拒绝；输入/输出长度超限报错，不截断文本。

服务器类`TextGlossPlanner(config)`提供`plan(text)->dict`、`status()->dict`、`close()`；config包含python/script/model_dir/device。status为starting/ready/failed；模型缺失时HTTP503，不用逐字分词或伪造Gloss顶替。worker使用同一请求ID，超时/退出失效并终止自有进程；请求串行，忙时返回429。固定模型摘要，beam等配置记录。

## 在线组合与HTTP

`WordLibrary(path)`加载manifest，`compose(text,glosses)->dict`；选择最多32词、总长≤1800帧。每词取真实候选，以前一片段末姿态与下一候选首姿态的关节角距离选择；词内保留，段间新增6帧shortest-arc SLERP（平滑进出权重）。只平滑连接，不将数值平滑称为语言正确。

`CompositionService(library,planner)`提供`compose(text)->dict`、`status()->dict`、`close()`。POST `/v1/animation/composition`只接受`{text}`。成功返回`{status:'ready',input_text,generation_text,glosses,motion:{clip,parents,restJoints,sha256},timings_ms,language_review:'NOT_RUN'}`；clip.text绑定规范化输入，source_kind=`retrieved_word_composition`，composition包含词级来源及输出区间。缺词返回HTTP200 `{status:'blocked',glosses,missing_glosses,...}`且无motion，禁止部分句子播放或静默同义替换。空／非法输入422、环境不可用503、忙429、资产失效503。

每条NPZ使用安全相对路径、摘要、形状和有限值校验，有界缓存；加载错误不能用旧缓存掩盖文件被修改。HTTP不暴露磁盘路径。词库非人工审核仍可作明确标识的实验演示。

实际测试补充：beam1在“明天早上我们一起去医院看朋友。”中遗漏“医院”。增加有限的已知地点/时间词存在性检查：只对原文中出现且词库有同名词的固定地点/时间集合检查Gloss是否保留；发现遗漏返回HTTP200 `status='needs_review', omitted_terms:[...], glosses:[...], missing_glosses:[]`且无motion。页面明确“规划可能遗漏”，不能当成缺动作，也不自动补译。该检查不能证明无其他遗漏、错译、否定或数量错误；语言仍NOT_RUN。保留原失败证据并重新验收修后结果。

## 页面

现有`/generate/`默认“快速组合”，可显式切换“模型生成（较慢）”。快模式时长由词动作决定，不显示固定4/6/8/10秒选择。输入提交后校验实际响应并加载、自动播放；显示Gloss、缺词、服务端规划/组合时间和页面观察到的首播耗时。人物初始化不算入已预热提交计时；另测冷启动。输入/模式变化、取消与停止使晚到结果失效；不得沿用旧动作或字幕。

新增`FastCompositionController` API与现有controller对应：constructor({stage,fetcher,onState,now})、generate(text)、invalidate/cancel/stop/play/pause；status包括submitting/loading/ready/blocked/error。ready state可带subtitle/glosses/timings，自动播放仅当前请求。保持现有慢生成。

## 验收

先fixture覆盖边界、来源、未知词、取消/晚到、时长和身份；再实际离线词库+真实T2G跑库外短句（无需精确句命中），浏览器播放并记录提交到首个播放状态的耗时。覆盖内短句预热后1–2秒是目标，未实测前不承诺。报告真实冷/热、各阶段与失败样例；不把有限三例报告为P95。语言、手形、非手控语法及自然度另行NOT_RUN。
