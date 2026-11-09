# 自由中文输入到手语演示

用户纠正了上轮理解：目标是自由输入文字后生成手语演示，不要求先选库内句子。本轮替代“只做原句接续”的主入口；旧场景库保留为辅助素材入口。

## 实现状态（2026-11-08）

自由文字提交、串行GPU任务、真实阶段进度、校验后发布、现有人物加载及播放控制的主链路已实现；`/generate/`为主入口，`/scenes/`重定向到该页，`/scenes/library`保留原句库。实现与资源修复已完成独立审查。当前运行配置使用只读checkpoint映射、CPU FP32原始文本编码缓存和CUDA逐层offload；这属于单独记录的执行配置，不声称与此前全CUDA路径逐位等价。

两条不同库外新句均已有真实任务及worker的`complete`产物：第一条“明天早上我想去图书馆学习。”，任务`04527b2e1ed04d7dac9956f79fabbe17`；第二条“今天晚上我们一起做饭。”，任务`10835c5c29fa43839cbe72718033c264`。两条均为显式4秒／120帧、50步、seed123，`target_motion=null`、`observed_keyframes=0`。证据见[库外输入检查](../../../artifacts/free-text-20261108/novel-inputs.json)、[任务1](../../../artifacts/free-text-20261108/accepted/job-1.json)、[生成1](../../../artifacts/free-text-20261108/accepted/generation-1.json)、[任务2](../../../artifacts/free-text-20261108/accepted/job-2.json)、[生成2](../../../artifacts/free-text-20261108/accepted/generation-2.json)。两份生成archive摘要及raw body摘要均不同，可证明产物身份不同，不能代替手语语义验收。

root已完成两条实际人物播放；第二条播放结束显示`0:04 / 0:04`，console error／warning均为空。该浏览器观察由root提供，完整浏览器证据由root归档。两条任务的`quality_warning=true`，语言审核均为`NOT_RUN`。root本轮新鲜回归：服务／HTTP／场景／两CLI的Python测试82通过，独立runtime的readonly／offload／generation／empty-loading测试48通过，generate＋scenes的Node测试18通过；分组记录，不把早期定向测试数叠加成新总数。

## 主流程与范围

`中文输入 → 提交本地生成任务 → 查看身体/双手/面部进度 → 获得新动作 → 在现有人物上播放`。

主入口`/generate/`；配置生成器后旧`/scenes/`重定向到该入口，原句页保留`/scenes/library`。自由文字不经过CorpusStore精确匹配，不因库外输入而阻断。

本轮复用已经实跑过的SignSparK预训练CLI，补齐网页调用链路。它是文本条件动作生成，不谎称本轮已经完成词级切分/检索增强。词级组合仍是用户偏好的质量改进路径。源文本、模型身份和原始动作保留，语言理解/手形/自然度仍须单独验证。

## 输入与任务

- 中文及普通文字1–200字符；换行/制表符可规范为普通空格，保留原输入与实际生成文字；其他控制字符拒绝。不删除否定、数字、人名、地点。不接受模型或本地路径等客户端参数。
- 演示时长可选4/6/8/10秒，换算为120/180/240/300帧；默认6秒。长度来自用户选择，不冒充学习到的动作时长。固定50步、seed123；服务器配置可由启动参数给出。
- 一次只有1个GPU任务运行，最多8个非终态任务；最多保留32个任务状态。queued/running/complete/failed/cancelled清楚区分。任务ID随机，客户端不能指定输出目录。
- 状态从实际worker manifest读取preparing/body/hand/face/packaging，不以定时器捏造百分比。失败保留本地日志，不返回服务器路径/堆栈。
- 用户取消时终止本服务创建的worker进程树；取消/输入变化使旧UI请求与晚到结果失效。取消后的任务不能发布成complete。

## 生成与发布合同

服务器使用固定可配置的python、worker_script、source、text_assets、models、base_archive、source_presentation、smplx_model及output_root，subprocess argv方式调用、shell=False、Windows隐藏窗口。每个任务创建独立目录，CLI输出位于新建前不存在的run子目录。源模型和语料只读。

仅当worker退出0、manifest.state=complete、requests文字/帧数与任务一致、target_motion=null、observed_keyframes=0，且candidate-motion.zip摘要通过MotionRegistry时发布。只发布text相符、frames相符的生成clip；保留quality_warning、原始/修订参数和语言NOT_RUN。返回motion结构兼容现有MotionStage与当前人物，无需更换模型或字幕伪装。

## 接口

模块`text_generation_jobs.py`：`GenerationService(config, runner=None)`；`submit(text, seconds=6)`、`get(job_id)`、`cancel(job_id)`、`motion(job_id)`、`close()`；`GenerationBusyError`表示队列满。config为server-owned字典，runner可用于测试；命名/签名实现前由root与worker实现者消息确认。

`POST /v1/animation/generation/jobs` body `{text,seconds}` → job；`GET /v1/animation/generation/jobs/{id}` → job；`DELETE .../{id}` → job；`GET .../{id}/motion` → MotionStage结构。queued/running的motion为409；unknown404；invalid422；queue full429；runtime unavailable503。

job最少字段：`job_id,text,generation_text,seconds,frames,state,phase,created_at,updated_at,language_review,quality_warning,motion_url,error`。服务端路径和命令不返回浏览器；error为可读的简短错误。

## 页面

新增`generate.html/js/css`及可测试controller。主区是文字输入、时长、生成/取消按钮及清楚状态；右侧/下方人物播放。已有原句库作为次要链接，不在生成前要求勾选库内句子。简短提示“生成演示仍需核对手语含义”，不强制重复审批。任务完成后显示可播放并提供播放/暂停/停止，默认自动加载但不擅自播放过时输入。

复用MotionStage；字幕来自经验证的生成clip.text并与generation_text比较。浏览器fetch正确绑定globalThis。所有用户文字用textContent；输入修改、取消、停止、请求乱序都有generation守卫。

## 验证

单测先行：任意库外文字不进入整句查找、状态机/串行/队列有界、取消防晚完成、失败不发布、receipt文字/帧数与摘要不符拒绝；HTTP输入和路径固定；前端延迟/取消/旧请求守卫。

真实验收：从网页提交至少一条未出现于本地库的中文句，观察真实GPU任务运行至complete并实际人物播放；第二条不同新句也实际生成，核对文本条件/产物身份及两结果不同。自动检查不替代真人CSL语义审核。测试新句不选此前两条run02示例，不使用源动作或参考动作。
