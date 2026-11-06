# SOKE 全量参数库与按需展示

用户要求持续推进已下载数据的充分利用。沿用当前工作树、Python/numpy/FastAPI/Three.js，不改模型调用速度、不重下载、不覆盖100条制作版、不提交推送。本方案的实现选择由执行者承担，依据用户连续推进授权直接执行。

## 目标与分期

1. 将全部20,654个标注身份建库；19,877段帧范围完整序列逐段CRC/数值验证并转换，不静默补775段不完整序列及2段缺失。失败明确记录。保留所有原始参数、来源摘要和划分。
2. 有界内存、单写者、原子文件、输入/协议绑定及可验证续跑。全包不展开数百万小文件。全量数值质量分级不等于手语质量批准。
3. 分页搜索、单条参数加载与实时人物播放；保留已有100条GLB入口与原始/修订对照。
4. 后续继续做原视频精确对齐、异常动作逐段修订、非手控表达与词级切分/独立审核。上述未完成前不宣称词典或总目标已完成。

## 方案选择

- 所有序列都放进现有ZIP/PresentationBundle：启动、全目录遍历、内存与导出成本随语料增长，不能作为全库方案。
- SQLite索引＋独立NPZ，人物实时重定向：采用；可逐条校验、续跑、分页，并与既有展示共用骨架和选流逻辑。
- 直接训练文生动作模型：当前缺少可靠对齐和独立质量审核，先完成可追溯动作库。

## 持久接口

库根目录含`corpus.sqlite3`、`progress.json`、`sequences/<ID前三位数字>/<ID>.npz`。SQLite WAL支持只读检索与单写者转换同时运行；不允许两个构建者写同库。

`metadata(key TEXT PRIMARY KEY,value TEXT)`中的value均为JSON。键：`build`（schema=soke-full-library-v1, source_archive_sha256, template_sha256, index_sha256, processing_version, refined）、`skeleton`（parents/restJoints/representation及模板SHA）。

`samples`列：`id TEXT PRIMARY KEY,text TEXT,gloss TEXT,split TEXT,signer INTEGER,expected_frames INTEGER,actual_frames INTEGER,state TEXT,parameter_file TEXT,parameter_sha256 TEXT,source_parameters_sha256 TEXT,metrics_json TEXT,error TEXT`。gloss和metrics_json为JSON；初始state为pending/incomplete_frames/missing_3d_sequence；数值转换后为converted_candidate，异常为failed。批准始终unverified。成员位置另存`members`表，仅构建者使用。

NPZ包含原始8个参数字段、effective_poses、features133_raw、frame_numbers。可选离线修订增加refined_poses/gesture_repair_mask；默认全量先保存原始数值流与旋转诊断，按需播放时可产生明确绑定的新修订候选，绝不继承旧批准。原始视频时间对齐仍unverified；本阶段只记录文件存在，不伪造裁剪/视频哈希。

## 约束与验收

- 逐段NPZ原子替换后提交数据库，续跑逐个核验已有文件SHA与参数，损坏或缺失重新转换；恢复必须拒绝不同输入/协议。
- ZIP目录流式扫描，位置批量落库；成员解压保留现有CRC/尺寸/静态pickle解析限制，复用一个文件流。
- 搜索每页最多100条，参数文件最多32MiB、每段最多3000帧；服务不整库加载姿态/生成数万DOM，不把未完成序列作为可播放项。
- 搜索、文本/Gloss、质量/划分筛选与分页必须有真实SQLite测试；加载损坏摘要、路径穿越、未知ID拒绝。
- 全量运行以数据库状态、实际文件和逐段校验为证据；浏览器新增非原100条的检索与播放单独验证。
- 表情、接触、原视频对齐和语言审核保持明确未验收，不由数值分级自动批准。
