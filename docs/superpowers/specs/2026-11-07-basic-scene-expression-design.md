# 基本场景表达覆盖与组合验证

用户已授权由本次工作确定方案、实施并验证；以已有连贯句级语料优先覆盖基本场景。采用 brainstorming、writing-plans 和并行实现流程，不再停在方案确认。

## 方案选择

比较三条路：仅建立原句场景目录可以快速覆盖但不能组合；立即切全部2,000词受未验证边界限制；完整表达检索加有界多句组合可保留原句内部动作并产生可追溯的新多句表达。本轮选第三条，词级边界建设作为后续阶段，不把多句顺接称为任意中文词级翻译。

## 本轮交付

1. 固定12场景：问候礼貌、身份家庭、时间安排、饮食、购物支付、出行问路、学校学习、工作沟通、健康就医、居家日常、情绪社交、求助安全。每场景8条事先写定目标文本，共96条，用于明确现有能力缺口；词目/候选句数量不冒充目标覆盖率。
2. 从只读数据库中索引完整train候选；按场景关键词组织真实原句，并提供原句检索。关键词归类仅为候选分组；原字幕不换成查询文本。数量分别报告词条、不同文本和实例。
3. 自定义输入先匹配完整表达；否则只按句号、问号、感叹号、分号、换行等显式边界分成至多6段，每段必须完整匹配已有原句。未覆盖任何一段时整个计划blocked，不替词、不漏词、不部分自动播放。问号与感叹号、数值小数、否定、专名不因规范化丢失；中英文宽窄标点可做等价转换，句末句号可忽略。
4. 一段直接复用原动作，多段保留每段全部55关节源帧，在段间插入9帧/30FPS的四元数SLERP过渡；不裁剪词内动作、不重采样原帧。每段源SHA、参数身份、原帧范围、输出帧范围、过渡区及处理版本随计划和动作返回。每次返回动作前重验来源，变化则拒绝。组合时基和人物模板必须一致，总帧数上限1800（含过渡）。过渡不代表自然度或接触已通过。
5. 独立基本场景页面`/scenes/`，共用现有MotionStage、人物和静态依赖；新CLI运行独立端口8879，保留既有8874/8878服务。可以选择原句、添加表达、输入多句、预览、停止、查看缺项与下载计划。输入/选择变化、停止、请求乱序均取消旧播放。页面不自动播放。
6. 固定目标覆盖报告、逐场景真实源参数审核、至少12场景各一条浏览器原句播放和一个多句组合播放、OOV/数字否定/问题标点/来源漂移等负控；语言/自然度、现场摄像头和词级新句能力不由这些检查升级，标NOT_RUN。

## 实现边界

复用已有功能工作树，不修改已有并发修改的api.py、stage.js、workspace.js、retarget.js、rig.js和pyproject.toml。只新增scene相关文件。语料与第三方资产只读，不训练、不下载大包、不提交/推送其他工作。先保持原句连续运动，新的段间生成属于研究预览。界面只呈现用户需要的来源和未覆盖项，详细参数放可下载计划。

## 接口

- `SceneLibrary(store, definitions_path)`：只读构建train索引；`catalog(scene=None,q='',limit=30,offset=0)`；`coverage()`；`plan(text,confirmed=False,research_preview=False)`；`motion(plan)`。
- catalog：`scenes[{id,title,candidate_texts,target_total,target_exact}]`、`assets[{sample_id,text,frames,duration,split,...}]`、`total,limit,offset,has_more`；候选文本去重。
- plan：`plan_id,input_text,status,can_play,mode,items,unmatched,reason,language_review,composition_review`；items有`sample_id,asset_id,text,sha256,frames`。只有confirmed和research_preview都为true且全部匹配时can_play=true。
- `mount_scene_routes(app, library)`：`GET /v1/animation/scenes`（catalog）、`GET /v1/animation/scenes/coverage`、`POST /v1/animation/scenes/plans`、`GET /v1/animation/scenes/plans/{plan_id}/motion`。服务器只保存最多128个完整计划；未知/过期计划404，来源漂移409。服务端固定配置，不接受本地路径输入。
- 独立CLI `python -m slbridge.animation.scene_cli --archive ... --expected-sha256 ... --presentation ... --corpus ... --port 8879` 复用基础create_app后挂载新路由。

## 界面设计

复用现有冷灰蓝三维舞台；底色#f2f5f7、正文#183542、主操作#176b78、边线#cbd7dd、缺项#9b4d24。系统中文无衬线字体，左侧可换场景和检索、中央真实原句列表、右侧人物与表达编辑器；窄屏纵排。保留清晰键盘焦点，列表空项说明如何查找，不用进度百分比暗示语言正确。数据和用户文字使用textContent。
