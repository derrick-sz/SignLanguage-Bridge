# Uni-Sign句末推理去除训练损失计算

目标：降低实时视频流句末翻译对单GPU/native worker的占用，保持SLRT词、姿态、Uni-Sign视觉输入、beam4、最大100 token及原始输出不变。沿用现有产品链路，不把此项称为长时退化根因修复或泛化提升。

实测起点：独立已冻结官方训练姿态3条×2次，每次mT5 encoder调用2次。runtime先运行上游训练forward计算未使用的loss，再generate；句末与实时帧共享worker。首轮候选6对全部视觉embedding/mask字节、生成token及文本相同，模型总均值285→268ms，但单例有波动，不能据此保证现场提速。

实现：在本runtime拥有、由backend串行使用的模型上，临时mT5 forward pre-hook取得原始embedding/mask并在训练计算前结束该forward，然后移除hook，再调用原上游generate。无上游源码复制或修改，无新增权重/解码策略。固定版本边界改变则明确失败；正常/异常均移除hook，异常不残留视觉输入traceback。元数据记录推理路径。

验证：真实nn.Module边界5项RED→GREEN，补shape/异常与既有hook保护；30官方train输入及4/256帧合成形状控制，在同一模型实例交错旧/新路径并充分预热，逐条对比embedding/mask摘要及token。输出-only记录通过GT guard，目标不参与运行。定向旧适配器/预处理/API/实时回归、独立审查，然后冻结实际WS旧/新事件并比较排队及句末耗时。检查idle后仅用身份启动器重部署，原配置/权重保持；网页真实回放和持续负载验证部署。不同输入与计时边界分别报告，未完成项标NOT_RUN。

证据目录：`artifacts/unisign-generation-optimization-20261111/`。原runtime和首版baseline探针在source-before保留，不覆盖旧报告。物理摄像头、陌生签者、语义审核与24小时稳定性仍需另验。

## 本轮执行结果与后续

模型32输入/96对：视觉摘要32对、token/text 96对一致，encoder 2→1，独占模型均值325→300ms。5项RED→GREEN及50定向后端通过；独立静态审查问题已关闭（审查者测试因内存未完成，未计通过）。按身份Stop/Start部署后实际8 WS输入全部完成且语义事件相同，排队P95 146→130ms，首词总P95 1143→1202ms没有改善。最终5分钟单WS 9056帧全处理、零丢帧/过载、64机器句，排队P95 151ms。

网页复验**未通过**：IAB attach/鼠标/截图控制超时，既有页面媒体到EOF但native收帧0，旧主字幕保留，不能算新结果。后续先修当前会话主显示和空帧结束提示，再恢复真实网页采样验证。资源预检及诊断清理另有保留的失败/修复记录；这些发现没有被抹去。完整结果见[执行报告](../../reports/unisign-generation-optimization-2026-11-11.md)，不将本项模型优化当整体应用/比赛目标完成。
