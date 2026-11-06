# 数据驱动下颌候选（2026-11-07）

目标：把已有 SMPL-X joint 22 的真实下颌旋转接入经几何校准的 Godette 制作角色，不从文字、音素、随机函数或 expression 系数猜测嘴形。语言、表情及原片对齐仍未审核。

源模板 SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`；目标 GLB SHA256 `b42c21dacb5d07617790c40c69f68a074aca58d74d76b898f2cb98bbbf96862c`。实际源 LBS 内唇点证明 +X 为开口；目标蒙皮证明 +X 为下颌下降，而原手动 -X 为闭口方向。`Jaw_4` 与 `Lips_Lower_13` 是兄弟节点，需把下唇子树绕同一下颌支点联动。

公共接口：`compileJawCandidate(poses,fps,profile)` 生成只读源之外的有界角度轨和诊断；`sampleJawCandidate(track,time)` 由统一播放时钟采样，往返定位确定；`jawCandidateForClip(clip,avatarSha,baked)` 只接受明确 profile、两摘要和未烘焙语料。目标骨架提供校准专用下颌/下唇联动与中立复位。

保守几何范围为 `[0,0.16]` 弧度。只取 joint 22 绕 X 的 twist，源旋转范数超过 0.35 弧度不驱动，负值归零，三帧中值去单点噪声，相邻速度不超过 0.6 rad/s。这些是研究显示约束，不是语法校正、逐帧真值或无碰撞保证。保留原始姿态；其余源面部、眼球、下颌侧向通道及 expr10/50不映射。

全库新协议明确提供 `clip.nonmanual_candidate`；匹配 profile 默认启用，可关。原100条无此字段，不加入自动嘴部轨，不重烘 GLB。已校准角色手动下颌改用正确 +X 与下唇联动；手动值覆盖自动值，眉眼独立，恢复按钮清除手动覆盖。未知人物不套用该标定。

缓存身份包含新 PLAYBACK_VERSION；诊断导出包含有效角度、启用/覆盖状态、源/目标绑定和 `export_supported=false`。本轮不支持含候选嘴部的同轨 GLB 下载，不改变旧文件下载含义。
