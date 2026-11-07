# Godette v11 独立 presentation 接入

范围：最新源码快照 `ad3806f8e0ba2227231492860cfed33eb70347f9`，冻结角色 SHA256 `e4a5766d98ab53871415da3ef4609c4848386c8b4de2958d5e420a8d7524d476`。这是显示侧研究候选，不是识别、生成或手语质量提升的结论。

## 构建与使用

在完整项目环境中执行，路径替换为当前实际文件：

```sh
python scripts/animation/build_godette_v11_presentation.py \
  --archive EXISTING_MOTION.zip \
  --avatar godette-reference-v11.glb \
  --output NEW_PRESENTATION_DIRECTORY

python -m slbridge.animation.cli \
  --archive EXISTING_MOTION.zip \
  --corpus EXISTING_CORPUS_DIRECTORY \
  --presentation NEW_PRESENTATION_DIRECTORY
```

保留原来的 `--catalog`、`--model`、`--port` 等服务选项。构建器可额外传 `--expected-sha256`；否则读取已有 `ZIP.sha256`（存在时）。它通过真实 `MotionRegistry` 读取归档，而不是从交接 JSON 文件重新计算 registry 的源摘要。

- 构建器只读取归档与冻结模型，生成新目录中的模型、`manifest.json`、`ATTRIBUTION.md`。不改原 ZIP、SQLite/NPZ、19,877 条语料、生成器、权重或输入动作。
- 六条 pilot 必须全部仍在当前 archive catalog 中，且与固定 source SHA 相符；缺失、错误或重复 ID 直接失败。`cap-0067` 保留 `source-quaternion-temporal-v1`。其他语料/新句仍走既有 registry/corpus 通路。
- 新包不复制旧 `export_file`。旧动画 GLB 绑定旧角色摘要，不能挪用为 v11 导出。本次也不新增 AnimationMixer 写入骨架。
- 目标目录必须不存在。先完成暂存与校验，再使用独占 `mkdir` 认领目录，最后发布 `manifest.json`，不会用 POSIX `rename` 覆盖并发创建的空目录。目录发布不是单步事务；应等待命令成功返回后启动读取它的服务。
- 所有 language/motion/expression 审核仍为 `unverified`；网页保留研究标识，并显示模型来源、许可链接和 v11 修改说明。

## 嘴部与旧校准隔离

v11 只有独立的手动研究通道。必须同时满足 v11 精确角色摘要、`godette-v1` 和下颌/下唇/嘴角父子关系，才启用最大 `0.04 rad` 的双骨开口与嘴角反转。关闭、停止和重复采样可恢复原始骨变换。

既有 `b42c…` 角色的 `JAW_AVATAR_SHA256`、源模板摘要、`jawCandidateForClip` 与生成端预检全部不变。v11 的自动非手控仍不可用；不能改写旧 candidate 的目标摘要来绕过门控。`jawRadians` 源轨不会驱动 v11 手动通道。导出复核摘要中的手动角度按 v11 的实际上限记录。

SignSparK 的 `--source-presentation` 继续使用原先经过源/目标校准的 presentation。不要把 v11 包冒充它；该生成器原有源 manifest 和 avatar SHA 检查应继续拒绝 v11。本次没有提供 v11 的自动嘴形、语言审核或重新校准。

## 验证命令

```sh
python -m pytest -q tests/test_godette_v11_presentation.py tests/test_animation_presentation.py \
  tests/test_animation_nonmanual.py tests/test_signspark_presentation.py \
  tests/test_signspark_generation.py tests/test_custom_signspark_presentation_cli.py \
  tests/test_custom_signspark_cli.py tests/test_animation_api.py tests/test_animation_corpus.py

# 环境变量使用服务真正使用的 Three.js，以及旧校准模型与冻结 v11 的本地绝对路径。
ANIMATION_THREE_MODULE=/absolute/three.module.mjs \
ANIMATION_AVATAR_GLB=/absolute/old-calibrated-avatar-clean.glb \
ANIMATION_V11_AVATAR_GLB=/absolute/godette-reference-v11.glb \
node --test tests/*.test.mjs
```

未提供真实资产时，相关 Node 资产测试明确跳过，不视为通过。浏览器正常加载的 vendor 来自原归档；本地审查使用官方 Three r170，不等于验证其他版本。

## 交付验证范围与待办

本地已覆盖构建器及真实 registry/parser/CLI/API 的合成信任锚链路、旧校准与冻结 v11 真实骨架、六段 raw/temporal/temporal+IK、整数/半帧/倒跳/末帧保持、输入不变、101 档嘴部与保留蒙皮面片。对照的所有编译键与显式 `amount=0/1` 端点保持不变。名义 `time=i/fps` 可能有浮点余数，修复会纠正旧实现的近端点冻结，这不叫“所有名义整数时刻都不变”。

当前交付环境没有生产 motion ZIP、19,877 条实际 SQLite/NPZ 及 GPU 生成权重，因此生产 archive→构建→实际服务、整库数量/检索与新句推理未重新验收。完整 pytest 还受缺失桌面/视频依赖与快照未包含的项目资源限制，不能称为全套通过。实际浏览器/WebGL贴图、双屏/窄屏和部署 vendor、碰撞/接触、手语语义与移动端性能仍需独立验收。

许可：Godette (Rigged) by zahlenmaler，CC BY 4.0；不是 CC0。随包保留完整 ATTRIBUTION。适配代码与 Three.js 的 MIT 许可不变。
