# 核心框架初步实现验收

日期：2026-10-09（Asia/Shanghai）。范围：F0–F3 的真实运行闭环。

新增 `slbridge` 包、CLI、两份 RF 配置、核心依赖清单、包元数据及 45 项验收测试。原四个 app 脚本未修改，原始 NPY／MP4 和旧模型未覆盖。

## 实际验证

`D:/anaconda3/python.exe -m pytest -q`：**45 passed**。

原始两套 NPY 的训练后 hash 与冻结清单一致；四个 app 脚本 SHA256 与先前审计一致。`pip wheel --no-deps --no-build-isolation` 成功生成 `artifacts/packages/signlanguage_bridge_core-0.1.0-py3-none-any.whl`，未安装或变更运行环境依赖。

| 来源 | 索引 | Train / Validation / Test / Excluded | 实际结果 |
|---|---:|---|---|
| legacy-hand63 | 80 | 48 / 16 / 16 / 0 | RF 模型包已训练并在独立进程加载推理 |
| legacy-pose225 | 80 | 48 / 16 / 16 / 0 | 包含缺失 mask 的 RF 模型包已训练并在独立进程加载推理 |
| Libras RGB | 5347 | 3119 / 1042 / 1042 / 144 | 全量 hash 索引及条目内 local_informant 分组划分；尚未提取视频特征 |

两份遗留数据的 validation/test 各 16 个样本，accuracy 和 macro-F1 均为 1.0，阈值选择为 0、coverage 为 1。协议明确为 `legacy_sample_level_demo_signer_unknown`。这些数值只验证小规模遗留数据闭环，不能作为中国手语、Libras 或未知签者效果证明。

独立进程命令对 `A_000.npy` 返回 label A；单手示例 score 1.0，全身示例 score 0.89。两者均标为未经校准的概率，sign_language 为 null，display_text 为空仍识别成功。

Libras 5271 个不同字节 hash，144 条跨标签重复记录隔离；178 个条目中 174 个进入冻结 split。局部 INFORMANTE 不代表全局 signer_id；视频索引中保留 `rgb_decode_not_verified` 状态说明。

## 可复查产物

- `artifacts/datasets/legacy-hand63/{manifest,split}.json`
- `artifacts/datasets/legacy-pose225/{manifest,split}.json`
- `artifacts/datasets/libras-rgb/{manifest,split,summary}.json`
- `artifacts/models/core-hand63-20261009/`
- `artifacts/models/core-pose225-20261009/`
- `artifacts/verification/{hand63,pose225}-{train,predict}.json`

模型包包含输入契约、完整类别表、解析配置、冻结数据／split、评价、代码与依赖版本及校验和。实际验证没有安装额外依赖或启动摄像头。

## 审查与限制

审查发现并修复：核心 manifest 保存接口的原始目录保护；加载后完整 InputSpec 丢失造成相同长度异构输入误接收；配置字典可变造成参数记录失真。对应回归测试及独立复现探针均通过。

当前只实现一个 RF backend 和 clip classification。RGB 提取、视频／摄像头输入识别、连续手语、中文／葡语翻译及旧 UI 接入仍属于后续阶段。具体使用见[开发与使用说明](../核心开发与使用_2026-10-09.md)。
