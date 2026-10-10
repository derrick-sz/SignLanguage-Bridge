# 可运行的模型 adapter 示例

统一调用入口是 `ModelManager.infer_file`、`infer_clip` 或 `infer`；语言并行服务用
`ModelRouter` 的独立 manager。已有 RF、ONNX 分类和 ONNX/CTC 由注册 adapter 包装，
仍调用真实后端与原模型包内的预处理。添加新后端需要注册 `AdapterDefinition`，调用方无需
增加后端分支。模型包必须明确自己的输入、预处理版本、词表、任务、runtime、设备和模式。

下面示例仅计算两个数的和，**没有训练，也不是手语识别或翻译效果证明**。它展示新增表示、
loader、typed `text` 输出与完整请求/响应绑定。保存代码为 Python 文件，在仓库根目录运行
`.venv-libras/Scripts/python.exe <文件名>` 即可；只使用 NumPy 和标准库。

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from time import perf_counter
import numpy as np

from slbridge.contracts import RecognitionResult, TextPrediction
from slbridge.model_adapters import (
    AdapterDefinition, InputRepresentation, ModelCapabilities, ModelInputContract,
    register_input_representation, register_model_adapter,
)
from slbridge.runtime import ModelManager
from slbridge.runtime_contracts import InferenceRequest

register_input_representation(InputRepresentation("example_pair_v1", (1, 2), layout="B,F"))


class ExampleTextAdapter:
    def __init__(self, path, options):
        data = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if data != {"backend": "numpy_text_example_v1", "schema_version": 1}:
            raise ValueError("Invalid example bundle")
        if options.device == "cuda":
            raise ValueError("Example supports CPU only")
        self.capabilities = ModelCapabilities(
            model_id="sum-example", model_version="example-v1", sign_language="demo",
            task_type="text", inputs=(ModelInputContract(
                "example_pair_v1", (1, 2), "float32", "identity-v1", ("prepared",)),),
            vocabulary=(), runtime="numpy", devices=("cpu",), score_type="none",
        )
        self.closed = False

    @property
    def recognizer(self):
        return self

    @property
    def runtime_info(self):
        return {"runtime": "numpy", "device": "cpu", "synthetic_example": True}

    def warmup(self):
        # Exercise the same real deterministic computation used by inference.
        assert float(np.ones((1, 2), dtype=np.float32).sum()) == 2.0

    def infer(self, source, request, *, input_kind):
        if self.closed:
            raise RuntimeError("Example adapter is closed")
        request.check_active()
        self.capabilities.negotiate(request, input_kind=input_kind).validate_prepared(source, request)
        started = perf_counter()
        prediction = TextPrediction(f"sum={float(source.sum()):.1f}", "en", score=None)
        result = RecognitionResult(
            "recognized", request.request_id, "sum-example", "demo", prediction,
            {"synthetic_example": True}, task_type="text", score_type="none",
        )
        request.check_active()
        return result, {"preprocess_ms": 0.0, "inference_ms": (perf_counter() - started) * 1000}

    def close(self):
        self.closed = True


register_model_adapter(AdapterDefinition("numpy_text_example_v1", ExampleTextAdapter))
with TemporaryDirectory() as directory:
    path = Path(directory)
    (path / "manifest.json").write_text(json.dumps({
        "backend": "numpy_text_example_v1", "schema_version": 1,
    }), encoding="utf-8")
    manager = ModelManager()
    try:
        manager.switch(path)
        request = InferenceRequest(
            "example-request", representation_id="example_pair_v1", sign_language="demo",
            expected_task_type="text", session_id="example-session", segment_id="segment-1",
            source_epoch=0, expected_generation=manager.generation, model_id="sum-example",
            preprocessing_version="identity-v1",
        )
        response = manager.infer(np.array([[1, 2]], dtype=np.float32), request)
        assert isinstance(response.result.prediction, TextPrediction)
        assert response.result.prediction.text == "sum=3.0"
        print(json.dumps(response.to_dict(), ensure_ascii=False))
    finally:
        manager.close()
```

`classification` 返回 `ClassificationPrediction`；`gloss_sequence` 返回有序
`GlossSequencePrediction`，不会自动变成自然语言翻译；`text` 必须明确返回 `TextPrediction`。
`ModelManager.capabilities()` 公开完整能力；`negotiate(request, input_kind=...)` 可提前检查，
实际推理还会在 generation lease 内再检查。响应给出实际 model/version/preprocessing/generation
和请求 session/segment/source epoch，调用端可据此拒绝迟到结果。

表示注册只登记协议身份。225 维与321维不能补零、截取或凭同名字段转换；新协议应使用新的
表示 ID，并实现经过验证的对应预处理和模型加载器。这里没有提供321维模型或在线提取器。

取消用 `CancellationToken.cancel()`；本进程 deadline 用
`deadline_monotonic=time.monotonic()+预算秒数`。远程调用应传 timeout budget，由服务端转换成
本地 monotonic deadline。框架在推理前后检查；adapter 可在自己的预处理阶段之间检查。
已进入 native 推理的调用无法安全强制中止，会保留模型 lease，结束后丢弃取消/过期结果。

注册的 loader 负责模型包完整性与资源释放。manager 对加载后校验或预热失败的候选执行
`close()`，保留活动模型和 generation；loader 若在构造过程中失败，必须释放自己已分配的资源。
显式注入的历史 run-only 对象仍可使用 `run()`，但不宣称 typed inference 能力。
