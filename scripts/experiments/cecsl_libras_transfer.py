"""Frozen MAM visual transfer probe on the exact historical Libras RF folds.

freeze and self-test never import torch or execute GPU inference. run is an
explicit, separately dispatched experiment; it trains only NumPy ridge heads.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import numpy as np

from slbridge._io import canonical_hash, read_json, resolve_resource, sha256_file, write_json_atomic
from slbridge.cecsl import RGBInputSpec, prepare_rgb_file, preprocessing_manifest
from slbridge.contracts import DatasetManifest, SplitManifest
from slbridge.tfnet_native import TFNET_CHECKPOINT_SHA256, TFNET_SOURCE_COMMIT, verify_tfnet_source

TRANSFER_ROOT = ROOT / "artifacts/cecsl-integration-20261009/transfer"
DEFAULT_PILOT = ROOT / "artifacts/experiments/libras-rf-pilot-20261009"
DOC = ROOT / "docs/CE-CSL_Libras迁移验证_2026-10-09.md"
ARMS = ("cecsl_pretrained", "random_seed42")
LAMBDAS = (0.01, 1.0, 100.0)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def within_transfer(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(TRANSFER_ROOT.resolve()) or resolved == TRANSFER_ROOT.resolve():
        raise ValueError("Probe outputs must stay within the assigned transfer directory")
    return resolved


def code_identity() -> dict:
    return {"scripts/experiments/cecsl_libras_transfer.py": sha256_file(__file__),
            **{name: sha256_file(ROOT / "slbridge" / name) for name in ("cecsl.py", "tfnet_native.py")}}


def freeze(pilot: Path, source_dir: Path, protocol_path: Path, spec: RGBInputSpec) -> dict:
    """Verify bytes, metadata and identical old folds before any model executes."""
    import cv2

    pilot, protocol_path = pilot.resolve(), within_transfer(protocol_path)
    files = ("source_manifest.json", "feature_manifest.json", "feature_split.json", "result.json")
    raw = {name: read_json(pilot / name) for name in files}
    source = DatasetManifest.from_dict(raw["source_manifest.json"])
    features = DatasetManifest.from_dict(raw["feature_manifest.json"])
    split = SplitManifest.from_dict(raw["feature_split.json"])
    if features.fingerprint != split.dataset_hash:
        raise ValueError("Historical feature split does not match its dataset fingerprint")
    folds = {name: list(getattr(split, name)) for name in ("train", "validation", "test")}
    chosen = [identifier for name in folds for identifier in folds[name]]
    if len(chosen) != len(set(chosen)):
        raise ValueError("Historical folds overlap")
    source_by_id = {row.sample_id: row for row in source.samples}
    feature_by_id = {row.sample_id: row for row in features.samples}
    if set(chosen) | set(split.excluded) != set(feature_by_id):
        raise ValueError("Historical split omits feature samples")
    samples = []
    for identifier in chosen:
        original, derived = source_by_id[identifier], feature_by_id[identifier]
        if original.quality_status != "ready" or derived.quality_status != "ready":
            raise ValueError("Transfer cannot add historically excluded/nonready rows")
        if (derived.derived_from != identifier or original.label_id != derived.label_id
                or derived.groups.get("raw_source_hash") != original.content_hash):
            raise ValueError("Historical raw-to-feature provenance differs")
        path, feature = resolve_resource(source.root, original.resource), resolve_resource(features.root, derived.resource)
        if sha256_file(path) != original.content_hash or sha256_file(feature) != derived.content_hash:
            raise ValueError(f"Historical bytes changed: {identifier}")
        capture = cv2.VideoCapture(str(path))
        try:
            count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or count < 2 or not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Unusable original video metadata: {identifier}")
        finally:
            capture.release()
        seconds, size = count / fps, path.stat().st_size
        if count > spec.max_frames or seconds > spec.max_seconds or size > spec.max_resource_bytes:
            raise ValueError(f"Same historical sample exceeds frozen RGB limits: {identifier}; frames={count}, seconds={seconds}, bytes={size}")
        samples.append({"sample_id": identifier, "label_id": original.label_id,
                        "resource": original.resource, "video_sha256": original.content_hash,
                        "historical_feature_sha256": derived.content_hash, "groups": original.groups,
                        "metadata_frames": count, "metadata_fps": fps, "metadata_seconds": seconds,
                        "video_bytes": size})
    labels = sorted({row["label_id"] for row in samples})
    if len(labels) != 12 or len(samples) != 360 or [len(folds[name]) for name in folds] != [216, 72, 72]:
        raise ValueError("Probe requires the exact historical 360-sample/12-class/216-72-72 pilot")
    source_identity = verify_tfnet_source(source_dir)
    identity = {
        "schema_version": 1, "name": "frozen_visual_MAM_Libras_transfer_probe",
        "pilot": str(pilot), "historical_files_sha256": {name: sha256_file(pilot / name) for name in files},
        "feature_dataset_fingerprint": features.fingerprint, "source_root": str(Path(source.root).resolve()),
        "source": source_identity, "checkpoint_sha256": TFNET_CHECKPOINT_SHA256,
        "labels": labels, "folds": folds, "historical_excluded": list(split.excluded), "samples": samples,
        "input_spec": spec.to_dict(), "preprocessing": preprocessing_manifest(spec), "code_sha256": code_identity(),
        "arms": {"cecsl_pretrained": "Strictly loaded CE-CSL TFNet conv2d MAM; frozen eval weights/buffers",
                 "random_seed42": "Same audited TFNet constructor; torch seed42; no checkpoint state loaded"},
        "feature": {"dimension": 1024, "visual_dimension": 512, "frame_selection": "6:6+source_frames",
                    "pooling": "concatenate temporal mean and population std; boundary padding excluded from pooling",
                    "whole_clip_context": True, "backbone_gradient_updates": False},
        "head": {"name": "NumPy ridge one-hot classification with unpenalized intercept",
                 "normalization": "train-only feature mean/std; std<1e-8 replaced by1",
                 "lambdas": list(LAMBDAS), "selection": "validation accuracy only; tie picks first (smallest lambda)",
                 "selection_frozen_before_test": True, "refit_train_plus_validation": False},
        "metrics": ["accuracy", "macro_f1", "confusion_matrix", "per_class", "sample_predictions"],
        "random_seeds": [42], "deterministic_torch": True,
        "historical_RF_context": raw["result.json"],
        "limitations": ["Single-seed bounded representation probe, not statistical significance",
                        "Same MediaPipe-ready historical subset; selection bias retained",
                        "Local informant groups; globally independent signer identity unverified",
                        "RF differs in input modality and head, so RF comparison does not isolate CE pretraining gain",
                        "No CE-CSL/Libras gloss vocabulary alignment or natural-language translation claim"],
    }
    protocol = {"created_at": utc(), "status": "frozen_not_run", "protocol_hash": canonical_hash(identity), "identity": identity}
    if protocol_path.exists():
        old = read_json(protocol_path)
        if old.get("identity") != identity:
            raise ValueError("Refusing to change an existing frozen transfer protocol")
        return old
    write_json_atomic(protocol_path, protocol)
    write_doc(protocol, None)
    return protocol


def load_protocol(path: Path) -> dict:
    protocol = read_json(within_transfer(path))
    identity = protocol["identity"]
    if canonical_hash(identity) != protocol["protocol_hash"] or code_identity() != identity["code_sha256"]:
        raise ValueError("Transfer protocol or implementation identity changed")
    if verify_tfnet_source(identity["source"]["source_dir"]) != identity["source"]:
        raise ValueError("Trusted source identity changed since protocol freeze")
    for name, digest in identity["historical_files_sha256"].items():
        if sha256_file(Path(identity["pilot"]) / name) != digest:
            raise ValueError("Historical pilot artifact changed since protocol freeze")
    return protocol


def metrics(truth: np.ndarray, predicted: np.ndarray, labels: list[str], sample_ids: list[str]) -> dict:
    if truth.shape != predicted.shape or truth.ndim != 1 or not len(truth) or len(sample_ids) != len(truth):
        raise ValueError("Classification metrics require aligned nonempty predictions")
    classes = len(labels)
    if np.any(truth < 0) or np.any(truth >= classes) or np.any(predicted < 0) or np.any(predicted >= classes):
        raise ValueError("Classification prediction IDs are outside the frozen label mapping")
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    per_class = []
    for index, label in enumerate(labels):
        true_positive, support, predicted_count = int(confusion[index, index]), int(confusion[index].sum()), int(confusion[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.
        recall = true_positive / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        per_class.append({"label_id": label, "support": support, "predicted_count": predicted_count,
                          "precision": precision, "recall": recall, "f1": f1})
    return {"samples": len(truth), "accuracy": float(np.mean(truth == predicted)),
            "macro_f1": sum(row["f1"] for row in per_class) / classes,
            "label_order": labels, "confusion_matrix": confusion.tolist(), "per_class": per_class,
            "predictions": [{"sample_id": sample_id, "truth": labels[int(target)], "prediction": labels[int(candidate)]}
                            for sample_id, target, candidate in zip(sample_ids, truth, predicted)]}


def ridge_fit(train: np.ndarray, targets: np.ndarray, classes: int, regularization: float) -> dict:
    """Dual solve is bounded by 216 training samples, avoiding a 1025² solve."""
    if train.ndim != 2 or targets.shape != (len(train),) or regularization <= 0 or not np.isfinite(train).all():
        raise ValueError("Invalid ridge training inputs")
    train = np.asarray(train, dtype=np.float64)
    mean, scale = train.mean(0), train.std(0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    normalized = (train - mean) / scale
    one_hot = np.eye(classes, dtype=np.float64)[targets]
    bias = one_hot.mean(0)
    gram = normalized @ normalized.T
    weights = normalized.T @ np.linalg.solve(gram + regularization * np.eye(len(train)), one_hot - bias)
    return {"mean": mean, "scale": scale, "weights": weights, "bias": bias}


def ridge_predict(head: dict, values: np.ndarray) -> np.ndarray:
    scores = ((np.asarray(values, dtype=np.float64) - head["mean"]) / head["scale"]) @ head["weights"] + head["bias"]
    if not np.isfinite(scores).all():
        raise ValueError("Ridge produced nonfinite scores")
    return scores.argmax(1)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.parent / f".{path.name}.{uuid.uuid4().hex}.partial"
    try:
        with staging.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def backbone_hash(backbone) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(backbone.state_dict().items()):
        array = value.detach().cpu().numpy()
        digest.update(name.encode())
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_backbone(arm: str, protocol: dict, checkpoint: Path, device: str):
    import cv2
    import torch
    from slbridge.tfnet_native import TFNetNativeRunner, _load_source_model

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = torch.device(device)
    if target.type not in ("cpu", "cuda") or target.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("Probe requires an available explicit CPU or CUDA device")
    source = Path(protocol["identity"]["source"]["source_dir"])
    if arm == "cecsl_pretrained":
        runner = TFNetNativeRunner(checkpoint, source, device="cpu")
        backbone = runner.model.conv2d
        runner.close()
    else:
        model = _load_source_model(source, torch.device("cpu"))
        backbone = model.conv2d
        del model
    digest = backbone_hash(backbone)
    backbone.requires_grad_(False).to(target).eval()
    environment = {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
                   "opencv_version": cv2.__version__, "python_version": sys.version,
                   "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
                   "device": str(target), "device_name": torch.cuda.get_device_name(target) if target.type == "cuda" else "cpu",
                   "deterministic_algorithms": True, "tf32": False}
    return backbone, target, digest, environment


def embeddings(arm: str, protocol: dict, checkpoint: Path, device: str, output: Path) -> tuple[np.ndarray, dict]:
    import torch

    identity, started = protocol["identity"], time.perf_counter()
    backbone, target, model_hash, environment = build_backbone(arm, protocol, checkpoint, device)
    spec = RGBInputSpec.from_dict(identity["input_spec"])
    cache_identity = {"protocol_hash": protocol["protocol_hash"], "arm": arm, "model_state_sha256": model_hash,
                      "preprocessing_hash": canonical_hash(identity["preprocessing"]), "environment": environment}
    cache_root = output / arm / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    vectors, ledger, reused = [], [], 0
    try:
        for ordinal, sample in enumerate(identity["samples"], 1):
            path = resolve_resource(identity["source_root"], sample["resource"])
            if sha256_file(path) != sample["video_sha256"]:
                raise ValueError(f"Frozen video bytes changed: {sample['sample_id']}")
            key = canonical_hash({**cache_identity, "video_sha256": sample["video_sha256"]})
            cached, cached_meta = cache_root / f"{key}.npz", cache_root / f"{key}.json"
            if cached.is_file() and cached_meta.is_file():
                row = read_json(cached_meta)
                if row.get("cache_key") != key or row.get("npz_sha256") != sha256_file(cached):
                    raise ValueError("Embedding cache identity/checksum changed")
                with np.load(cached, allow_pickle=False) as archive:
                    vector = archive["embedding"].copy()
                reused += 1
            else:
                preprocess_start = time.perf_counter()
                prepared = prepare_rgb_file(path, spec)
                preprocess_seconds = time.perf_counter() - preprocess_start
                if prepared.source_frames != sample["metadata_frames"]:
                    raise ValueError("Actual full decode frame count differs from frozen metadata")
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    tensor = torch.from_numpy(prepared.values.copy()).to(target)
                    visual, *auxiliary = backbone(tensor.transpose(1, 2))
                    original = visual.reshape(1, prepared.values.shape[1], 512)[0, 6:6 + prepared.source_frames]
                    vector = torch.cat((original.mean(0), original.std(0, unbiased=False))).float().cpu().numpy().copy()
                    del tensor, visual, original, auxiliary
                if target.type == "cuda":
                    torch.cuda.synchronize(target)
                inference_seconds = time.perf_counter() - inference_start
                atomic_npz(cached, embedding=vector)
                row = {"sample_id": sample["sample_id"], "video_sha256": sample["video_sha256"], "cache_key": key,
                       "npz_sha256": sha256_file(cached), "source_frames": prepared.source_frames,
                       "padded_frames": prepared.values.shape[1], "preprocess_seconds": preprocess_seconds,
                       "inference_seconds": inference_seconds}
                write_json_atomic(cached_meta, row)
            if vector.shape != (1024,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                raise ValueError("Embedding must be a finite float32 1024-vector")
            vectors.append(vector)
            ledger.append(row)
            state = {"status": "extracting", "arm": arm, "pid": os.getpid(), "completed": ordinal,
                     "selected": len(identity["samples"]), "cache_reused": reused,
                     "elapsed_seconds": time.perf_counter() - started, "last_sample_id": sample["sample_id"]}
            write_json_atomic(output / arm / "progress.json", state)
            print(json.dumps(state), flush=True)
        matrix = np.stack(vectors)
        atomic_npz(output / arm / "embeddings.npz", embeddings=matrix)
        evidence = {"cache_identity": cache_identity, "cache_reused": reused,
                    "embeddings_sha256": sha256_file(output / arm / "embeddings.npz"),
                    "elapsed_seconds": time.perf_counter() - started, "samples": ledger}
        write_json_atomic(output / arm / "embedding_ledger.json", evidence)
        return matrix, evidence
    finally:
        del backbone
        if target.type == "cuda":
            torch.cuda.empty_cache()


def select_head(matrix: np.ndarray, protocol: dict, output: Path, arm: str) -> tuple[dict, dict]:
    identity, labels = protocol["identity"], protocol["identity"]["labels"]
    by_id = {row["sample_id"]: index for index, row in enumerate(identity["samples"])}
    targets = np.array([labels.index(row["label_id"]) for row in identity["samples"]], dtype=np.int64)
    indices = {fold: np.array([by_id[identifier] for identifier in members], dtype=np.int64)
               for fold, members in identity["folds"].items()}
    candidates, heads = [], []
    for value in identity["head"]["lambdas"]:
        head = ridge_fit(matrix[indices["train"]], targets[indices["train"]], len(labels), value)
        validation = ridge_predict(head, matrix[indices["validation"]])
        candidates.append({"lambda": value, "validation_accuracy": float(np.mean(validation == targets[indices["validation"]]))})
        heads.append(head)
    chosen = max(range(len(candidates)), key=lambda index: candidates[index]["validation_accuracy"])
    selection = {"protocol_hash": protocol["protocol_hash"], "arm": arm,
                 "candidates": candidates, "selected_lambda": candidates[chosen]["lambda"],
                 "selection_rule": identity["head"]["selection"], "test_metrics_consulted": False}
    write_json_atomic(output / arm / "selected_config.json", selection)
    atomic_npz(output / arm / "head.npz", **heads[chosen], labels=np.asarray(labels))
    result = {"arm": arm, "selection": selection, "head_sha256": sha256_file(output / arm / "head.npz")}
    predicted = ridge_predict(heads[chosen], matrix[indices["validation"]])
    result["validation"] = metrics(targets[indices["validation"]], predicted, labels, identity["folds"]["validation"])
    return heads[chosen], result


def evaluate_test(head: dict, matrix: np.ndarray, protocol: dict) -> dict:
    identity, labels = protocol["identity"], protocol["identity"]["labels"]
    by_id = {row["sample_id"]: index for index, row in enumerate(identity["samples"])}
    indices = np.array([by_id[identifier] for identifier in identity["folds"]["test"]], dtype=np.int64)
    targets = np.array([labels.index(identity["samples"][index]["label_id"]) for index in indices], dtype=np.int64)
    return metrics(targets, ridge_predict(head, matrix[indices]), labels, identity["folds"]["test"])


def write_doc(protocol: dict, report: dict | None) -> None:
    identity = protocol["identity"]
    rf = identity["historical_RF_context"]
    status = "协议已冻结；迁移实验尚未运行。" if report is None else "迁移实验已完成，以下数值来自本地实际运行。"
    rows = [f"| MediaPipe + RF（历史上下文） | {rf['validation']['accuracy']:.4%} | {rf['test']['accuracy']:.4%} | {rf['test']['macro_f1']:.6f} | 72 |"]
    for arm in ARMS:
        if report is None:
            rows.append(f"| {arm} + ridge | 未运行 | 未运行 | 未运行 | 72（计划） |")
        else:
            result = report["arms"][arm]
            rows.append(f"| {arm} + ridge | {result['validation']['accuracy']:.4%} | {result['test']['accuracy']:.4%} | {result['test']['macro_f1']:.6f} | {result['test']['samples']} |")
    text = f"""# CE-CSL → Libras 冻结视觉表示迁移探针

{status}

固定协议标识：`{protocol['protocol_hash']}`。协议及逐条视频身份记录见 `artifacts/cecsl-integration-20261009/transfer/protocol.json`；可提交实现见 `scripts/experiments/cecsl_libras_transfer.py`。当前样本数360、类别数12，沿用历史 feature_split 的216训练/72验证/72测试成员及顺序，核对原始视频与历史特征文件哈希，不重新划分或补入曾被 MediaPipe 过滤的样本。

冻结命令：`python scripts/experiments/cecsl_libras_transfer.py freeze --source-dir artifacts/cecsl-tfnet-audit-20261009/TFNet`。实际运行命令：`python scripts/experiments/cecsl_libras_transfer.py run --checkpoint D:/Zenodo/CE-CSL/TFNet-CE-CSL-CSLDaily-32.46.pth --device cuda:0`。PyTorch/CUDA 与独立审计源码、下载 checkpoint 均属于本地可选实验依赖；未将上游源码或原 checkpoint 复制到新脚本。

CE-CSL 对照使用固定 checkpoint 严格加载后的 MAM ResNet34 视觉骨干；随机对照使用同一审计源码架构与 torch seed42。两者均 eval、冻结参数和归一化缓冲，不使用原 CE-CSL CTC 分类头。每段完整原始 RGB 视频采用相同 TFNet JPEG/裁剪/归一化/边界填充，仅将原始帧位置6:6+n的512维视觉特征按时间均值与总体标准差拼接成1024维向量。

两组都只从训练集拟合特征标准化与 one-hot 岭分类头，固定 λ 候选0.01/1/100；仅用验证准确率选 λ，平局按候选顺序选较小 λ，先保存配置再读取测试预测，不重训 train+validation。所有缓存键绑定视频、模型状态、源码、预处理、执行环境和协议哈希。

| 方法 | 验证准确率 | 测试准确率 | 测试 macro-F1 | 测试 N |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

单个seed的有限子集结果不能证明统计显著性或完整 Libras 数据集能力。历史分组只使用 local_informant，跨条目全局签者身份尚未核实；保留旧 MediaPipe 就绪子集的选择偏差。RF 与本探针的输入表示和分类头都不同，因此 RF 表格只提供上下文；CE 与随机 MAM 之间的同头对照才对应当前协议下的预训练表示差异。这里没有 CE-CSL/Libras 词表对齐、中葡句子翻译或实时因果推理证据。
"""
    if report is not None:
        text += f"\n实际结果及逐类混淆/逐样本预测见 `transfer/result.json`。CE 相对随机的测试准确率差为 {report['comparison']['accuracy_difference_percentage_points']:.4f} 个百分点，macro-F1 差为 {report['comparison']['macro_f1_difference']:.6f}；该差值仅描述当前固定划分。\n"
    DOC.write_text(text, encoding="utf-8")


@contextmanager
def output_lock(path: Path, protocol_hash: str):
    """OS locks protect concurrent runs and are released even after process kill."""
    handle = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(handle, "r+b") as stream:
        if os.fstat(stream.fileno()).st_size < 1:
            stream.write(b" ")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("Another transfer process holds this output directory") from error
        try:
            stream.seek(1)
            stream.write(json.dumps({"pid": os.getpid(), "protocol_hash": protocol_hash, "started_at": utc()}).encode())
            stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def run(protocol_path: Path, checkpoint: Path, device: str) -> dict:
    protocol = load_protocol(protocol_path)
    output = within_transfer(protocol_path).parent
    with output_lock(output / ".running.lock", protocol["protocol_hash"]):
        return run_locked(protocol, checkpoint, device, output)


def run_locked(protocol: dict, checkpoint: Path, device: str, output: Path) -> dict:
    if (output / "result.json").exists():
        raise FileExistsError("Completed transfer results are immutable; use a new frozen protocol directory")
    started = time.perf_counter()
    state = {"status": "running", "pid": os.getpid(), "started_at": utc(), "protocol_hash": protocol["protocol_hash"]}
    write_json_atomic(output / "state.json", state)
    try:
        results, models = {}, {}
        for arm in ARMS:
            print(f"Probe {arm}: frozen visual features, identical ridge search; test is consulted after validation selection.", flush=True)
            state.update(active_arm=arm, stage="extracting_then_validation_selection")
            write_json_atomic(output / "state.json", state)
            matrix, ledger = embeddings(arm, protocol, checkpoint, device, output)
            head, result = select_head(matrix, protocol, output, arm)
            result["embedding_evidence"] = ledger
            models[arm], results[arm] = (head, matrix), result
        # Both immutable selection artifacts now exist; neither test metric
        # was consulted while selecting either arm's regularization value.
        state.update(stage="test_evaluation_after_both_selections")
        write_json_atomic(output / "state.json", state)
        for arm in ARMS:
            head, matrix = models[arm]
            results[arm]["test"] = evaluate_test(head, matrix, protocol)
            write_json_atomic(output / arm / "result.json", results[arm])
        trained, random = results[ARMS[0]]["test"], results[ARMS[1]]["test"]
        pairs = {"both_correct": 0, "pretrained_only_correct": 0, "random_only_correct": 0, "neither_correct": 0}
        for first, second in zip(trained["predictions"], random["predictions"]):
            a, b = first["truth"] == first["prediction"], second["truth"] == second["prediction"]
            key = "both_correct" if a and b else "pretrained_only_correct" if a else "random_only_correct" if b else "neither_correct"
            pairs[key] += 1
        report = {"schema_version": 1, "status": "complete", "created_at": utc(), "protocol_hash": protocol["protocol_hash"],
                  "arms": results, "historical_RF_context": protocol["identity"]["historical_RF_context"],
                  "comparison": {"accuracy_difference_percentage_points": 100 * (trained["accuracy"] - random["accuracy"]),
                                 "macro_f1_difference": trained["macro_f1"] - random["macro_f1"], "paired_test_outcomes": pairs},
                  "elapsed_seconds": time.perf_counter() - started, "limitations": protocol["identity"]["limitations"]}
        write_json_atomic(output / "result.json", report)
        write_doc(protocol, report)
        state.update(status="complete", elapsed_seconds=report["elapsed_seconds"])
        write_json_atomic(output / "state.json", state)
        return report
    except BaseException as error:
        state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                     error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started)
        write_json_atomic(output / "state.json", state)
        raise


def self_test() -> dict:
    # Analytical toy checks only validate helper math, not a model result.
    train = np.array([[-2., 5.], [-1., 5.], [1., 5.], [2., 5.]])
    targets = np.array([0, 0, 1, 1], dtype=np.int64)
    head = ridge_fit(train, targets, 2, 0.01)
    assert head["mean"].tolist() == [0., 5.]
    assert head["scale"][1] == 1.
    assert ridge_predict(head, train).tolist() == targets.tolist()
    result = metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]), ["a", "b"], ["1", "2", "3", "4"])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["accuracy"] == 0.75
    assert abs(result["macro_f1"] - (2 / 3 + 4 / 5) / 2) < 1e-12
    original_mean = head["mean"].copy()
    ridge_predict(head, np.full((2, 2), 1000.))
    assert np.array_equal(head["mean"], original_mean)
    from tempfile import TemporaryDirectory
    TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".helper-", dir=TRANSFER_ROOT) as directory:
        test_root = Path(directory).resolve()
        assert test_root.is_relative_to(TRANSFER_ROOT.resolve())
        lock = test_root / "test.lock"
        with output_lock(lock, "synthetic-helper-only"):
            try:
                with output_lock(lock, "synthetic-helper-only"):
                    raise AssertionError("Concurrent output lock unexpectedly acquired")
            except RuntimeError:
                pass
        with output_lock(lock, "synthetic-helper-only"):
            pass
        cache = test_root / "test.npz"
        atomic_npz(cache, embedding=np.arange(1024, dtype=np.float32))
        with np.load(cache, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["embedding"], np.arange(1024, dtype=np.float32))
    return {"status": "helper_checks_passed", "gpu_executed": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("self-test")
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    frozen.add_argument("--source-dir", type=Path, required=True)
    frozen.add_argument("--protocol", type=Path, default=TRANSFER_ROOT / "protocol.json")
    frozen.add_argument("--max-frames", type=int, default=600)
    frozen.add_argument("--max-seconds", type=float, default=15.)
    frozen.add_argument("--max-resource-bytes", type=int, default=128 * 1024 * 1024)
    execution = commands.add_parser("run")
    execution.add_argument("--protocol", type=Path, default=TRANSFER_ROOT / "protocol.json")
    execution.add_argument("--checkpoint", type=Path, required=True)
    execution.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.command == "self-test":
        result = self_test()
    elif args.command == "freeze":
        result = freeze(args.pilot, args.source_dir, args.protocol,
                        RGBInputSpec(max_frames=args.max_frames, max_seconds=args.max_seconds,
                                     max_resource_bytes=args.max_resource_bytes))
        result = {"protocol_hash": result["protocol_hash"], "status": result["status"], "samples": len(result["identity"]["samples"])}
    else:
        result = run(args.protocol, args.checkpoint, args.device)
        result = {"status": result["status"], "comparison": result["comparison"], "elapsed_seconds": result["elapsed_seconds"]}
    print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
