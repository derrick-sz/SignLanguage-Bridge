# E00 audit helper: runtime/environment probe. Read-only; no model loading, no inference.
import json, platform, sys, os, shutil, subprocess, importlib

out = {
    "python_executable": sys.executable,
    "python_version": platform.python_version(),
    "python_bits": platform.architecture()[0],
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
}
# package versions (no heavy model load)
mods = {}
for name in ["torch", "torchvision", "numpy", "cv2", "onnxruntime", "transformers",
             "rtmlib", "slbridge", "slbridge.slrt", "PIL", "scipy", "sklearn",
             "PySide6", "mmpose", "mmdet", "mmcv"]:
    try:
        m = importlib.import_module(name)
        mods[name] = {"import": "OK", "version": getattr(m, "__version__", None),
                      "file": getattr(m, "__file__", None)}
    except Exception as e:
        mods[name] = {"import": "FAIL", "error": f"{type(e).__name__}: {e}"}
out["modules"] = mods
# torch details without allocating a tensor
try:
    import torch
    out["torch"] = {
        "version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as e:
    out["torch"] = {"error": f"{type(e).__name__}: {e}"}
# ffmpeg/ffprobe
for exe in ["ffmpeg", "ffprobe"]:
    p = shutil.which(exe)
    ver = None
    if p:
        try:
            ver = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
        except Exception as e:
            ver = f"ERROR {e}"
    out[exe] = {"which": p, "version_line": ver}
print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
