# E00 audit helper: data snapshot. Read-only. No full-video hashing.
import csv, hashlib, json, os, pickle, gzip, re, sys

ROOT = r"E:\githubproject\SignLanguage-Bridge"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def csv_info(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

out = {}
# --- CSVs
manifest = csv_info(os.path.join(ROOT, "experiments/p1/datasets/manifest_1086.csv"))
pilot = csv_info(os.path.join(ROOT, "experiments/p1/datasets/pilot_50.csv"))
gt = csv_info(os.path.join(ROOT, "experiments/p1/references/gt_1086.csv"))
def summarize(rows, name, path):
    present_col = "raw_video_present" if rows and "raw_video_present" in rows[0] else None
    d = {
        "path": os.path.relpath(path, ROOT).replace("\\", "/"),
        "rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
    }
    if present_col:
        d["raw_video_present_true"] = sum(1 for r in rows if str(r.get(present_col)).lower() == "true")
        d["raw_video_present_false"] = sum(1 for r in rows if str(r.get(present_col)).lower() != "true")
    if "split" in (rows[0] if rows else {}):
        splits = {}
        for r in rows:
            splits[r["split"]] = splits.get(r["split"], 0) + 1
        d["splits"] = splits
    if "cohort" in (rows[0] if rows else {}):
        coh = {}
        for r in rows:
            coh[r["cohort"]] = coh.get(r["cohort"], 0) + 1
        d["cohorts"] = coh
    if "usage" in (rows[0] if rows else {}):
        u = {}
        for r in rows:
            u[r["usage"]] = u.get(r["usage"], 0) + 1
        d["usage"] = u
    return d
out["manifest_1086"] = summarize(manifest, "manifest", os.path.join(ROOT, "experiments/p1/datasets/manifest_1086.csv"))
out["pilot_50"] = summarize(pilot, "pilot", os.path.join(ROOT, "experiments/p1/datasets/pilot_50.csv"))
out["gt_1086"] = summarize(gt, "gt", os.path.join(ROOT, "experiments/p1/references/gt_1086.csv"))
# unique sample ids / signers
out["manifest_1086"]["unique_sample_ids"] = len({r["sample_id"] for r in manifest})
out["pilot_50"]["unique_sample_ids"] = len({r["sample_id"] for r in pilot})
out["gt_1086"]["unique_sample_ids"] = len({r["sample_id"] for r in gt})
out["pilot_50"]["signers"] = len({r["signer_id"] for r in pilot})
out["manifest_1086"]["signers"] = len({r["signer_id"] for r in manifest})

# --- local CSL-Daily video directory
video_dir = r"D:\Zenodo\csl-daliy"
vid_count = 0
vid_bytes = 0
vid_names = set()
for base, dirs, files in os.walk(video_dir):
    for fn in files:
        if fn.lower().endswith(".mp4"):
            vid_count += 1
            vid_bytes += os.path.getsize(os.path.join(base, fn))
            vid_names.add(os.path.splitext(fn)[0])
out["local_csl_daily_videos"] = {"dir": video_dir, "mp4_files": vid_count, "total_bytes": vid_bytes}
# coverage of the 1086 manifest rows by local mp4 filename stem
out["local_csl_daily_videos"]["manifest_1086_matched_local"] = sum(1 for r in manifest if r["sample_id"] in vid_names)

# --- official pose archive (Uni-Sign author release)
pose_dir = os.path.join(ROOT, r"artifacts\cslnews-eval\csl-daily-pose\pose-format")
pose_count = 0
pose_bytes = 0
if os.path.isdir(pose_dir):
    for base, dirs, files in os.walk(pose_dir):
        for fn in files:
            pose_count += 1
            pose_bytes += os.path.getsize(os.path.join(base, fn))
out["official_pose_archive"] = {"dir": pose_dir, "files": pose_count, "total_bytes": pose_bytes,
                                "exists": os.path.isdir(pose_dir)}

# --- test_tmp_refs.txt
refs = os.path.join(ROOT, r"models\external\uni_sign_source\out\eval_csl_daily_pose_only\test_tmp_refs.txt")
with open(refs, "r", encoding="utf-8", errors="replace") as f:
    ref_lines = [ln.rstrip("\n") for ln in f]
out["upstream_test_tmp_refs"] = {
    "path": os.path.relpath(refs, ROOT).replace("\\", "/"),
    "sha256": sha256(refs),
    "bytes": os.path.getsize(refs),
    "lines": len(ref_lines),
    "nonempty_lines": sum(1 for ln in ref_lines if ln.strip()),
}
ref_names = set()
for ln in ref_lines:
    m = re.search(r"sample:\s*([^,\s]+)", ln)
    if m:
        ref_names.add(m.group(1))
out["upstream_test_tmp_refs"]["local_coverage_of_refs"] = sum(1 for n in ref_names if n in vid_names)
out["upstream_test_tmp_refs"]["refs_without_local_video"] = len(ref_names) - out["upstream_test_tmp_refs"]["local_coverage_of_refs"]

# --- F: model_weigth data files: size + sha (small ones only; big keypoints hash comes from PS run)
fdir = r"F:\model_weigth\data\csl-daily"
fsum = {}
for name in ["csl-daily-keypoints.pkl", "csl-daily.test", "csl-daily.train", "csl-daily.dev", "gloss2ids.pkl"]:
    p = os.path.join(fdir, name)
    if os.path.isfile(p):
        fsum[name] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    else:
        fsum[name] = {"missing": True}
out["f_model_weigth_data"] = fsum
# official sentence-level metadata (gzip pickle) and local coverage of official 1,176 test
for name in ["csl-daily.test", "csl-daily.train", "csl-daily.dev"]:
    p = os.path.join(fdir, name)
    try:
        with gzip.open(p, "rb") as f:
            obj = pickle.load(f)
        entry = {
            "entries": len(obj), "type": type(obj).__name__,
            "sample_keys": sorted(list(obj[0].keys())) if isinstance(obj, (list, tuple)) and obj else None,
        }
        if name == "csl-daily.test":
            test_names = {r["name"] for r in obj}
            entry["local_video_coverage"] = sum(1 for n in test_names if n in vid_names)
            entry["missing_local_video"] = len(test_names) - entry["local_video_coverage"]
        out.setdefault("official_sentence_meta", {})[name] = entry
    except Exception as e:
        out.setdefault("official_sentence_meta", {})[name] = {"error": f"{type(e).__name__}: {e}"}

# pose archive zip (author release, next to pose-format dir)
zip_path = os.path.join(ROOT, r"artifacts\cslnews-eval\csl-daily-pose\csl_daily_pose_format.zip")
out["official_pose_archive"]["zip_path"] = os.path.relpath(zip_path, ROOT).replace("\\", "/")
out["official_pose_archive"]["zip_bytes"] = os.path.getsize(zip_path) if os.path.isfile(zip_path) else None

print(json.dumps(out, ensure_ascii=False, indent=2, default=str))
