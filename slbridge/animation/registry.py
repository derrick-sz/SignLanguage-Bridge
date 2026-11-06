"""Validated immutable snapshot of a server-owned SignBridge motion archive."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RESOURCES = {
    "model.glb": ("avatar/model.glb", "model/gltf-binary"),
    "three.module.js": ("avatar/three.module.js", "text/javascript"),
    "GLTFLoader.js": ("avatar/jsm/loaders/GLTFLoader.js", "text/javascript"),
    "BufferGeometryUtils.js": ("avatar/jsm/utils/BufferGeometryUtils.js", "text/javascript"),
}


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pose_digest(poses) -> str:
    # Upstream approval v1: quantize to 1e-6, ties toward +infinity.
    values = (str(math.floor(float(v) * 1_000_000 + 0.5))
              for frame in poses for joint in frame for v in joint)
    return _digest(",".join(values).encode("ascii"))


def _number(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _skeleton(bundle):
    parents, rest = bundle.get("parents"), bundle.get("restJoints")
    if (not isinstance(parents, list) or len(parents) != 55 or parents[0] != -1
            or any(type(p) is not int or not 0 <= p < i for i, p in enumerate(parents[1:], 1))
            or not isinstance(rest, list) or len(rest) != 55
            or any(not isinstance(j, list) or len(j) != 3 or not all(map(_number, j)) for j in rest)):
        raise ValueError("Invalid motion skeleton")
    return parents, rest


def _validate_clip(clip):
    frames, fps, poses = clip.get("frames"), clip.get("fps"), clip.get("poses")
    if (not isinstance(clip.get("id"), str) or not _ID.fullmatch(clip["id"])
            or not isinstance(clip.get("text"), str) or not 0 < len(clip["text"]) <= 500
            or type(frames) is not int or not 1 <= frames <= 3000
            or not _number(fps) or not 0 < fps <= 120
            or not isinstance(poses, list) or len(poses) != frames
            or any(not isinstance(f, list) or len(f) != 55 or
                   any(not isinstance(j, list) or len(j) != 3 or not all(map(_number, j)) for j in f)
                   for f in poses)):
        raise ValueError("Invalid motion clip")
    if 'refined_poses' in clip or 'gesture_refinement' in clip:
        refined=clip.get('refined_poses');profile=clip.get('gesture_refinement')
        if (not isinstance(profile,dict) or profile.get('profile')!='soke-gesture-refinement-v1'
                or not isinstance(refined,list) or len(refined)!=frames
                or any(not isinstance(f,list) or len(f)!=55 or
                       any(not isinstance(j,list) or len(j)!=3 or not all(map(_number,j)) for j in f)
                       for f in refined)):
            raise ValueError('Invalid refined gesture track')
    aliases = clip.get("aliases", [])
    if (not isinstance(aliases, list) or len(aliases) > 50
            or any(not isinstance(a, str) or not 0 < len(a) <= 500 for a in aliases)):
        raise ValueError("Invalid motion aliases")


class MotionRegistry:
    """Retains validated archive bytes so later file changes cannot alter a plan."""

    def __init__(self, archive: str | Path, expected_sha256: str | None = None):
        source = Path(archive)
        if source.stat().st_size > 128 * 1024 * 1024:
            raise ValueError("Motion archive exceeds size limit")
        raw = source.read_bytes()
        self.sha256 = _digest(raw)
        if expected_sha256 is not None and self.sha256 != expected_sha256.lower():
            raise ValueError("Motion archive checksum mismatch")
        self._clips, self._assets, self._resources = {}, [], {}
        with zipfile.ZipFile(BytesIO(raw)) as z:
            entries = z.infolist()
            names = [i.filename for i in entries]
            if (len(names) != len(set(names)) or len(names) > 1000
                    or sum(i.file_size for i in entries) > 256 * 1024 * 1024):
                raise ValueError("Invalid archive size or duplicate members")
            for entry in entries:
                n = entry.filename
                if ("\\" in n or n.startswith("/") or ":" in n or ".." in PurePosixPath(n).parts
                        or entry.file_size > 32 * 1024 * 1024):
                    raise ValueError("Unsafe archive member")
            manifest = json.loads(z.read("BUILD.json"))
            self._motion_attribution = manifest.get("motion_attribution", "Source archive approvals apply to motion quality only; language unverified.")
            if not isinstance(self._motion_attribution, str) or not 0 < len(self._motion_attribution) <= 2000:
                raise ValueError("Invalid motion attribution")
            files = manifest.get("files")
            if not isinstance(files, dict) or not files:
                raise ValueError("Missing archive file manifest")
            for name, checksum in files.items():
                if name not in names or _digest(z.read(name)) != checksum:
                    raise ValueError("Archive member checksum mismatch")

            def checked(name):
                if name not in files:
                    raise ValueError("Resource missing from archive manifest")
                return z.read(name)

            for public, (member, mime) in _RESOURCES.items():
                self._resources[public] = (checked(member), mime)
            for group, member in (("reference", "demo/t08-motion.json"),
                                  ("candidate", "demo/t08-candidates.json")):
                bundle = json.loads(checked(member))
                skeleton = _skeleton(bundle)
                for clip in bundle.get("clips", []):
                    self._add(group, clip, skeleton, _digest(checked(member)))
            index = json.loads(checked("demo/experimental-index.json"))
            skeleton = _skeleton(index)
            approval_member = "demo/experimental/motion-approval.json"
            approvals = {}
            approval_id = None
            if approval_member in names:
                approval = json.loads(checked(approval_member))
                if approval.get("scope") == "motion_quality_only":
                    approvals = {c["id"]: c for c in approval.get("clips", [])}
                    approval_id = approval.get("id")
            for entry in index.get("clips", []):
                member = "demo/" + entry["asset"]
                payload = checked(member)
                if _digest(payload) != entry["sha256"]:
                    raise ValueError("Motion index checksum mismatch")
                clip = json.loads(payload)
                if any(clip.get(k) != entry.get(k) for k in ("id", "text", "frames", "fps")):
                    raise ValueError("Motion index identity mismatch")
                _validate_clip(clip)
                reviewed = approvals.get(entry["id"], {})
                approved = ('refined_poses' not in clip and entry.get("human_approved") is True and bool(approval_id) and
                            entry.get("motion_approval_id") == clip.get("motion_approval_id") == approval_id and
                            all(isinstance(reviewed.get(k), str) and
                                re.fullmatch(r"[0-9a-f]{64}", reviewed[k]) and
                                reviewed[k] == entry.get(k) == clip.get(k)
                                for k in ("source_sha256", "fitted_sha256")) and
                            reviewed.get("poses_sha256") == _pose_digest(clip["poses"]))
                generated=clip.get('source_kind')=='pretrained_text_conditioned_generation'
                self._add("generated" if generated else "fitted", clip, skeleton, _digest(payload), approved=approved and not generated,
                          quality_warning=bool(entry.get("quality_warning")))
        if not self._assets:
            raise ValueError("Motion archive has no assets")

    def _add(self, group, clip, skeleton, content_hash, *, approved=False, quality_warning=False):
        _validate_clip(clip)
        identity = clip["id"]
        if identity in self._clips:
            raise ValueError("Duplicate motion identity")
        self._clips[identity] = {"clip": clip, "parents": skeleton[0], "restJoints": skeleton[1]}
        self._assets.append({
            "id": identity, "text": clip["text"], "aliases": list(clip.get("aliases", [])),
            "group": group, "frames": clip["frames"], "fps": clip["fps"],
            "duration": clip["frames"] / clip["fps"], "sha256": content_hash,
            "language_review": "unverified", "motion_review": "demo_motion_only" if approved else "unverified",
            "quality_warning": quality_warning, "sign_system": "CSL",
        })

    def catalog(self) -> dict:
        return {"schema_version": 1, "archive_sha256": self.sha256, "research_only": True,
                "assets": deepcopy(self._assets),
                "attribution": {"code": "SignBridge / MIT", "avatar": "Godette (Rigged), zahlenmaler / CC BY 4.0",
                                "avatar_source": "https://sketchfab.com/3d-models/godette-rigged-dd05b69799a2438e97c90d166f6e416a",
                                "motion": self._motion_attribution}}

    def motion(self, asset_id: str) -> dict:
        return deepcopy(self._clips[asset_id])

    def resource(self, name: str) -> tuple[bytes, str]:
        return self._resources[name]
