"""Motion content/provenance tests; no language-quality claim."""
import hashlib
import json
import math
import zipfile

import pytest

from slbridge.animation.registry import MotionRegistry
from slbridge.animation.planner import plan_motion


def make_archive(tmp_path, *, tamper=False, unsafe=False, approval=False, changed=None):
    def clip(identity, text):
        return {"id": identity, "text": text, "aliases": [text], "frames": 2, "fps": 25,
                "poses": [[[0., 0., 0.] for _ in range(55)] for _ in range(2)]}
    rest = {"parents": [-1] + [0] * 54, "restJoints": [[0., float(i), 0.] for i in range(55)]}
    reference = {**rest, "clips": [clip("hello", "你好！"), clip("thanks", "谢谢")]}
    fitted = clip("fit-thanks", "谢谢")
    fitted.update(source_sha256="a" * 64, fitted_sha256="b" * 64,
                  motion_approval_id="review-1", human_approved=True)
    pose_hash = hashlib.sha256(",".join(str(math.floor(v * 1e6 + 0.5))
        for frame in fitted["poses"] for joint in frame for v in joint).encode()).hexdigest()
    if changed == "poses":
        fitted["poses"][0][0][0] = 0.5
    if changed == "fit":
        fitted["fitted_sha256"] = "c" * 64
    payload = json.dumps(fitted).encode()
    files = {"demo/t08-motion.json": json.dumps(reference).encode(),
             "demo/t08-candidates.json": json.dumps({**rest, "clips": []}).encode(),
             "demo/experimental/fitted.json": payload,
             "demo/experimental-index.json": json.dumps({**rest, "clips": [
                 {k: v for k, v in fitted.items() if k != "poses"} | {
                     "asset": "experimental/fitted.json", "sha256": hashlib.sha256(payload).hexdigest(),
                     "motion_review_status": "approved_for_demo_motion", "human_approved": True}
             ]}).encode(), "avatar/model.glb": b"test-model",
             "avatar/three.module.js": b"export const test = 1;",
             "avatar/jsm/loaders/GLTFLoader.js": b"export class GLTFLoader {}",
             "avatar/jsm/utils/BufferGeometryUtils.js": b"export const mergeVertices = () => {};"}
    if approval:
        files["demo/experimental/motion-approval.json"] = json.dumps({
            "id": "review-1", "scope": "motion_quality_only", "clips": [{
                "id": "fit-thanks", "source_sha256": "a" * 64,
                "fitted_sha256": "b" * 64, "poses_sha256": pose_hash}]}).encode()
    manifest = {"files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    if tamper:
        files["avatar/model.glb"] = b"changed"
    if unsafe:
        files["../escape"] = b"unsafe"
    path = tmp_path / "motion.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
        z.writestr("BUILD.json", json.dumps(manifest))
    return path


@pytest.fixture
def registry(tmp_path):
    return MotionRegistry(make_archive(tmp_path))


@pytest.mark.parametrize("changed", [None, "poses", "fit"])
def test_review_is_bound_to_fitted_motion_not_only_source_video(tmp_path, changed):
    registry = MotionRegistry(make_archive(tmp_path, approval=True, changed=changed))
    asset = next(a for a in registry.catalog()["assets"] if a["id"] == "fit-thanks")
    assert asset["motion_review"] == ("demo_motion_only" if changed is None else "unverified")


def test_registry_preserves_motion_source_and_marks_language_unverified(registry):
    catalog = registry.catalog()
    assert len(catalog["assets"]) == 3
    item = next(a for a in catalog["assets"] if a["id"] == "hello")
    assert item["language_review"] == "unverified"
    motion = registry.motion("hello")
    assert motion["clip"]["frames"] == 2
    assert len(motion["parents"]) == 55
    motion["clip"]["poses"][0][0][0] = 999
    assert registry.motion("hello")["clip"]["poses"][0][0][0] == 0


def test_exact_full_phrase_requires_confirmation_and_research_preview(registry):
    result = plan_motion(registry, "你好")
    assert result["status"] == "needs_confirmation"
    assert not result["can_play"]
    assert not plan_motion(registry, "你好", confirmed=True)["can_play"]
    ready = plan_motion(registry, "你好", confirmed=True, research_preview=True,
                        origin={"request_id": "r1", "raw_text": "你好"})
    assert ready["status"] == "research_preview"
    assert ready["can_play"]
    assert ready["items"][0]["asset_id"] == "hello"
    assert ready["origin"]["request_id"] == "r1"
    assert ready["language_review"] == "unverified"


@pytest.mark.parametrize("text", ["你好不要", "你好2", "谢谢你", "完全未覆盖", "不你好"])
def test_partial_content_never_disappears(registry, text):
    plan = plan_motion(registry, text, confirmed=True, research_preview=True)
    assert plan["status"] == "blocked"
    assert plan["unmatched"] == [text]
    assert not plan["items"]
    assert not plan["can_play"]


def test_ambiguous_alias_requires_explicit_asset(registry):
    plan = plan_motion(registry, "谢谢", confirmed=True, research_preview=True)
    assert plan["status"] == "ambiguous"
    assert len(plan["candidates"]) == 2
    assert not plan["can_play"]
    chosen = plan_motion(registry, "谢谢", asset_id="fit-thanks", confirmed=True, research_preview=True)
    assert chosen["can_play"]
    assert chosen["items"][0]["asset_id"] == "fit-thanks"
    assert not plan_motion(registry, "不谢谢", asset_id="fit-thanks",
                           confirmed=True, research_preview=True)["can_play"]


@pytest.mark.parametrize("kwargs", [{"tamper": True}, {"unsafe": True}])
def test_archive_rejects_modified_or_unsafe_members(tmp_path, kwargs):
    with pytest.raises(ValueError):
        MotionRegistry(make_archive(tmp_path, **kwargs))


def test_archive_hash_and_fixed_resource_boundary(tmp_path, registry):
    path = make_archive(tmp_path)
    with pytest.raises(ValueError):
        MotionRegistry(path, expected_sha256="0" * 64)
    with pytest.raises(KeyError):
        registry.resource("../BUILD.json")
    with pytest.raises(KeyError):
        registry.motion("../hello")
    assert registry.resource("model.glb")[0] == b"test-model"


@pytest.mark.parametrize("value", ["", " " * 3, "x" * 501])
def test_empty_and_overlong_inputs_rejected(registry, value):
    with pytest.raises(ValueError):
        plan_motion(registry, value)
