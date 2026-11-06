import pytest
from fastapi.testclient import TestClient

from slbridge.api import create_app
from slbridge.animation.registry import MotionRegistry
from test_animation_registry import make_archive


def test_optional_animation_does_not_change_existing_app():
    with TestClient(create_app({})) as client:
        assert client.get("/v1/models").status_code == 200
        assert client.get("/v1/animation/assets").status_code == 404


def test_same_origin_catalog_motion_and_confirmation(tmp_path):
    registry = MotionRegistry(make_archive(tmp_path))
    with TestClient(create_app({}, animation_registry=registry)) as client:
        catalog = client.get("/v1/animation/assets")
        assert catalog.status_code == 200
        assert len(catalog.json()["assets"]) == 3
        assert str(tmp_path) not in catalog.text
        body = {"text": "你好", "confirmed": True, "research_preview": True,
                "origin": {"kind": "recognition", "request_id": "r1", "raw_text": "你好",
                           "model_generation": 2, "source_epoch": 3}}
        plan = client.post("/v1/animation/plans", json=body)
        assert plan.status_code == 200, plan.text
        assert plan.json()["can_play"]
        assert plan.json()["origin"] == body["origin"]
        motion = client.get("/v1/animation/motions/hello")
        assert motion.status_code == 200
        assert motion.json()["clip"]["id"] == "hello"
        assert client.get("/v1/animation/motions/missing").status_code == 404
        assert client.get("/animation/vendor/model.glb").content == b"test-model"
        assert client.get("/animation/vendor/BUILD.json").status_code == 404
        assert client.get("/animation/vendor/jsm/loaders/GLTFLoader.js").status_code == 200
        assert client.get("/animation/vendor/jsm/utils/BufferGeometryUtils.js").status_code == 200
        assert client.get("/animation/vendor/model.glb").headers["x-content-type-options"] == "nosniff"


def test_bad_and_partial_plans_are_not_playable(tmp_path):
    with TestClient(create_app({}, animation_registry=MotionRegistry(make_archive(tmp_path)))) as client:
        assert client.post("/v1/animation/plans", json={"text": "你好"}).json()["status"] == "needs_confirmation"
        assert client.post("/v1/animation/plans", json={
            "text": "不你好", "confirmed": True, "research_preview": True}).json()["status"] == "blocked"
        assert client.post("/v1/animation/plans", json={"text": ""}).status_code == 422
        assert client.post("/v1/animation/plans", json={"text": "你好", "confirmed": "yes"}).status_code == 422
        assert client.post("/v1/animation/plans", json={"text": "你好", "path": "/secret"}).status_code == 422


def test_existing_bearer_policy_also_protects_animation(tmp_path):
    with TestClient(create_app({}, bearer_token="test-secret",
                              animation_registry=MotionRegistry(make_archive(tmp_path)))) as client:
        assert client.get("/v1/animation/assets").status_code == 401
        assert client.get("/v1/animation/assets", headers={"Authorization": "Bearer test-secret"}).status_code == 200
