"""Scene HTTP behavior; use an in-memory library to isolate request handling."""
from importlib import import_module

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


class Library:
    def catalog(self, **kwargs):
        return {"scenes": [], "assets": [], "total": 0, **kwargs}

    def coverage(self):
        return {"target_total": 96, "target_exact": 3}

    def plan(self, text, confirmed=False, research_preview=False):
        valid = text == "你好" and confirmed and research_preview
        return {"plan_id": "plan-1", "input_text": text, "can_play": valid,
                "status": "research_preview" if valid else "blocked", "items": []}

    def motion(self, plan):
        if getattr(self, "changed", False):
            from slbridge.animation.corpus_store import CorpusIntegrityError
            raise CorpusIntegrityError("changed")
        return {"clip": {"id": "test", "text": plan["input_text"]}, "sha256": "a" * 64}


@pytest.fixture
def env():
    module = import_module("slbridge.animation.scene_routes")
    app, lib = FastAPI(), Library()
    module.mount_scene_routes(app, lib)
    return TestClient(app), lib


def test_catalog_and_coverage(env):
    client, _ = env
    assert client.get("/v1/animation/scenes").json()["total"] == 0
    assert client.get("/v1/animation/scenes/coverage").json()["target_total"] == 96


def test_confirmed_plan_is_server_owned(env):
    client, _ = env
    result = client.post("/v1/animation/scenes/plans", json={
        "text": "你好", "confirmed": True, "research_preview": True})
    assert result.status_code == 200
    pid = result.json()["plan_id"]
    motion = client.get(f"/v1/animation/scenes/plans/{pid}/motion")
    assert motion.status_code == 200
    assert motion.json()["clip"]["text"] == "你好"


def test_blocked_plan_cannot_be_played(env):
    client, _ = env
    result = client.post("/v1/animation/scenes/plans", json={
        "text": "未知内容", "confirmed": True, "research_preview": True}).json()
    assert not result["can_play"]
    assert client.get(f"/v1/animation/scenes/plans/{result['plan_id']}/motion").status_code == 404


def test_missing_confirmation_never_stores_playable_plan(env):
    client, _ = env
    result = client.post("/v1/animation/scenes/plans", json={"text": "你好"}).json()
    assert not result["can_play"]
    assert client.get(f"/v1/animation/scenes/plans/{result['plan_id']}/motion").status_code == 404


def test_unknown_plan_is_404(env):
    assert env[0].get("/v1/animation/scenes/plans/unknown/motion").status_code == 404


def test_source_drift_is_409(env):
    client, lib = env
    pid = client.post("/v1/animation/scenes/plans", json={
        "text": "你好", "confirmed": True, "research_preview": True}).json()["plan_id"]
    lib.changed = True
    assert client.get(f"/v1/animation/scenes/plans/{pid}/motion").status_code == 409


@pytest.mark.parametrize("body", [
    {"text": ""}, {"text": "x" * 501}, {"text": "你好", "confirmed": 1},
    {"text": "你好", "path": "C:/untrusted"},
])
def test_invalid_requests_are_422(env, body):
    assert env[0].post("/v1/animation/scenes/plans", json=body).status_code == 422


@pytest.mark.parametrize("query", ["limit=101", "offset=-1", "q=" + "x" * 501])
def test_invalid_paging_is_422(env, query):
    assert env[0].get("/v1/animation/scenes?" + query).status_code == 422


def test_static_allowlist(env):
    client, _ = env
    assert client.get("/scenes/not-allowed.txt").status_code == 404
