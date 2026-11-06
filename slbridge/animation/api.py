"""Optional same-origin routes; only registry-owned logical assets are served."""
from pathlib import Path
import sqlite3
from typing import Annotated

from fastapi import HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .planner import plan_motion

_WEB = Path(__file__).with_name("web")
_STATIC = {"index.html", "workspace.js", "stage.js", "player-core.mjs", "workspace.css",
           "rig.js", "retarget.js", "stabilization.js", "motion-geometry.mjs", "expression.mjs", "nonmanual.mjs"}
_VENDOR = {"model.glb": "model.glb", "three.module.js": "three.module.js",
           "jsm/loaders/GLTFLoader.js": "GLTFLoader.js",
           "jsm/utils/BufferGeometryUtils.js": "BufferGeometryUtils.js"}


class PlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)
    asset_id: str | None = Field(default=None, max_length=128)
    confirmed: StrictBool = False
    research_preview: StrictBool = False
    origin: dict | None = None


class CorpusPlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=500)
    confirmed: StrictBool = False
    research_preview: StrictBool = False
    origin: dict | None = None


def install_animation_routes(app, registry):
    def corpus():
        store = getattr(registry, 'corpus', None)
        if store is None:
            raise HTTPException(404, 'Full corpus is not configured')
        return store

    def corpus_call(method, *args, **kwargs):
        from .corpus_store import CorpusIntegrityError
        try:
            return getattr(corpus(), method)(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(404, 'Unknown corpus sample') from exc
        except CorpusIntegrityError as exc:
            raise HTTPException(409, str(exc)) from exc
        except sqlite3.Error as exc:
            raise HTTPException(503, 'Corpus index is temporarily unavailable') from exc
        except ValueError as exc:
            raise HTTPException(422, 'Invalid corpus request or metadata') from exc

    @app.get('/v1/animation/corpus')
    def corpus_status():
        return corpus_call('status') if getattr(registry, 'corpus', None) else {'enabled': False, 'schema_version': 1}

    @app.get('/v1/animation/corpus/assets')
    def corpus_assets(q: str = Query('', max_length=500), split: str | None = None,
                      quality: str = 'all', state: str = 'converted_candidate',
                      limit: int = Query(30, ge=1, le=100), offset: int = Query(0, ge=0)):
        return corpus_call('search', q=q, split=split, quality=quality, state=state, limit=limit, offset=offset)

    @app.get('/v1/animation/corpus/assets/{sample_id}')
    def corpus_asset(sample_id: Annotated[str, ApiPath(max_length=64)]):
        return corpus_call('detail', sample_id)

    @app.post('/v1/animation/corpus/plans')
    def corpus_plan(body: CorpusPlanInput):
        import json
        if body.origin is not None and len(json.dumps(body.origin)) > 16384:
            raise HTTPException(422, 'Origin metadata is too large')
        return corpus_call('plan', body.sample_id, body.text, confirmed=body.confirmed,
                           research_preview=body.research_preview, origin=body.origin)

    @app.get('/v1/animation/corpus/motions/{sample_id}')
    def corpus_motion(sample_id: Annotated[str, ApiPath(max_length=64)],
                      expected_sha256: str | None = Query(None, pattern=r'^[0-9a-f]{64}$')):
        return corpus_call('motion', sample_id, expected_sha256=expected_sha256)

    @app.get("/v1/animation/presentation")
    def presentation_catalog():
        bundle = getattr(registry, "presentation", None)
        return bundle.catalog() if bundle else {"schema_version": 1, "avatar": None, "pilots": [],
                                               "language_review": "unverified"}

    @app.get("/animation/presentation/{name}")
    def presentation_resource(name: str):
        bundle = getattr(registry, "presentation", None)
        try:
            if bundle is None:
                raise KeyError(name)
            data, mime = bundle.resource(name)
        except KeyError as exc:
            raise HTTPException(404, "Unknown presentation resource") from exc
        return Response(data, media_type=mime, headers={"X-Content-Type-Options": "nosniff",
                                                      "Cache-Control": "no-cache"})

    @app.get("/v1/animation/assets")
    def assets():
        return registry.catalog()

    @app.post("/v1/animation/plans")
    def plans(body: PlanInput):
        import json
        if body.origin is not None and len(json.dumps(body.origin)) > 16384:
            raise HTTPException(422, "Origin metadata is too large")
        try:
            return plan_motion(registry, body.text, asset_id=body.asset_id, confirmed=body.confirmed,
                               research_preview=body.research_preview, origin=body.origin)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/v1/animation/motions/{asset_id}")
    def motion(asset_id: Annotated[str, ApiPath(max_length=128)]):
        try:
            return registry.motion(asset_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown motion asset") from exc

    @app.get("/animation/vendor/{name:path}")
    def vendor(name: str):
        if name not in _VENDOR:
            raise HTTPException(404, "Unknown renderer resource")
        data, mime = registry.resource(_VENDOR[name])
        return Response(data, media_type=mime, headers={
            "X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=3600"})

    @app.get("/animation/")
    @app.get("/animation/static/{name}")
    def workspace(name: str = "index.html"):
        if name not in _STATIC or not (_WEB / name).is_file():
            raise HTTPException(404, "Unknown workspace resource")
        return FileResponse(_WEB / name, headers={"X-Content-Type-Options": "nosniff",
                                                 "Cache-Control": "no-cache",
                                                 "Referrer-Policy": "no-referrer"})
