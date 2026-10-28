"""Optional same-origin routes; only registry-owned logical assets are served."""
from pathlib import Path
from typing import Annotated

from fastapi import HTTPException, Path as ApiPath
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .planner import plan_motion

_WEB = Path(__file__).with_name("web")
_STATIC = {"index.html", "workspace.js", "stage.js", "player-core.mjs", "workspace.css",
           "rig.js", "retarget.js", "stabilization.js"}
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


def install_animation_routes(app, registry):
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
