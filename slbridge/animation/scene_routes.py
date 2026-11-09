"""Local scene browsing and server-owned full-expression preview plans."""
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Annotated

from fastapi import HTTPException, Path as ApiPath, Query
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, StrictBool

from .corpus_store import CorpusIntegrityError

_WEB = Path(__file__).with_name("web")
_STATIC = {"scenes.html", "scenes.js", "scenes.css", "scenes-controller.mjs"}


class ScenePlanInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=500)
    confirmed: StrictBool = False
    research_preview: StrictBool = False


def mount_scene_routes(app, library):
    plans, lock = OrderedDict(), RLock()

    def call(method, *args, **kwargs):
        try:
            return getattr(library, method)(*args, **kwargs)
        except CorpusIntegrityError as exc:
            raise HTTPException(409, "动作来源或版本已改变，请重新规划。") from exc
        except (sqlite3.Error, OSError) as exc:
            raise HTTPException(503, "场景语料暂时无法读取。") from exc
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, "场景请求或数据无效。") from exc

    @app.get("/v1/animation/scenes")
    def catalog(scene: str | None = Query(None, max_length=64),
                q: str = Query("", max_length=500),
                limit: int = Query(30, ge=1, le=100),
                offset: int = Query(0, ge=0, le=2**63 - 1)):
        return call("catalog", scene=scene, q=q, limit=limit, offset=offset)

    @app.get("/v1/animation/scenes/coverage")
    def coverage():
        return call("coverage")

    @app.post("/v1/animation/scenes/plans")
    def create_plan(body: ScenePlanInput):
        plan = call("plan", body.text, confirmed=body.confirmed,
                    research_preview=body.research_preview)
        if plan["can_play"]:
            with lock:
                plans[plan["plan_id"]] = deepcopy(plan)
                plans.move_to_end(plan["plan_id"])
                while len(plans) > 128:
                    plans.popitem(last=False)
        return plan

    @app.get("/v1/animation/scenes/plans/{plan_id}/motion")
    def motion(plan_id: Annotated[str, ApiPath(min_length=1, max_length=128)]):
        with lock:
            plan = deepcopy(plans.get(plan_id))
        if plan is None:
            raise HTTPException(404, "预览计划不存在或已过期，请重新规划。")
        return call("motion", plan)

    @app.get("/scenes/")
    @app.get("/scenes/{name}")
    def page(name: str = "scenes.html"):
        if getattr(app.state, 'text_generation_enabled', False) and name == 'scenes.html':
            return RedirectResponse('/generate/')
        if name == 'library':
            name = 'scenes.html'
        if name not in _STATIC or not (_WEB / name).is_file():
            raise HTTPException(404, "Unknown scene resource")
        return FileResponse(_WEB / name, headers={"Cache-Control": "no-cache",
                            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})
