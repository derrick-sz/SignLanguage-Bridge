"""Local quick-composition API; missing words never trigger slow generation."""
from contextlib import asynccontextmanager

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, StrictStr

from .word_composition import (CompositionBusyError, CompositionUnavailableError,
                               WordLibraryError)


class CompositionInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: StrictStr = Field(min_length=1, max_length=200)


def mount_word_composition_routes(app, service):
    @app.get('/v1/animation/composition')
    def capabilities():
        return service.status() if service is not None else {
            'enabled': False, 'planner': {'state': 'unconfigured'}, 'language_review': 'NOT_RUN'}

    @app.post('/v1/animation/composition')
    def compose(body: CompositionInput):
        if service is None:
            raise HTTPException(503, '快速组合环境尚未配置，请检查本机词库与文字规划器。')
        try:
            return service.compose(body.text)
        except CompositionBusyError as exc:
            raise HTTPException(429, '正在处理另一条输入，请稍后再试。') from exc
        except (CompositionUnavailableError, WordLibraryError) as exc:
            raise HTTPException(503, '快速组合资源暂时不可用，请检查本机服务状态。') from exc
        except ValueError as exc:
            raise HTTPException(422, '输入或动作长度超限，请使用1–200字的短句重新尝试。') from exc

    if service is not None:
        previous = app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(application):
            try:
                async with previous(application):
                    yield
            finally:
                service.close()
        app.router.lifespan_context = lifespan
