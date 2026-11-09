"""Free-text generation transport; runtime paths remain server-owned."""
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import HTTPException, Path as ApiPath
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

_WEB=Path(__file__).with_name('web')
_STATIC={'generate.html','generate.js','generate.css','generate-controller.mjs','fast-composition-controller.mjs'}


class GenerationInput(BaseModel):
    model_config=ConfigDict(extra='forbid')
    text: str=Field(min_length=1,max_length=200)
    seconds: StrictInt=6

    @field_validator('seconds')
    @classmethod
    def seconds_allowed(cls,value):
        if value not in (4,6,8,10):
            raise ValueError('Use 4, 6, 8 or 10 seconds')
        return value


def mount_text_generation_routes(app, service, *, home_path='/generate/'):
    app.state.text_generation_enabled=True

    def call(method,*args,**kwargs):
        if service is None:
            raise HTTPException(503,'生成环境尚未配置。')
        from .text_generation_jobs import GenerationBusyError,GenerationUnavailableError
        try:
            return getattr(service,method)(*args,**kwargs)
        except KeyError as exc:
            raise HTTPException(404,'生成任务不存在或已过期。') from exc
        except GenerationBusyError as exc:
            raise HTTPException(429,'生成队列已满，请稍后再试。') from exc
        except GenerationUnavailableError as exc:
            raise HTTPException(503,'生成环境暂时不可用，请检查本机生成配置。') from exc
        except ValueError as exc:
            message='动作尚未完成，不能播放。' if method=='motion' else '请输入1–200字，演示时长选择4、6、8或10秒。'
            raise HTTPException(409 if method=='motion' else 422,message) from exc

    @app.get('/v1/animation/generation')
    def capabilities():
        return {'enabled':service is not None,'max_characters':200,'seconds':[4,6,8,10],
                'language_review':'NOT_RUN'}

    @app.post('/v1/animation/generation/jobs',status_code=202)
    def submit(body:GenerationInput):
        return call('submit',body.text,seconds=body.seconds)

    @app.get('/v1/animation/generation/jobs/{job_id}')
    def status(job_id:Annotated[str,ApiPath(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')]):
        return call('get',job_id)

    @app.delete('/v1/animation/generation/jobs/{job_id}')
    def cancel(job_id:Annotated[str,ApiPath(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')]):
        return call('cancel',job_id)

    @app.get('/v1/animation/generation/jobs/{job_id}/motion')
    def motion(job_id:Annotated[str,ApiPath(min_length=1,max_length=64,pattern=r'^[A-Za-z0-9_-]+$')]):
        return call('motion',job_id)

    @app.get('/')
    def home():
        return RedirectResponse(home_path)

    @app.get('/generate/')
    @app.get('/generate/{name}')
    def page(name:str='generate.html'):
        if name not in _STATIC or not (_WEB/name).is_file():
            raise HTTPException(404,'Unknown generation resource')
        return FileResponse(_WEB/name,headers={'Cache-Control':'no-cache',
                            'X-Content-Type-Options':'nosniff','Referrer-Policy':'no-referrer'})

    if service is not None:
        previous=app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(application):
            try:
                async with previous(application):
                    yield
            finally:
                service.close()
        app.router.lifespan_context=lifespan
