"""Product page and a bounded raw-video bridge; no model imports here."""
from pathlib import Path
import asyncio
import tempfile

from fastapi import HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from starlette.requests import ClientDisconnect

from .communication import RecognitionBusy, RecognitionContractChanged, RecognitionUnavailable, VideoInputError

_WEB = Path(__file__).with_name('animation')/'web'/'communication'
_STATIC = {'index.html', 'assist.html', 'meeting.html', 'app.js', 'product.css', 'conversation.mjs', 'recognition-controller.mjs', 'capture.mjs',
           'meeting-app.js', 'meeting.css', 'selection.mjs', 'screen-capture.mjs', 'caption-session.mjs',
           'realtime.html', 'realtime.css', 'realtime-app.mjs', 'realtime-client.mjs', 'realtime-capture.mjs', 'realtime-demo.mjs',
           'conversation-transfer.mjs'}


def mount_communication_routes(app, gateway, *, demo_catalog=None):
    from .communication_demos import mount_demo_routes
    mount_demo_routes(app,demo_catalog)
    from .communication_live import mount_live_bridge
    mount_live_bridge(app, gateway)
    @app.get('/communicate/')
    @app.get('/communicate/{name}')
    def page(name: str = 'meeting.html'):
        if name not in _STATIC or not (_WEB/name).is_file():
            raise HTTPException(404, 'Unknown product resource')
        return FileResponse(_WEB/name, headers={'Cache-Control': 'no-cache', 'X-Content-Type-Options': 'nosniff'})

    @app.get('/v1/communication/recognition')
    async def recognition_status():
        if gateway is None:
            return {'enabled': False, 'ready': False, 'languages': [], 'reason': 'not_configured'}
        try:
            return await gateway.status()
        except RecognitionUnavailable:
            return {'enabled': True, 'ready': False, 'languages': [], 'reason': 'unavailable'}

    @app.post('/v1/communication/recognitions')
    async def recognize(request: Request, sign_language: str = Query(pattern='^(csl|libras)$'),
                        request_id: str = Query(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9][A-Za-z0-9_.:-]*$'),
                        filename: str = Query(min_length=1, max_length=200)):
        if gateway is None:
            raise HTTPException(503, '识别服务尚未配置。文字回复仍可使用。')
        suffix = Path(filename).suffix.lower()
        if suffix not in gateway.extensions or any(c in filename for c in '/\\\r\n\x00'):
            raise HTTPException(422, '请选择MP4、MOV、AVI、MKV或WebM视频。')
        if gateway.lock.locked():
            raise HTTPException(429, '上一段还在识别，请稍后再试。')
        async with gateway.lock:
            try:
                content_length = request.headers.get('content-length')
                try:
                    declared_size = int(content_length) if content_length is not None else None
                except ValueError as exc:
                    raise HTTPException(400, '无法读取视频长度，请重新上传。') from exc
                if declared_size is not None and declared_size > gateway.max_upload_bytes:
                    raise HTTPException(413, '视频文件过大，请使用128MB以内的短视频。')
                with tempfile.TemporaryDirectory(prefix='signbridge-product-', dir=gateway.temporary_root) as directory:
                    source = Path(directory)/('video'+suffix)
                    size = 0
                    with source.open('wb') as output:
                        async for chunk in request.stream():
                            size += len(chunk)
                            if size > gateway.max_upload_bytes:
                                raise HTTPException(413, '视频文件过大，请使用128MB以内的短视频。')
                            output.write(chunk)
                    if not size:
                        raise HTTPException(422, '视频内容为空，请重新选择。')
                    task = asyncio.create_task(gateway.recognize(source, sign_language, request_id, filename))
                    try:
                        while not task.done():
                            if await request.is_disconnected():
                                raise ClientDisconnect()
                            await asyncio.wait({task}, timeout=.1)
                        return await task
                    finally:
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
            except ClientDisconnect:
                return Response(status_code=499)
            except RecognitionBusy as exc:
                raise HTTPException(429, str(exc)) from exc
            except RecognitionContractChanged as exc:
                raise HTTPException(409, str(exc)) from exc
            except RecognitionUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            except VideoInputError as exc:
                raise HTTPException(422, str(exc)) from exc
