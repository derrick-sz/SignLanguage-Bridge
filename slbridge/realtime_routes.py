"""Optional live-session transport, separate from the v3 clip contract."""
from __future__ import annotations

import asyncio
import hmac

from fastapi import HTTPException, WebSocket
import re
from starlette.websockets import WebSocketDisconnect

from .realtime_protocol import FrameDecoder, PROTOCOL, parse_message, validate_start


def same_origin(websocket):
    origin = websocket.headers.get('origin')
    expected = ('https://' if websocket.url.scheme == 'wss' else 'http://') + websocket.url.netloc
    return origin is None or origin == expected


async def _message(websocket):
    message = await websocket.receive()
    if message['type'] == 'websocket.disconnect':
        raise WebSocketDisconnect(message.get('code', 1000))
    return parse_message(message.get('text'))


def mount_realtime_routes(app, service_getter, *, bearer_token=None):
    @app.get('/v3/realtime/assistance/{job_id}')
    async def advice_result(job_id: str):
        service = service_getter()
        if service is None or not re.fullmatch(r'[0-9a-f]{32}',job_id):
            raise HTTPException(404,'Unknown assistance job')
        result = service.assistance_result(job_id)
        if result is None:
            raise HTTPException(404,'Unknown or expired assistance job')
        return result

    @app.get('/v3/realtime')
    async def live_status():
        service = service_getter()
        return service.status() if service is not None else {
            'enabled':False, 'ready':False, 'protocol':PROTOCOL, 'reason':'not_configured'}

    @app.websocket('/v3/realtime')
    async def live(websocket: WebSocket):
        service = service_getter()
        authorized = (bearer_token is None or hmac.compare_digest(
            websocket.headers.get('authorization',''), 'Bearer '+bearer_token))
        if not authorized or not same_origin(websocket) or service is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        session = None
        tasks = []
        try:
            start = validate_start(await asyncio.wait_for(_message(websocket), 10))
            session = service.create_session(start)
            decoder = FrameDecoder(source_epoch=start['source_epoch'], source_id=session.session_id)
            session.start()

            async def receive_frames():
                while True:
                    data = await asyncio.wait_for(_message(websocket), 30)
                    kind = data.get('type')
                    if kind == 'frame':
                        frame = decoder.decode(data)
                        if not session.submit(frame):
                            return
                    elif kind in {'finish','stop','ping','boundary'} and set(data) == {'type'}:
                        if kind == 'finish':
                            session.finish()
                            return
                        if kind == 'stop':
                            session.cancel('client_stop')
                            return
                        if kind == 'boundary':
                            if not session.boundary():
                                return
                            continue
                        session.emit('pong')
                    else:
                        raise ValueError('Unknown live message')

            async def send_events():
                while True:
                    for event in session.poll():
                        await websocket.send_json(event)
                    if not session.is_running:
                        for event in session.poll():
                            await websocket.send_json(event)
                        return
                    await asyncio.sleep(.01)

            receiver = asyncio.create_task(receive_frames())
            sender = asyncio.create_task(send_events())
            tasks = [receiver, sender]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            if receiver in done:
                receiver.result()
                await asyncio.wait_for(asyncio.shield(sender), 30)
            else:
                sender.result()
        except (ValueError, TimeoutError) as error:
            if session is not None:
                session.cancel('invalid_input' if isinstance(error, ValueError) else 'timeout')
            await websocket.send_json(dict(type='error', protocol=PROTOCOL,
                code='invalid_input' if isinstance(error, ValueError) else 'timeout',
                message='实时输入无效或已超时，请重新开始。'))
        except WebSocketDisconnect:
            pass
        except RuntimeError:
            if session is None:
                await websocket.send_json(dict(type='error', protocol=PROTOCOL, code='runtime_busy',
                                               message='识别服务正在使用，请稍后重试。'))
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if session is not None and session.is_running:
                session.cancel('client_disconnected')
                # Native work keeps its model lease if this bounded wait expires.
                # The service stays busy until the worker's own finally runs.
                await asyncio.to_thread(session.join, 5)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass
