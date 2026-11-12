"""Loopback-only bridge for versioned live events; no model imports."""
from __future__ import annotations

import asyncio
import json
import re
from urllib.parse import urlsplit, urlunsplit

from fastapi import HTTPException, WebSocket
from starlette.websockets import WebSocketDisconnect

MAX_MESSAGE_BYTES = 256 * 1024


def mount_live_bridge(app, gateway, *, connector=None):
    @app.get('/v1/communication/realtime/assistance/{job_id}')
    async def advice_result(job_id: str):
        if gateway is None or not re.fullmatch(r'[0-9a-f]{32}',job_id):
            raise HTTPException(404,'Unknown assistance job')
        try:
            async with gateway.client() as client:
                response = await client.get('/v3/realtime/assistance/'+job_id,timeout=3)
                if response.status_code != 200:
                    raise HTTPException(404,'Unknown or expired assistance job')
                data = response.json()
                return {k:v for k,v in data.items() if k in {'job_id','status','text','model','session_id',
                    'source_epoch','utterance_id','revision','latency_ms','reason'}}
        except HTTPException:
            raise
        except Exception as error:
            raise HTTPException(503,'Assistance temporarily unavailable') from error

    @app.get('/v1/communication/realtime')
    async def status():
        unavailable = {'enabled':False, 'ready':False, 'protocol':'csl-realtime-v1'}
        if gateway is None:
            return dict(unavailable, reason='not_configured')
        try:
            async with gateway.client() as client:
                response = await client.get('/v3/realtime',timeout=3)
                if response.status_code != 200:
                    return dict(unavailable, reason='unavailable')
                data = response.json()
                return {k:v for k,v in data.items() if k in {
                    'enabled','ready','busy','protocol','reason','pipeline','input','features','limits','performance'}}
        except Exception:
            return dict(unavailable, reason='unavailable')

    @app.websocket('/v1/communication/realtime')
    async def stream(websocket: WebSocket):
        origin = websocket.headers.get('origin')
        expected = ('https://' if websocket.url.scheme == 'wss' else 'http://')+websocket.url.netloc
        if gateway is None or origin is not None and origin != expected:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        tasks = []
        try:
            connect = connector
            if connect is None:
                from websockets.asyncio.client import connect
            parsed = urlsplit(gateway.base_url)
            url = urlunsplit(('ws', parsed.netloc, '/v3/realtime', '', ''))
            headers = {'Authorization':'Bearer '+gateway.token} if gateway.token else {}
            async with connect(url, additional_headers=headers, proxy=None, open_timeout=5,
                               max_size=MAX_MESSAGE_BYTES, max_queue=8, write_limit=65536,
                               close_timeout=3) as upstream:
                async def send_frames():
                    while True:
                        message = await websocket.receive()
                        if message['type'] == 'websocket.disconnect':
                            return
                        text = message.get('text')
                        if not isinstance(text,str) or len(text.encode('utf-8')) > MAX_MESSAGE_BYTES:
                            raise ValueError('Invalid live message')
                        await upstream.send(text)

                async def receive_events():
                    while True:
                        message = await upstream.recv()
                        if not isinstance(message,str) or len(message.encode('utf-8')) > MAX_MESSAGE_BYTES:
                            raise ValueError('Invalid upstream message')
                        await websocket.send_text(message)

                tasks = [asyncio.create_task(send_frames()), asyncio.create_task(receive_events())]
                done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
        except WebSocketDisconnect:
            pass
        except Exception:
            try:
                await websocket.send_json({'type':'error','protocol':'csl-realtime-v1',
                    'code':'unavailable','message':'实时识别连接已结束，请检查服务后重新开始。'})
            except (RuntimeError, WebSocketDisconnect):
                pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            try:
                await websocket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass
