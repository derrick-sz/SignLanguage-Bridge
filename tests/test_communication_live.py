from contextlib import asynccontextmanager
import asyncio
import json
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from slbridge import communication_live as live
import httpx
from slbridge.communication import RecognitionGateway


def test_realtime_page_is_linked_and_its_modules_are_served():
    from slbridge.communication_routes import mount_communication_routes
    app=FastAPI();mount_communication_routes(app,None)
    with TestClient(app) as client:
        assert '/communicate/realtime.html' in client.get('/communicate/').text
        page=client.get('/communicate/realtime.html')
        assert page.status_code==200 and '实时词' in page.text
        for name in ['realtime-app.mjs','realtime-client.mjs','realtime-capture.mjs','realtime.css']:
            assert client.get('/communicate/'+name).status_code==200


def test_disabled_live_bridge_and_hostile_browser_origin():
    app=FastAPI();live.mount_live_bridge(app,None)
    with TestClient(app) as client:
        assert client.get('/v1/communication/realtime').json()['enabled'] is False
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/v1/communication/realtime'):pass
    app=FastAPI();gateway=SimpleNamespace(base_url='http://127.0.0.1:8765',token=None)
    live.mount_live_bridge(app,gateway)
    with TestClient(app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect('/v1/communication/realtime',headers={'origin':'https://elsewhere.invalid'}):pass


def test_live_bridge_forwards_frames_to_only_configured_backend_and_keeps_token_server_side():
    calls=[];received=[]
    class Upstream:
        def __init__(self):self.responses=asyncio.Queue()
        async def send(self,message):
            received.append(json.loads(message))
            await self.responses.put(json.dumps({'type':'ready','protocol':'csl-realtime-v1'}))
        async def recv(self):return await self.responses.get()
    @asynccontextmanager
    async def connect(url,**kwargs):
        calls.append((url,kwargs));yield Upstream()
    app=FastAPI();gateway=SimpleNamespace(base_url='http://127.0.0.1:8765',token='private-token')
    live.mount_live_bridge(app,gateway,connector=connect)
    with TestClient(app) as client:
        with client.websocket_connect('/v1/communication/realtime',headers={'origin':'http://testserver'}) as ws:
            ws.send_json({'type':'start'})
            result=ws.receive_json();assert result['type']=='ready'
            assert 'private-token' not in json.dumps(result)
    assert received==[{'type':'start'}]
    assert calls[0][0]=='ws://127.0.0.1:8765/v3/realtime'
    assert calls[0][1]['additional_headers']['Authorization']=='Bearer private-token'


def test_advice_polling_keeps_backend_tokens_off_the_browser():
    def handle(request):
        assert request.url.path=='/v3/realtime/assistance/'+'a'*32
        assert request.headers['authorization']=='Bearer private-token'
        return httpx.Response(200,json={'job_id':'a'*32,'status':'risk','text':'请核对。','secret':'never-forward'})
    gateway=RecognitionGateway('http://127.0.0.1:8765',token='private-token',transport=httpx.MockTransport(handle))
    app=FastAPI();live.mount_live_bridge(app,gateway)
    with TestClient(app) as client:
        result=client.get('/v1/communication/realtime/assistance/'+'a'*32)
        assert result.status_code==200 and result.json()['status']=='risk'
        assert 'private-token' not in result.text and 'never-forward' not in result.text
