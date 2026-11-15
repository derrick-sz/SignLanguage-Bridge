from contextlib import contextmanager
import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from slbridge import realtime_routes as r
from slbridge.realtime_worker import LiveSession
from tests.test_realtime_protocol import frame


class Service:
    def __init__(self):self.sessions=[]
    def status(self):return {'enabled':True,'ready':True}
    def create_session(self, start):
        @contextmanager
        def factory(session):
            class Engine:
                info={'model_id':'fixture'}
                def feed(self, value):session.emit('word',gloss='我',status='partial')
                def finish(self):pass
                def cancel(self):pass
            yield Engine()
        s=LiveSession(source_epoch=start['source_epoch'],engine_factory=factory)
        self.sessions.append(s);return s


def client(service=None, token=None):
    app=FastAPI();r.mount_realtime_routes(app,lambda:service,bearer_token=token)
    return TestClient(app)


def start(ws):
    ws.send_json(dict(type='start',protocol='csl-realtime-v1',sign_language='csl',source_epoch=3,fps=30))
    assert ws.receive_json()['type']=='ready'


def test_disabled_route_does_not_advertise_live_inference():
    with client() as c:
        assert c.get('/v3/realtime').json()['enabled'] is False
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect('/v3/realtime'):pass


def test_websocket_checks_token_and_origin_before_starting_a_worker():
    service=Service()
    with client(service,'secret') as c:
        for headers in [{}, {'authorization':'Bearer wrong'},
                        {'authorization':'Bearer secret','origin':'https://hostile.invalid'}]:
            with pytest.raises(WebSocketDisconnect):
                with c.websocket_connect('/v3/realtime',headers=headers):pass
    assert service.sessions==[]


def test_incremental_frame_events_and_natural_finish():
    service=Service()
    with client(service) as c:
        with c.websocket_connect('/v3/realtime',headers={'origin':'http://testserver'}) as ws:
            start(ws);ws.send_json(frame());kinds=[]
            while 'ack' not in kinds:kinds.append(ws.receive_json()['type'])
            assert 'word' in kinds
            ws.send_json({'type':'finish'})
            assert ws.receive_json()['type']=='finished'
    assert service.sessions[0].join(1)


def test_bad_epoch_stops_the_worker_and_never_flushes_a_sentence():
    service=Service()
    with client(service) as c:
        with c.websocket_connect('/v3/realtime') as ws:
            start(ws);ws.send_json(frame(epoch=4))
            assert ws.receive_json()['type']=='error'
    assert service.sessions[0].join(1)


def test_advice_jobs_can_be_read_after_the_native_stream_has_finished():
    service=Service();service.assistance_result=lambda job:{'status':'risk','text':'请核对。'} if job=='a'*32 else None
    with client(service) as c:
        assert c.get('/v3/realtime/assistance/'+'a'*32).json()['status']=='risk'
        assert c.get('/v3/realtime/assistance/not-a-job').status_code==404
