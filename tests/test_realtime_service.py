import json

from fastapi.testclient import TestClient
import pytest

from slbridge import realtime_service as m
from slbridge.api import create_app
from slbridge.runtime import ModelManager, RuntimeBusy
from tests.test_runtime import Model


def config(tmp_path,**changes):
    slrt=tmp_path/'slrt.json';slrt.write_text(json.dumps({'device':'cpu','pose_backend':'rtmpose-experimental'}))
    path=tmp_path/'realtime.json';path.write_text(json.dumps(dict(schema_version=1,slrt_config_file='slrt.json',**changes)))
    return m.RealtimeConfig.from_file(path)


class Runtime:
    info={}
    def __init__(self,*args):self.closed=False
    def predict(self,*args):return None
    def clear_frame_cache(self):pass
    def close(self):self.closed=True


def model():
    native=Model('native');native.live_pose_extractor=lambda **kwargs:None
    native.bundle.manifest.sign_language='csl'
    native.translate_pose_arrays=lambda *args,**kwargs:None
    return native


def test_server_config_resolves_slrt_path_and_rejects_unknown_limits(tmp_path):
    c=config(tmp_path);assert c.slrt.device=='cpu' and c.queue_size<=32
    with pytest.raises(ValueError):config(tmp_path,queue_size=10000)
    with pytest.raises(ValueError):config(tmp_path,endpoint='http://not-a-server-config-field')


def test_single_session_reservation_is_busy_even_before_worker_start(tmp_path):
    manager=ModelManager(initial_recognizer=model())
    service=m.RealtimeService(manager,config(tmp_path),runtime_factory=Runtime);service.warmup()
    start=dict(source_epoch=1,fps=30)
    first=service.create_session(start)
    with pytest.raises(RuntimeBusy):service.create_session(start)
    assert service.status()['busy']
    service.close();assert service.runtime is None


def test_model_switch_disables_old_coordinator_without_touching_libras(tmp_path):
    manager=ModelManager(initial_recognizer=model());service=m.RealtimeService(manager,config(tmp_path),runtime_factory=Runtime)
    service.warmup();manager.switch('next',loader=lambda _:model())
    assert not service.status()['ready']
    with pytest.raises(RuntimeBusy):service.create_session(dict(source_epoch=1,fps=30))
    service.close();manager.close()


def test_api_default_advertises_live_as_disabled():
    manager=ModelManager(initial_recognizer=Model('existing'))
    with TestClient(create_app({},manager=manager)) as client:
        response=client.get('/v3/realtime')
        assert response.status_code==200 and response.json()['enabled'] is False


def test_cli_does_not_silently_drop_the_opt_in_live_configuration(tmp_path,monkeypatch):
    import slbridge.api as api
    from slbridge.api_cli import main
    import uvicorn
    catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps({'schema_version':1,'models':{'test':'weights'}}))
    received=[]
    monkeypatch.setattr(api,'create_app',lambda *a,**kw:received.append(kw) or object())
    monkeypatch.setattr(uvicorn,'run',lambda *a,**kw:None)
    path=tmp_path/'live.json'
    assert main(['--catalog',str(catalog),'--realtime-config',str(path)])==0
    assert received[0].get('realtime_config')==path
