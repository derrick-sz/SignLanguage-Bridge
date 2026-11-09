from importlib import import_module
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


class Service:
    def __init__(self):
        self.jobs = {}
    def submit(self, text, seconds=6):
        if seconds not in (4,6,8,10):
            raise ValueError('bad seconds')
        if text == 'busy':
            from slbridge.animation.text_generation_jobs import GenerationBusyError
            raise GenerationBusyError('busy')
        job = {'job_id':'job1','text':text,'generation_text':text,'state':'queued','seconds':seconds}
        self.jobs['job1'] = job
        return job
    def get(self, job_id):
        return self.jobs[job_id]
    def cancel(self, job_id):
        self.jobs[job_id]['state']='cancelled'
        return self.jobs[job_id]
    def motion(self, job_id):
        if self.jobs[job_id]['state']!='complete':
            raise ValueError('not complete')
        return {'clip':{'text':self.jobs[job_id]['text']},'sha256':'a'*64}


@pytest.fixture
def client():
    app=FastAPI()
    import_module('slbridge.animation.text_generation_routes').mount_text_generation_routes(app, Service())
    return TestClient(app)


def test_out_of_corpus_text_queues_without_library_lookup(client):
    response=client.post('/v1/animation/generation/jobs',json={'text':'明天我要去图书馆学习。','seconds':6})
    assert response.status_code==202
    job=response.json()
    assert job['state']=='queued' and job['text']=='明天我要去图书馆学习。'
    assert client.get('/v1/animation/generation/jobs/job1').json()['text']==job['text']


def test_incomplete_motion_409_and_cancel(client):
    client.post('/v1/animation/generation/jobs',json={'text':'新的表达'})
    assert client.get('/v1/animation/generation/jobs/job1/motion').status_code==409
    assert client.delete('/v1/animation/generation/jobs/job1').json()['state']=='cancelled'


@pytest.mark.parametrize('suffix',['','/motion'])
def test_unknown_job_404(client,suffix):
    assert client.get('/v1/animation/generation/jobs/unknown'+suffix).status_code==404


@pytest.mark.parametrize('body',[{'text':''},{'text':'字'*201},{'text':'hello','seconds':5},
                                {'text':'hello','seconds':'6'},{'text':'hello','output':'C:/bad'}])
def test_invalid_body_422(client,body):
    assert client.post('/v1/animation/generation/jobs',json=body).status_code==422


def test_busy_429(client):
    assert client.post('/v1/animation/generation/jobs',json={'text':'busy'}).status_code==429


def test_missing_runtime_503():
    app=FastAPI()
    import_module('slbridge.animation.text_generation_routes').mount_text_generation_routes(app,None)
    c=TestClient(app)
    assert c.post('/v1/animation/generation/jobs',json={'text':'任意新句'}).status_code==503


def test_static_paths_are_fixed(client):
    assert client.get('/generate/private.json').status_code==404
