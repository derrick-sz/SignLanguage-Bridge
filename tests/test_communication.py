import asyncio
import json
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from slbridge.communication import RecognitionGateway, RecognitionUnavailable
from slbridge.communication_routes import mount_communication_routes


def backend(*, wrong_generation=False, wrong_language=False, task='gloss_sequence'):
    calls=[]
    def handle(request):
        calls.append(request)
        cap={'sign_language':'csl','task_type':task,'model_id':'csl-model','inputs':[{'representation_id':'tfnet_rgb_v1','preprocessing_version':'pre-v1','input_kinds':['file']}],'vocabulary':['你','好']}
        if request.url.path=='/v3/models':
            return httpx.Response(200,json={'capabilities':{'csl':cap},'status':{'csl':{'state':'ready','generation':3}}})
        if request.url.path=='/v3/models/negotiate':
            assert json.loads(request.content)['sign_language']=='csl'
            return httpx.Response(200,json={'input_contract':cap['inputs'][0],'model_generation':3,'capabilities':cap})
        assert request.url.path=='/v3/recognitions'
        content=request.read().decode('utf-8',errors='replace')
        assert 'expected_generation' in content and 'pre-v1' in content and 'tfnet_rgb_v1' in content
        request_id=next(part.split('\r\n\r\n')[1].split('\r\n')[0] for part in content.split('--') if 'name="request_id"' in part)
        prediction={'tokens':[{'gloss':'你'},{'gloss':'好'}]} if task=='gloss_sequence' else {'text':'你好','language':'zh-CN'}
        return httpx.Response(200,json={'request_id':request_id,'model_generation':4 if wrong_generation else 3,'model_id':'csl-model','model_version':'model-v1','preprocessing_version':'pre-v1','result':{'status':'recognized','task_type':task,'sign_language':'libras' if wrong_language else 'csl','prediction':prediction}})
    return httpx.MockTransport(handle),calls


def app_for(tmp_path,**kwargs):
    transport,calls=backend(**kwargs)
    gateway=RecognitionGateway('http://127.0.0.1:8765',transport=transport,temporary_root=tmp_path,max_upload_bytes=1024)
    app=FastAPI();mount_communication_routes(app,gateway)
    return TestClient(app),calls


def test_real_contract_is_negotiated_and_gloss_is_not_relabelled_as_translation(tmp_path):
    client,calls=app_for(tmp_path)
    with client:
        status=client.get('/v1/communication/recognition').json()
        assert status['ready'] and status['languages'][0]['task_type']=='gloss_sequence'
        response=client.post('/v1/communication/recognitions?sign_language=csl&request_id=r1&filename=clip.mp4',content=b'video-fixture')
    assert response.status_code==200,response.text
    data=response.json()
    assert data['text']=='你 好' and data['output_kind']=='gloss_sequence'
    assert data['review_status']=='unverified' and data['provenance']['model_generation']==3
    assert not list(tmp_path.iterdir()),'temporary uploads must be removed'
    assert [r.url.path for r in calls][-2:]==['/v3/models/negotiate','/v3/recognitions']


@pytest.mark.parametrize('kwargs',[{'wrong_generation':True},{'wrong_language':True}])
def test_changed_model_or_wrong_language_never_reaches_product_text(tmp_path,kwargs):
    client,_=app_for(tmp_path,**kwargs)
    with client:response=client.post('/v1/communication/recognitions?sign_language=csl&request_id=r2&filename=c.mp4',content=b'clip')
    assert response.status_code==409
    assert '你 好' not in response.text
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('body,filename,status',[(b'', 'c.mp4',422),(b'x'*1025,'c.mp4',413),(b'clip','weights.npy',422)])
def test_empty_oversized_and_nonvideo_inputs_do_not_call_inference(tmp_path,body,filename,status):
    client,calls=app_for(tmp_path)
    with client:response=client.post('/v1/communication/recognitions',params={'sign_language':'csl','request_id':'r3','filename':filename},content=body)
    assert response.status_code==status
    assert not any(c.url.path=='/v3/recognitions' for c in calls)
    assert not list(tmp_path.iterdir())


def test_missing_recognition_service_keeps_product_page_available():
    app=FastAPI();mount_communication_routes(app,None)
    with TestClient(app) as client:
        assert client.get('/communicate/').status_code==200
        assert client.get('/v1/communication/recognition').json()['ready'] is False
        assert client.post('/v1/communication/recognitions?sign_language=csl&filename=a.mp4&request_id=r4',content=b'c').status_code==503


@pytest.mark.parametrize('url',['https://example.com','http://127.0.0.1:8765/private','http://name:password@localhost:8765','http://localhost:8765?target=x'])
def test_backend_is_server_owned_loopback_origin(url):
    with pytest.raises(ValueError):RecognitionGateway(url)


def test_sentence_adapter_preserves_text_without_rewriting_gloss(tmp_path):
    client,_=app_for(tmp_path,task='text')
    with client:
        response=client.post('/v1/communication/recognitions?sign_language=csl&request_id=text1&filename=c.mp4',content=b'clip')
    assert response.status_code==200
    assert response.json()['text']=='你好'
    assert response.json()['output_kind']=='text'


@pytest.mark.parametrize('status,code',[(429,'request_capacity'),(409,'runtime_busy')])
def test_runtime_capacity_is_retryable_not_a_model_identity_error(tmp_path,status,code):
    transport,_=backend()
    def handle(request):
        if request.url.path=='/v3/recognitions':
            return httpx.Response(status,json={'error':{'code':code}})
        return transport.handle_request(request)
    gateway=RecognitionGateway('http://127.0.0.1:8765',transport=httpx.MockTransport(handle),temporary_root=tmp_path)
    app=FastAPI();mount_communication_routes(app,gateway)
    with TestClient(app) as client:
        response=client.post('/v1/communication/recognitions?sign_language=csl&request_id=busy1&filename=c.mp4',content=b'clip')
    assert response.status_code==429
    assert not list(tmp_path.iterdir())


def test_disconnected_upload_cancels_inference_before_removing_temporary_file(tmp_path):
    async def scenario():
        started=asyncio.Event();observed={};messages=[]
        class Gateway:
            extensions={'.mp4'}
            max_upload_bytes=1024
            temporary_root=tmp_path
            lock=asyncio.Lock()
            async def recognize(self,source,*args):
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    observed['cleanup_before_file_removal']=source.is_file()
                    observed['cancelled']=True
        gateway=Gateway();app=FastAPI();mount_communication_routes(app,gateway)
        sent=False
        async def receive():
            nonlocal sent
            if not sent:
                sent=True
                return {'type':'http.request','body':b'clip','more_body':False}
            await started.wait()
            return {'type':'http.disconnect'}
        async def send(message):messages.append(message)
        scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
               'scheme':'http','path':'/v1/communication/recognitions','root_path':'',
               'query_string':b'sign_language=csl&request_id=cancel1&filename=c.mp4',
               'headers':[],'client':('127.0.0.1',1234),'server':('127.0.0.1',8879)}
        await asyncio.wait_for(app(scope,receive,send),2)
        assert messages[0]['status']==499
        assert observed=={'cleanup_before_file_removal':True,'cancelled':True}
        assert not gateway.lock.locked()
        assert not list(tmp_path.iterdir())
    asyncio.run(scenario())


def test_inferencing_model_stays_available_after_reloading_product(tmp_path):
    transport,_=backend()
    def handle(request):
        response=transport.handle_request(request)
        if request.url.path=='/v3/models':
            data=response.json();data['status']['csl']['state']='inferencing'
            return httpx.Response(200,json=data)
        return response
    gateway=RecognitionGateway('http://127.0.0.1:8765',transport=httpx.MockTransport(handle),temporary_root=tmp_path)
    app=FastAPI();mount_communication_routes(app,gateway)
    with TestClient(app) as client:
        data=client.get('/v1/communication/recognition').json()
    assert data['ready'] and data['languages'][0]['ready']
    assert data['languages'][0]['busy']
