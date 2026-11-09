from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


class Service:
    def __init__(self):
        self.error = None
        self.closed = 0
    def status(self):
        return {'enabled': True, 'planner': {'state': 'ready'}}
    def compose(self, text):
        if self.error:
            raise self.error
        return {'status': 'blocked', 'input_text': text, 'generation_text': text,
                'glosses': ['未知'], 'missing_glosses': ['未知']}
    def close(self):
        self.closed += 1


def client_for(service):
    from slbridge.animation.word_composition_routes import mount_word_composition_routes
    app = FastAPI()
    mount_word_composition_routes(app, service)
    return TestClient(app)


def test_actual_transport_preserves_missing_words_and_closes_planner():
    service = Service()
    with client_for(service) as client:
        assert client.get('/v1/animation/composition').json()['planner']['state'] == 'ready'
        response = client.post('/v1/animation/composition', json={'text': '我不去医院。'})
        assert response.status_code == 200
        assert response.json()['input_text'] == '我不去医院。'
        assert response.json()['missing_glosses'] == ['未知']
        assert 'motion' not in response.json()
    assert service.closed == 1


@pytest.mark.parametrize('body', [{}, {'text': ''}, {'text': 'x' * 201},
                                 {'text': 42}, {'text': '你好', 'model': 'C:/private'}])
def test_invalid_client_values_never_enter_service(body):
    with client_for(Service()) as client:
        assert client.post('/v1/animation/composition', json=body).status_code == 422


def test_missing_config_is_unavailable_not_slow_fallback():
    with client_for(None) as client:
        assert client.get('/v1/animation/composition').json()['enabled'] is False
        assert client.post('/v1/animation/composition', json={'text': '你好'}).status_code == 503


@pytest.mark.parametrize('error_name,status', [('CompositionBusyError', 429),
                                              ('CompositionUnavailableError', 503),
                                              ('WordLibraryError', 503), ('ValueError', 422)])
def test_public_error_status_and_paths_are_sanitized(error_name, status):
    from slbridge.animation import word_composition as core
    service = Service()
    error = ValueError if error_name == 'ValueError' else getattr(core, error_name)
    service.error = error('private C:/models/weights')
    with client_for(service) as client:
        response = client.post('/v1/animation/composition', json={'text': '你好'})
        assert response.status_code == status
        assert 'C:/' not in response.text and 'weights' not in response.text
