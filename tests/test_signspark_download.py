"""Exercise actual curl transfers against bounded, local HTTP range fixtures."""
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts.animation.download_signspark_models import ModelSpec, download_stream


@pytest.fixture
def range_server():
    data = bytes(range(251)) * 31
    requests = []
    mode = {'value': 'correct'}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            start, end = map(int, self.headers['Range'][6:].split('-'))
            requests.append((start, end, parse_qs(urlsplit(self.path).query)))
            body = data[start:end + 1]
            behavior = mode['value']
            if behavior == 'slow':
                time.sleep(2)
            if behavior == 'ignore':
                body = data
            if behavior == 'short':
                body = body[:-1]
            self.send_response(200 if behavior == 'ignore' else 206)
            claimed_start = start + 1 if behavior == 'wrong_range' or (behavior == 'fail_second' and start == 1024) else start
            self.send_header('Content-Range', f'bytes {claimed_start}-{end}/{len(data)}')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield data, f'http://127.0.0.1:{server.server_port}/model', requests, mode
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_real_curl_reassembles_multiple_ranges_and_checks_digest(tmp_path, range_server):
    data, url, requests, _ = range_server
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    result = download_stream(spec, tmp_path, part_size=1024, workers=4)
    assert result.read_bytes() == data
    assert len(requests) == 8
    assert len({q['part'][0] for _, _, q in requests}) == 8
    assert not list((tmp_path / '.parts').rglob('*.chunk'))


@pytest.mark.parametrize('behavior', ['ignore', 'wrong_range', 'short'])
def test_invalid_range_response_never_publishes_model(tmp_path, range_server, behavior):
    data, url, _, mode = range_server
    mode['value'] = behavior
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    with pytest.raises(ValueError):
        download_stream(spec, tmp_path, part_size=1024, workers=2, attempts=1)
    assert not (tmp_path / 'body' / 'ema_0.9999_200000.pt').exists()
    assert list((tmp_path / '.parts').rglob('*.error.json'))


def test_failed_download_resumes_verified_parts_without_repeat_transfer(tmp_path, range_server):
    data, url, requests, mode = range_server
    mode['value'] = 'fail_second'
    spec = ModelSpec('hand', len(data), hashlib.sha256(data).hexdigest(), url)
    with pytest.raises(ValueError):
        download_stream(spec, tmp_path, part_size=1024, workers=1, attempts=1)
    assert list((tmp_path / '.parts').rglob('000000.chunk'))
    assert not (tmp_path / 'hand' / 'ema_0.9999_200000.pt').exists()
    requests.clear()
    mode['value'] = 'correct'
    result = download_stream(spec, tmp_path, part_size=1024, workers=2)
    assert result.read_bytes() == data
    assert all(start != 0 for start, _, _ in requests)


def test_corrupted_cached_part_is_replaced_even_with_same_length(tmp_path, range_server):
    data, url, requests, _ = range_server
    spec = ModelSpec('face', len(data), hashlib.sha256(data).hexdigest(), url)
    result = download_stream(spec, tmp_path, part_size=1024, keep_parts=True)
    result.unlink()
    chunk = next((tmp_path / '.parts').rglob('000002.chunk'))
    chunk.write_bytes(b'x' * chunk.stat().st_size)
    requests.clear()
    result = download_stream(spec, tmp_path, part_size=1024)
    assert result.read_bytes() == data
    assert [(start, end) for start, end, _ in requests] == [(2048, 3071)]


def test_wrong_final_digest_keeps_evidence_and_never_publishes(tmp_path, range_server):
    data, url, _, _ = range_server
    spec = ModelSpec('body', len(data), '0' * 64, url)
    with pytest.raises(ValueError, match='SHA256'):
        download_stream(spec, tmp_path, part_size=1024, workers=4)
    assert not (tmp_path / 'body' / 'ema_0.9999_200000.pt').exists()
    assert not (tmp_path / 'body' / '.signspark-assembling').exists()
    evidence = next((tmp_path / '.parts').rglob('assembly.error.json'))
    assert json.loads(evidence.read_text())['sha256'] == hashlib.sha256(data).hexdigest()
    assert len(list((tmp_path / '.parts').rglob('*.chunk'))) == 8


def test_transfer_timeout_is_bounded(tmp_path, range_server):
    data, url, _, mode = range_server
    mode['value'] = 'slow'
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    started = time.monotonic()
    with pytest.raises(ValueError, match='curl exit'):
        download_stream(spec, tmp_path, part_size=1024, workers=1, timeout=1, attempts=1)
    assert time.monotonic() - started < 5


def test_other_writer_cannot_mutate_same_output(tmp_path, range_server):
    data, url, requests, mode = range_server
    mode['value'] = 'slow'
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(download_stream, spec, tmp_path, part_size=1024,
                              timeout=1, attempts=1, workers=1)
        deadline = time.monotonic() + 3
        while not requests and time.monotonic() < deadline:
            time.sleep(.01)
        assert requests
        with pytest.raises(ValueError, match='writer'):
            download_stream(spec, tmp_path, part_size=1024)
        with pytest.raises(ValueError):
            running.result(timeout=5)


@pytest.mark.parametrize('kwargs', [{'part_size': 16 * 1024**2 + 1}, {'workers': 17},
                                   {'attempts': 0}, {'timeout': 601}])
def test_unbounded_settings_fail_before_request(tmp_path, range_server, kwargs):
    data, url, requests, _ = range_server
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    with pytest.raises(ValueError):
        download_stream(spec, tmp_path, **kwargs)
    assert not requests


def test_existing_unrelated_part_is_never_overwritten_or_cleaned(tmp_path, range_server):
    data, url, _, _ = range_server
    unrelated = tmp_path / 'body.part'
    unrelated.write_bytes(b'legacy curl data')
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    download_stream(spec, tmp_path, part_size=1024)
    assert unrelated.read_bytes() == b'legacy curl data'


def test_verified_final_resume_cleans_owned_leftover_parts_without_network(tmp_path, range_server):
    data, url, requests, _ = range_server
    spec = ModelSpec('body', len(data), hashlib.sha256(data).hexdigest(), url)
    target = download_stream(spec, tmp_path, part_size=1024, keep_parts=True)
    parts = next((tmp_path / '.parts').iterdir())
    unrelated = parts / 'user-notes.txt'
    unrelated.write_text('keep this')
    assert list(parts.glob('*.chunk'))
    requests.clear()
    assert download_stream(spec, tmp_path, part_size=1024) == target
    assert not requests
    assert not list(parts.glob('*.chunk'))
    assert unrelated.read_text() == 'keep this'
