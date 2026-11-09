"""Lightweight product adapter for the existing, separately hosted v3 runtime."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from urllib.parse import urlsplit

import httpx


class RecognitionUnavailable(RuntimeError):
    pass


class RecognitionContractChanged(RuntimeError):
    pass


class RecognitionBusy(RuntimeError):
    pass


class VideoInputError(ValueError):
    pass


class RecognitionGateway:
    extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

    def __init__(self, base_url, *, token=None, transport=None, temporary_root=None,
                 max_upload_bytes=128*1024*1024, ffmpeg=None):
        parsed = urlsplit(base_url)
        if (parsed.scheme != 'http' or parsed.hostname not in {'localhost', '127.0.0.1', '::1'}
                or parsed.username or parsed.password or parsed.path not in {'', '/'}
                or parsed.query or parsed.fragment or not parsed.port):
            raise ValueError('Recognition backend must be a server-owned loopback HTTP origin')
        if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
            raise ValueError('Invalid upload limit')
        self.base_url = base_url.rstrip('/')
        self.token, self.transport = token, transport
        self.temporary_root = temporary_root
        self.max_upload_bytes = max_upload_bytes
        self.ffmpeg = str(ffmpeg) if ffmpeg else shutil.which('ffmpeg')
        self.lock = asyncio.Lock()
        catalog = Path(__file__).with_name('resources')/'libras_caption_catalog.json'
        self.captions = {entry['label_id']: entry for entry in json.loads(catalog.read_text(encoding='utf-8'))['labels']}

    def client(self):
        return httpx.AsyncClient(base_url=self.base_url, transport=self.transport,
                                 headers={'Authorization': 'Bearer '+self.token} if self.token else {},
                                 timeout=httpx.Timeout(120, connect=3), trust_env=False)

    @staticmethod
    async def payload(client, method, path, **kwargs):
        try:
            response = await client.request(method, path, **kwargs)
            if response.status_code in {409, 429}:
                try:
                    error = response.json().get('error', {})
                except (ValueError, AttributeError):
                    error = {}
                if response.status_code == 429 or error.get('code') in {'runtime_busy', 'request_capacity'}:
                    raise RecognitionBusy('上一段还在识别，请稍后再试。')
            if response.status_code == 409:
                raise RecognitionContractChanged('识别服务已更新，请重新识别这段视频。')
            if response.status_code in {400, 413, 422}:
                raise VideoInputError('这段视频无法识别。请使用15秒以内、双手清楚可见的视频。')
            if not response.is_success:
                raise RecognitionUnavailable('识别服务暂不可用，请稍后重试。')
            return response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            raise RecognitionUnavailable('无法连接识别服务。文字回复仍可使用。') from exc

    async def status(self):
        async with self.client() as client:
            data = await self.payload(client, 'GET', '/v3/models')
        caps = data.get('capabilities', {})
        if 'inputs' in caps:
            caps = {caps['sign_language']: caps}
        languages = []
        for language, cap in caps.items():
            if language not in {'csl', 'libras'} or not any('file' in x.get('input_kinds', []) for x in cap.get('inputs', [])):
                continue
            state = data.get('status', {})
            state = state.get(language, state)
            languages.append({'id': language, 'label': '中国手语' if language == 'csl' else '巴西手语 Libras',
                              'ready': state.get('state') in {'ready', 'inferencing'},
                              'busy': state.get('state') == 'inferencing', 'task_type': cap.get('task_type'),
                              'scope': '中文手语词识别，请核对后使用' if cap.get('task_type') == 'gloss_sequence'
                              else '现有12词试点，中葡释义需核对' if language == 'libras' else '中文句子识别，请核对后使用'})
        return {'enabled': True, 'ready': any(x['ready'] for x in languages), 'languages': languages,
                'max_upload_bytes': self.max_upload_bytes, 'max_recording_seconds': 12,
                'recording_available': self.ffmpeg is not None, 'busy': self.lock.locked()}

    async def convert_recording(self, source):
        if source.suffix.lower() != '.webm':
            return source
        if not self.ffmpeg:
            raise RecognitionUnavailable('浏览器录制暂不可识别，请上传MP4视频。')
        output = source.with_suffix('.mp4')
        # Bound decoding and reject the over-limit output; never publish a
        # truncated recording as if the complete utterance had been recognized.
        args = [self.ffmpeg, '-v', 'fatal', '-xerror', '-y', '-i', str(source), '-map', '0:v:0',
                '-an', '-t', '15.1', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                '-pix_fmt', 'yuv420p', '-threads', '2', str(output)]
        await self.command(args)
        probe = str(Path(self.ffmpeg).with_name('ffprobe.exe' if Path(self.ffmpeg).suffix == '.exe' else 'ffprobe'))
        result = await self.command([probe, '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(output)])
        try:
            duration = float(json.loads(result)['format']['duration'])
        except (ValueError, KeyError, TypeError) as exc:
            raise VideoInputError('无法读取录制视频，请重新录制。') from exc
        if not 0 < duration <= 15:
            raise VideoInputError('请将这段视频缩短到15秒以内。')
        return output

    @staticmethod
    async def command(args):
        try:
            process = await asyncio.create_subprocess_exec(*args, stdout=asyncio.subprocess.PIPE,
                                                           stderr=asyncio.subprocess.PIPE)
        except OSError as exc:
            raise RecognitionUnavailable('录制视频处理暂不可用，请上传MP4视频。') from exc
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), 30)
        except BaseException as exc:
            if process.returncode is None:
                process.kill()
                await process.wait()
            if isinstance(exc, TimeoutError):
                raise VideoInputError('处理视频用时过长，请换一段更短的视频。') from exc
            raise
        if process.returncode:
            raise VideoInputError('录制视频无法读取，请重新录制或上传MP4。')
        return stdout

    async def recognize(self, source, language, request_id, filename):
        async with self.client() as client:
            negotiation = await self.payload(client, 'POST', '/v3/models/negotiate',
                                             json={'sign_language': language, 'mode': 'clip'})
            cap, contract = negotiation.get('capabilities', {}), negotiation.get('input_contract', {})
            generation = negotiation.get('model_generation')
            if cap.get('sign_language') != language or not contract.get('representation_id') or type(generation) is not int:
                raise RecognitionContractChanged('识别能力已变化，请刷新后重试。')
            fields = {'sign_language': language, 'mode': 'clip', 'request_id': request_id,
                      'representation_id': contract['representation_id'],
                      'preprocessing_version': contract['preprocessing_version'],
                      'task_type': cap['task_type'], 'expected_generation': str(generation)}
            original_suffix = source.suffix.lower()
            source = await self.convert_recording(source)
            with source.open('rb') as video:
                data = await self.payload(client, 'POST', '/v3/recognitions', data=fields,
                                          files={'file': (Path(filename).stem+source.suffix, video, 'application/octet-stream')})
        result = data.get('result', {})
        if (data.get('request_id') != request_id or data.get('model_generation') != generation
                or data.get('preprocessing_version') != contract['preprocessing_version']
                or result.get('sign_language') != language or result.get('task_type') != cap['task_type']
                or data.get('model_id') != cap['model_id']):
            raise RecognitionContractChanged('识别结果与本次输入不一致，请重新识别。')
        text, translations, review = '', {}, 'unverified'
        prediction = result.get('prediction')
        if result.get('status') == 'recognized' and isinstance(prediction, dict):
            if cap['task_type'] == 'text':
                text = prediction.get('text', '')
            elif cap['task_type'] == 'gloss_sequence':
                text = ' '.join(token['gloss'] for token in prediction.get('tokens', []))
            elif cap['task_type'] == 'classification':
                translations = dict(prediction.get('display_text', {}))
                entry = self.captions.get(prediction.get('label_id')) if language == 'libras' else None
                if entry:
                    translations.update(entry['display_text'])
                    review = entry['review_status']
                text = translations.get('zh-CN') or translations.get('pt-BR') or ''
        if not isinstance(text, str):
            raise RecognitionContractChanged('识别返回格式已变化，请重新识别。')
        return {'request_id': request_id, 'recognized': bool(text.strip()), 'text': text,
                'translations': translations, 'output_kind': cap['task_type'], 'sign_language': language,
                'review_status': review, 'media': {'source_format': original_suffix, 'inference_format': source.suffix,
                                                  'transcoded': original_suffix != source.suffix},
                'provenance': {key: data.get(key) for key in
                    ['model_id', 'model_version', 'model_generation', 'preprocessing_version']}}
