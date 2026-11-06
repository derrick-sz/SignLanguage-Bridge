"""Bounded native-curl Range downloads of pinned SignSparK EMA checkpoints.

CLI URLs and LFS identities are fixed; ModelSpec also permits small local HTTP
fixtures for transport verification. No Torch, Hugging Face SDK, or shell calls.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import uuid

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from slbridge.animation.signspark_generation import MODEL_FILES, MODEL_REVISION

PART_BYTES = 16 * 1024 * 1024
MAX_FILE_BYTES = 6 * 1024**3
MAX_TOTAL_BYTES = 17 * 1024**3
FILENAME = 'ema_0.9999_200000.pt'


@dataclass(frozen=True)
class ModelSpec:
    stream: str
    size: int
    sha256: str
    url: str

    def __post_init__(self):
        parsed = urlsplit(self.url)
        if (self.stream not in MODEL_FILES or type(self.size) is not int
                or not 1 <= self.size <= MAX_FILE_BYTES
                or not re.fullmatch('[0-9a-f]{64}', self.sha256)
                or parsed.scheme not in ('http', 'https') or not parsed.netloc
                or parsed.username or parsed.password or parsed.fragment):
            raise ValueError('Invalid bounded model identity or URL')


def official_spec(stream):
    size, digest = MODEL_FILES[stream]
    return ModelSpec(stream, size, digest,
                     f'https://huggingface.co/LionelLow/SignSparK/resolve/'
                     f'{MODEL_REVISION}/{stream}/{FILENAME}')


def _hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        while raw := source.read(4 * 1024 * 1024):
            digest.update(raw)
    return digest.hexdigest()


def _safe(path, root):
    """Never follow a symlink/junction out of the explicitly selected output."""
    path, root = Path(path), Path(root)
    path.absolute().relative_to(root.absolute())
    current = path
    while current != root.parent:
        if current.is_symlink() or (hasattr(current, 'is_junction') and current.is_junction()):
            raise ValueError(f'Unsafe linked download path: {current}')
        if current == root:
            break
        current = current.parent
    path.resolve().relative_to(root.resolve())
    return path


@contextmanager
def _writer_lock(root):
    lock = _safe(root / '.signspark-download.lock', root)
    with lock.open('a+b') as handle:
        handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError('Another SignSparK download writer is active') from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _json(path, value, root):
    temporary = _safe(path.with_name(path.name + '.tmp'), root)
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, _safe(path, root))


def _get_part(spec, root, parts, index, start, end, curl, timeout, attempts, run):
    chunk = _safe(parts / f'{index:06d}.chunk', root)
    receipt = _safe(parts / f'{index:06d}.json', root)
    identity = {'stream': spec.stream, 'model_sha256': spec.sha256,
                'model_bytes': spec.size, 'start': start, 'end': end}
    expected = end - start + 1
    if chunk.is_file() and receipt.is_file():
        try:
            record = json.loads(receipt.read_text(encoding='utf-8'))
            if (all(record.get(key) == value for key, value in identity.items())
                    and chunk.stat().st_size == expected
                    and _hash(chunk) == record.get('part_sha256')):
                return chunk
        except (ValueError, OSError):
            pass
    temporary = _safe(parts / f'{index:06d}.incoming', root)
    headers = _safe(parts / f'{index:06d}.headers', root)
    error = _safe(parts / f'{index:06d}.error.json', root)
    last_error = None
    for attempt in range(attempts):
        for path in (temporary, headers):
            path.unlink(missing_ok=True)
        parsed = urlsplit(spec.url)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        query += [('download', 'true'), ('part', f'{start}-{end}'),
                  ('attempt', str(attempt)), ('transfer', run)]
        url = urlunsplit(parsed._replace(query=urlencode(query)))
        args = [curl, '--location', '--fail', '--silent', '--show-error',
                '--connect-timeout', str(min(20, timeout)), '--max-time', str(timeout),
                '--max-filesize', str(expected), '--retry', '0',
                '--header', 'Accept-Encoding: identity', '--range', f'{start}-{end}',
                '--dump-header', str(headers), '--output', str(temporary), url]
        try:
            completed = subprocess.run(args, capture_output=True, timeout=timeout + 10,
                                       shell=False)
            if completed.returncode:
                raise ValueError(f'curl exit {completed.returncode}: '
                                 + completed.stderr.decode(errors='replace')[-1500:])
            raw_headers = headers.read_bytes()
            if len(raw_headers) > 256 * 1024:
                raise ValueError('Oversized HTTP headers')
            blocks = list(re.finditer(rb'(?im)^HTTP/\S+\s+(\d{3})[^\r\n]*\r?\n', raw_headers))
            if not blocks or blocks[-1].group(1) != b'206':
                raise ValueError('Server must honor Range with HTTP 206')
            final = raw_headers[blocks[-1].end():]
            ranges = re.findall(rb'(?im)^Content-Range:\s*([^\r\n]+)', final)
            required = f'bytes {start}-{end}/{spec.size}'.encode()
            if len(ranges) != 1 or ranges[0].strip() != required:
                raise ValueError('Content-Range mismatch')
            lengths = re.findall(rb'(?im)^Content-Length:\s*([^\r\n]+)', final)
            if lengths and (len(lengths) != 1 or lengths[0].strip() != str(expected).encode()):
                raise ValueError('Content-Length mismatch')
            if temporary.stat().st_size != expected:
                raise ValueError('Range payload length mismatch')
            digest = _hash(temporary)
            os.replace(temporary, chunk)
            _json(receipt, identity | {'part_sha256': digest}, root)
            error.unlink(missing_ok=True)
            return chunk
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            last_error = str(exc)
            _json(error, identity | {'attempt': attempt + 1, 'error': last_error}, root)
    raise ValueError(f'{spec.stream} range {start}-{end} failed: {last_error}')


def _cleanup_parts(parts, ranges, root):
    if not parts.exists():
        return
    for index, _, _ in ranges:
        for extension in ('chunk', 'json', 'incoming', 'headers', 'error.json', 'json.tmp', 'error.json.tmp'):
            _safe(parts / f'{index:06d}.{extension}', root).unlink(missing_ok=True)
    _safe(parts / 'assembly.error.json', root).unlink(missing_ok=True)
    if not any(parts.iterdir()):
        parts.rmdir()


def download_stream(spec, output, *, part_size=PART_BYTES, workers=4,
                    timeout=180, attempts=3, keep_parts=False, progress=None):
    """Return a final verified model; interrupted/failed parts remain resumable."""
    if (not isinstance(spec, ModelSpec) or type(part_size) is not int
            or not 1 <= part_size <= PART_BYTES or type(workers) is not int
            or not 1 <= workers <= 16 or type(timeout) is not int
            or not 1 <= timeout <= 600 or type(attempts) is not int or not 1 <= attempts <= 5):
        raise ValueError('Invalid bounded download settings')
    root = Path(output).absolute()
    _safe(root, root)
    root.mkdir(parents=True, exist_ok=True)
    curl = shutil.which('curl.exe') or shutil.which('curl')
    if not curl:
        raise ValueError('Native curl executable is required')
    if (spec.size + part_size - 1) // part_size > 4096:
        raise ValueError('Too many ranges for bounded download budget')
    destination = _safe(root / spec.stream, root)
    destination.mkdir(exist_ok=True)
    with _writer_lock(destination):
        ranges = [(i, start, min(start + part_size, spec.size) - 1)
                  for i, start in enumerate(range(0, spec.size, part_size))]
        parts = _safe(root / '.parts' / f'{spec.stream}-{spec.sha256}-p{part_size}', root)
        target = _safe(destination / FILENAME, root)
        if target.exists():
            if target.stat().st_size == spec.size and _hash(target) == spec.sha256:
                if not keep_parts:
                    _cleanup_parts(parts, ranges, root)
                return target
            raise ValueError('Existing final model has wrong size or SHA256; left unchanged')
        parts.mkdir(parents=True, exist_ok=True)
        run = uuid.uuid4().hex
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_get_part, spec, root, parts, i, start, end,
                                   curl, timeout, attempts, run) for i, start, end in ranges]
            try:
                for completed_parts, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if progress is not None:
                        progress({'state': 'parts', 'stream': spec.stream,
                                  'completed_parts': completed_parts, 'total_parts': len(ranges)})
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
        assembly = _safe(destination / '.signspark-assembling', root)
        digest = hashlib.sha256()
        size = 0
        with assembly.open('wb') as assembled:
            for index, _, _ in ranges:
                with _safe(parts / f'{index:06d}.chunk', root).open('rb') as source:
                    while raw := source.read(4 * 1024 * 1024):
                        size += len(raw)
                        if size > spec.size:
                            raise ValueError('Assembly exceeds model budget')
                        digest.update(raw)
                        assembled.write(raw)
            assembled.flush()
            os.fsync(assembled.fileno())
        if size != spec.size or digest.hexdigest() != spec.sha256:
            _json(parts / 'assembly.error.json', {'bytes': size, 'sha256': digest.hexdigest(),
                                                 'expected_sha256': spec.sha256}, root)
            assembly.unlink(missing_ok=True)
            raise ValueError('Final model SHA256 or size mismatch; model not published')
        os.replace(assembly, target)
        if not keep_parts:
            _cleanup_parts(parts, ranges, root)
        return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--streams', nargs='+', choices=sorted(MODEL_FILES),
                        default=['body', 'hand', 'face'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, help='Concurrent 16MiB ranges per stream')
    parser.add_argument('--timeout', type=int, default=180, help='Seconds per curl transfer')
    parser.add_argument('--attempts', type=int, default=3)
    parser.add_argument('--keep-parts', action='store_true')
    args = parser.parse_args(argv)
    streams = list(dict.fromkeys(args.streams))
    specs = [official_spec(stream) for stream in streams]
    if sum(spec.size for spec in specs) > MAX_TOTAL_BYTES:
        parser.error('Requested downloads exceed total byte budget')
    for spec in specs:  # Avoid assembling all three 5.6GB streams simultaneously.
        print(json.dumps({'state': 'starting', 'stream': spec.stream, 'bytes': spec.size}), flush=True)
        target = download_stream(spec, args.output, workers=args.workers, timeout=args.timeout,
                                 attempts=args.attempts, keep_parts=args.keep_parts,
                                 progress=lambda event: print(json.dumps(event), flush=True))
        print(json.dumps({'state': 'verified', 'stream': spec.stream,
                          'path': str(target), 'sha256': spec.sha256}), flush=True)


if __name__ == '__main__':
    main()
