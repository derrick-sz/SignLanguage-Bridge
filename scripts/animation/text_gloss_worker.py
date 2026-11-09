"""Persistent local MBart text-to-gloss worker; stdout is bounded JSONL only."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from slbridge.animation.text_gloss_planner import (
    MAX_NEW_TOKENS, MAX_PROTOCOL_BYTES, NUM_BEAMS, PROTOCOL, normalize_text,
    planner_identity, validate_glosses, verify_assets,
)


def build_prompt(text):
    # Exact prompt used by the author-released CSL-Daily T2G model.
    return f'Translate Chinese Sentence to Chinese Sign Language Gloss: {text}\nGloss:'


def encode_prompt(tokenizer, text, max_positions):
    # The released trimmed vocabulary intentionally encodes the fixed English
    # training prompt as UNKs. Check the actual user text separately; preserve
    # the author's prompt tokens exactly rather than rejecting or rewriting it.
    raw = tokenizer(text, add_special_tokens=False, truncation=False)['input_ids']
    if tokenizer.unk_token_id in raw:
        raise ValueError('unknown_token')
    encoded = tokenizer([build_prompt(text)], return_tensors='pt', padding=True, truncation=False)
    if encoded['input_ids'].shape[1] > max_positions:
        raise ValueError('input_limit')
    return encoded


def decode_glosses(sequence, tokenizer):
    if not isinstance(sequence, list) or len(sequence) < 2:
        raise ValueError('empty_output')
    generated = sequence[1:]  # decoder-start language token is not generated text
    # Forced EOS at max_new_tokens is still truncation: reject the limit itself.
    if len(generated) >= MAX_NEW_TOKENS or tokenizer.eos_token_id not in generated:
        raise ValueError('output_limit')
    end = generated.index(tokenizer.eos_token_id)
    tokens = generated[:end]
    if tokenizer.unk_token_id in tokens:
        raise ValueError('unknown_token')
    glosses = tokenizer.decode(tokens, skip_special_tokens=True).strip().split()
    try:
        return validate_glosses(glosses)
    except ValueError:
        raise ValueError('output_limit' if glosses else 'empty_output') from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda', *[f'cuda:{i}' for i in range(16)]], default='cpu')
    parser.add_argument('--identity-json', required=True)
    args = parser.parse_args(argv)
    sys.stdin.reconfigure(encoding='utf-8', errors='strict')
    sys.stdout.reconfigure(encoding='utf-8', errors='strict')
    protocol_out = sys.stdout

    def emit(value):
        print(json.dumps(value, ensure_ascii=False, separators=(',', ':')), file=protocol_out, flush=True)

    identity = planner_identity(args.device)
    try:
        load_start = time.perf_counter()
        if json.loads(args.identity_json) != identity:
            raise ValueError('Worker model identity differs')
        verify_assets(args.model_dir)
        with redirect_stdout(sys.stderr):
            import torch
            from transformers import MBartForConditionalGeneration, MBartTokenizer
            torch.set_num_threads(4)
            tokenizer = MBartTokenizer.from_pretrained(str(args.model_dir), src_lang='zh_CN', tgt_lang='zh_CN',
                                                       local_files_only=True)
            model = MBartForConditionalGeneration.from_pretrained(str(args.model_dir), local_files_only=True,
                    use_safetensors=True, low_cpu_mem_usage=True, torch_dtype=torch.float32)
            model.eval().requires_grad_(False).to(args.device)
            verify_assets(args.model_dir)
            start_id = tokenizer.convert_tokens_to_ids('zh_CN')
            if start_id == tokenizer.unk_token_id or model.config.vocab_size != len(tokenizer):
                raise ValueError('Tokenizer and model differ')
        load_ms = (time.perf_counter() - load_start) * 1000

        def infer(text):
            with redirect_stdout(sys.stderr), torch.inference_mode():
                encoded = encode_prompt(tokenizer, text,
                        min(tokenizer.model_max_length, model.config.max_position_embeddings)).to(args.device)
                output = model.generate(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask'],
                        decoder_start_token_id=start_id, num_beams=NUM_BEAMS, do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS)
                return decode_glosses(output[0].tolist(), tokenizer)

        warmup_start = time.perf_counter()
        infer('你好。')  # same model path, never returned as a user's result
        warmup_ms = (time.perf_counter() - warmup_start) * 1000
        emit({'type': 'ready', 'protocol': PROTOCOL, 'model_identity': identity,
              'startup_metrics': {'load_ms': load_ms, 'warmup_ms': warmup_ms}})
    except Exception:
        traceback.print_exc(file=sys.stderr)
        emit({'type': 'error', 'error': 'model_unavailable'})
        return 1

    while True:
        line = sys.stdin.buffer.readline(MAX_PROTOCOL_BYTES + 1)
        if not line:
            return 0
        request_id = None
        try:
            if len(line) > MAX_PROTOCOL_BYTES or not line.endswith(b'\n'):
                raise ValueError('invalid_input')
            request = json.loads(line.decode('utf-8'))
            if not isinstance(request, dict) or set(request) != {'id', 'text'}:
                raise ValueError('invalid_input')
            request_id = request['id']
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 64:
                raise ValueError('invalid_input')
            try:
                text = normalize_text(request['text'])
            except ValueError:
                raise ValueError('invalid_input') from None
            start = time.perf_counter()
            glosses = infer(text)
            elapsed = (time.perf_counter() - start) * 1000
            emit({'type': 'result', 'id': request_id, 'glosses': glosses,
                  'model_identity': identity, 'elapsed_ms': elapsed})
        except ValueError as error:
            code = str(error)
            if code not in {'invalid_input', 'input_limit', 'output_limit', 'empty_output', 'unknown_token'}:
                code = 'invalid_input'
            emit({'type': 'error', 'id': request_id, 'error': code})
        except Exception:
            traceback.print_exc(file=sys.stderr)
            emit({'type': 'error', 'id': request_id, 'error': 'inference_failed'})
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
