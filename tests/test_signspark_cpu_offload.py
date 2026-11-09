"""Real CPU hooks and tiny FP32 modules exercise bounded text conditioning."""
import hashlib

import pytest


def toy_model(torch):
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.text_encoder = torch.nn.Linear(2, 2)
            self.projection = torch.nn.Linear(2, 1)
            self.register_buffer('position_ids', torch.arange(3), persistent=False)
            self.encode_calls = 0
            with torch.no_grad():
                self.text_encoder.weight.copy_(torch.tensor([[2., 0.], [0., 3.]]))
                self.text_encoder.bias.copy_(torch.tensor([5., 7.]))
                self.projection.weight.copy_(torch.tensor([[2., 3.]]))
                self.projection.bias.fill_(4.)

        def encode_text(self, texts):
            self.encode_calls += 1
            inputs = torch.tensor([{'one': [1., 2.], 'two': [3., 4.]}[text] for text in texts])
            return self.text_encoder(inputs)

        def forward(self, inputs, texts):
            return self.projection(inputs) + self.encode_text(texts).sum(1, keepdim=True) + self.position_ids[-1]

    return Toy()


def test_cpu_offload_caches_real_fp32_embeddings_once_and_keeps_buffers(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('accelerate')
    from slbridge.animation.signspark_generation import load_strict_checkpoint, prepare_cpu_offload
    model = toy_model(torch)
    path = tmp_path / 'toy.pt'
    torch.save(model.state_dict(), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    load_strict_checkpoint(model, path, expected_sha256=digest,
                           expected_bytes=path.stat().st_size, assign=True, readonly_mmap=True)
    parameter_pointers = {name: value.data_ptr() for name, value in model.named_parameters()}
    report = prepare_cpu_offload(model, ['one', 'two', 'one'], device='cpu')
    assert report['cached_texts'] == 2 and report['offload_buffers'] is False
    assert report['execution_device'] == 'cpu' and report['embedding_dtype'] == 'float32'
    assert model.encode_calls == 2
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    assert all(p.is_meta for p in model.parameters())
    assert model.position_ids.device.type == 'cpu' and model.position_ids.tolist() == [0, 1, 2]
    torch.testing.assert_close(model.encode_text(['one']), torch.tensor([[7., 13.]]), rtol=0, atol=0)
    torch.testing.assert_close(model.encode_text(['two', 'one', 'two']),
                               torch.tensor([[11., 19.], [7., 13.], [11., 19.]]), rtol=0, atol=0)
    assert model.encode_text(['one']).dtype == torch.float32
    assert not model.encode_text(['one']).requires_grad
    with torch.inference_mode():
        torch.testing.assert_close(model(torch.tensor([[1., 1.]]), ['one']),
                                   torch.tensor([[31.]]), rtol=0, atol=0)
    assert model.encode_calls == 2
    assert all(p.is_meta for p in model.parameters())  # real hooks evict after forward
    weights_map = model.projection._hf_hook.weights_map
    assert weights_map['weight'].data_ptr() == parameter_pointers['projection.weight']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('texts', [[], 'one', [''], [None]])
def test_cpu_offload_rejects_invalid_requested_texts_before_encoding(texts):
    torch = pytest.importorskip('torch')
    from slbridge.animation.signspark_generation import prepare_cpu_offload
    model = toy_model(torch)
    with pytest.raises(ValueError, match='text'):
        prepare_cpu_offload(model, texts, device='cpu')
    assert model.encode_calls == 0


def test_cpu_offload_rejects_uncached_or_malformed_prompt_without_encoding():
    torch = pytest.importorskip('torch')
    pytest.importorskip('accelerate')
    from slbridge.animation.signspark_generation import prepare_cpu_offload
    model = toy_model(torch)
    prepare_cpu_offload(model, ['one'], device='cpu')
    for values in [['two'], ['one', 'two'], 'one', [], [None]]:
        with pytest.raises(ValueError, match='cached|text'):
            model.encode_text(values)
    assert model.encode_calls == 1


@pytest.mark.parametrize('change', ['meta_parameter', 'meta_buffer', 'half'])
def test_cpu_offload_rejects_unmaterialized_or_reduced_precision_model(change):
    torch = pytest.importorskip('torch')
    pytest.importorskip('accelerate')
    from slbridge.animation.signspark_generation import prepare_cpu_offload
    model = toy_model(torch)
    if change == 'meta_parameter':
        model.projection.to('meta')
    elif change == 'meta_buffer':
        model.position_ids = torch.empty(3, device='meta', dtype=torch.int64)
    else:
        model.half()
    with pytest.raises(ValueError, match='CPU|FP32'):
        prepare_cpu_offload(model, ['one'], device='cpu')
    assert model.encode_calls == 0


@pytest.mark.parametrize('bad_embedding', ['half', 'nan', 'wrong_batch'])
def test_cpu_offload_rejects_invalid_original_embeddings(bad_embedding):
    torch = pytest.importorskip('torch')
    pytest.importorskip('accelerate')
    from slbridge.animation.signspark_generation import prepare_cpu_offload
    model = toy_model(torch)
    if bad_embedding == 'half':
        model.encode_text = lambda values: torch.ones(1, 2, dtype=torch.float16)
    elif bad_embedding == 'nan':
        model.encode_text = lambda values: torch.tensor([[float('nan'), 1.]])
    else:
        model.encode_text = lambda values: torch.ones(2, 2)
    with pytest.raises(ValueError, match='embedding'):
        prepare_cpu_offload(model, ['one'], device='cpu')
    assert not any(hasattr(module, '_hf_hook') for module in model.modules())
