"""Exercise the runtime's real nn.Module boundary without loading large weights."""
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.uni_sign import TorchUniSignRuntime
from slbridge.uni_sign_preprocess import PoseWindow

torch = pytest.importorskip('torch')


class MT5(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.training_calls = self.encoder_calls = 0

    def forward(self, *, inputs_embeds, attention_mask, labels, return_dict):
        self.training_calls += 1
        self.encoder_calls += 1
        return {'logits': inputs_embeds, 'loss': inputs_embeds.sum()}

    def generate(self, *, inputs_embeds, attention_mask, num_beams, max_new_tokens):
        self.encoder_calls += 1
        self.generated = dict(inputs_embeds=inputs_embeds.clone(), attention_mask=attention_mask.clone(),
                              num_beams=num_beams, max_new_tokens=max_new_tokens)
        return torch.tensor([[int(inputs_embeds.sum()), int(attention_mask.sum())]])


class PinnedUpstream(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mt5_model = MT5()
        self.fail_before_mt5 = self.skip_mt5 = self.fail_generate = False

    def forward(self, source, target):
        self.source = source
        if self.fail_before_mt5:
            raise ValueError('visual extraction failed')
        # Model-specific visual/prefix construction belongs to upstream.
        visual = source['body'].mean(dim=2)
        embeds = torch.cat([torch.ones((1, 1, 3)), visual], dim=1)
        mask = torch.cat([torch.ones((1, 1), dtype=torch.long), source['attention_mask']], dim=1)
        if not self.skip_mt5:
            loss = self.mt5_model(inputs_embeds=embeds, attention_mask=mask,
                                  labels=torch.tensor([[1]]), return_dict=True)['loss']
        else:
            loss = None
        return dict(inputs_embeds=embeds, attention_mask=mask, loss=loss)

    def generate(self, stack_out, *, max_new_tokens, num_beams):
        if self.fail_generate:
            raise ValueError('generation failed')
        assert set(stack_out) <= {'inputs_embeds', 'attention_mask', 'loss'}
        return self.mt5_model.generate(inputs_embeds=stack_out['inputs_embeds'],
                                       attention_mask=stack_out['attention_mask'],
                                       max_new_tokens=max_new_tokens, num_beams=num_beams)


def runtime():
    item = TorchUniSignRuntime(None, device='cpu', dtype='float32')
    item._torch = torch
    item._torch_dtype = torch.float32
    item._model = PinnedUpstream()
    item._tokenizer = SimpleNamespace(batch_decode=lambda tokens, **_: [str(tokens.tolist())])
    return item


def pose():
    return PoseWindow(**{k: np.ones((4, n, 3), np.float32) for k, n in
                         [('body', 9), ('left', 21), ('right', 21), ('face_all', 18)]},
                      sampled_indices=tuple(range(4)), quality={})


def test_inference_does_not_execute_unused_mt5_training_forward():
    item = runtime()
    text, timings = item.infer(pose(), num_beams=4, max_new_tokens=20)
    model = item._model.mt5_model
    assert model.training_calls == 0
    assert model.encoder_calls == 1
    assert text == '[[15, 5]]'
    assert model.generated['num_beams'] == 4 and model.generated['max_new_tokens'] == 20
    assert model.generated['inputs_embeds'].shape == (1, 5, 3)
    assert torch.equal(model.generated['attention_mask'], torch.ones((1, 5), dtype=torch.long))
    assert not model._forward_pre_hooks
    assert timings['model_total_ms'] >= timings['model_forward_ms'] >= 0


@pytest.mark.parametrize('flag,error', [('fail_before_mt5', 'visual extraction failed'),
                                       ('fail_generate', 'generation failed')])
def test_temporary_hook_is_removed_after_failure_and_retry(flag, error):
    item = runtime()
    setattr(item._model, flag, True)
    with pytest.raises(ValueError, match=error):
        item.infer(pose(), num_beams=4, max_new_tokens=20)
    assert not item._model.mt5_model._forward_pre_hooks
    setattr(item._model, flag, False)
    item.infer(pose(), num_beams=4, max_new_tokens=20)
    assert item._model.mt5_model.training_calls == 0
    assert not item._model.mt5_model._forward_pre_hooks


def test_changed_upstream_boundary_fails_instead_of_silent_fallback():
    item = runtime()
    item._model.skip_mt5 = True
    with pytest.raises(RuntimeError, match='generation inputs'):
        item.infer(pose(), num_beams=4, max_new_tokens=20)
    assert not item._model.mt5_model._forward_pre_hooks


def test_existing_hooks_are_preserved_and_generated_inputs_repeat_exactly():
    item = runtime()
    sentinel = item._model.mt5_model.register_forward_pre_hook(lambda *_: None)
    try:
        first = item.infer(pose(), num_beams=4, max_new_tokens=20)[0]
        second = item.infer(pose(), num_beams=4, max_new_tokens=20)[0]
        assert first == second
        assert len(item._model.mt5_model._forward_pre_hooks) == 1
        assert item._model.mt5_model.training_calls == 0
    finally:
        sentinel.remove()
