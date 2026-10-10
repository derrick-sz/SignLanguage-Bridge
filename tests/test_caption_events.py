from types import SimpleNamespace

import pytest

from slbridge.caption_events import CaptionLedger
from slbridge.contracts import RecognitionResult, ClassificationPrediction


def reserve(ledger, n, **overrides):
    args = dict(segment_id=f"s{n}", sequence=n, request_id=f"r{n}", source_epoch=1,
                model_generation=2, start_time=float(n), end_time=n+.5, source_id="camera:0")
    args.update(overrides)
    return ledger.reserve(**args)


def response(ledger, n, **overrides):
    value = dict(session_id=ledger.session_id, segment_id=f"s{n}", request_id=f"r{n}",
                 source_epoch=1, model_generation=2, model_id="m", model_version="m-v1",
                 preprocessing_version="p-v1", runtime={"provider": "cpu"}, timing_ms={"total_ms": 1},
                 result=RecognitionResult("recognized", f"r{n}", "m", "libras",
                    ClassificationPrediction("same-word", .8, {"same-word": .8}, {"pt-BR": "Olá"}), {}))
    value.update(overrides)
    return SimpleNamespace(**value)


def ledger(**kw):
    value = CaptionLedger(**kw)
    value.bind(1, 2)
    return value


def test_order_is_action_order_and_equal_words_remain_separate_events():
    value = ledger()
    reserve(value, 1)
    reserve(value, 2)
    assert value.complete(response(value, 2))
    assert not value.snapshot()["events"]
    assert value.complete(response(value, 1))
    assert [e["segment_id"] for e in value.snapshot()["events"]] == ["s1", "s2"]
    assert not value.complete(response(value, 1))
    assert reserve(value, 1) == value.snapshot()["events"][0]["event_id"]
    with pytest.raises(ValueError, match="another action"):
        reserve(value, 1, request_id="different")


def test_reject_unblocks_order_and_binding_discards_old_results_preserving_history():
    value = ledger()
    reserve(value, 1)
    reserve(value, 2)
    value.complete(response(value, 2))
    value.reject("s1", "quality_interrupt")
    assert len(value.snapshot()["events"]) == 1
    reserve(value, 3)
    value.bind(2, 3)
    assert not value.complete(response(value, 3))
    assert len(value.snapshot()["events"]) == 1


def test_clear_cancels_late_candidates_and_undo_is_idempotent():
    value = ledger()
    for n in (1, 2):
        reserve(value, n)
        value.complete(response(value, n))
    value.undo("undo-1")
    value.undo("undo-1")
    assert len(value.snapshot()["events"]) == 1
    reserve(value, 3)
    value.clear("clear-1")
    value.clear("clear-1")
    assert value.snapshot()["revision"] == 1
    assert not value.complete(response(value, 3))
    assert not value.snapshot()["events"]


def test_capacities_are_enforced_and_evicted_ids_cannot_be_replayed():
    value = ledger(max_pending=1, max_history=2, max_seen=2)
    reserve(value, 1)
    with pytest.raises(RuntimeError, match="capacity"):
        reserve(value, 2)
    for n in range(1, 6):
        if n > 1:
            reserve(value, n)
        value.complete(response(value, n))
    assert len(value.snapshot()["events"]) == 2
    assert len(value._seen) == 2
    with pytest.raises(ValueError, match="expired"):
        reserve(value, 1)


def test_action_capacity_does_not_evict_idempotence_tombstones():
    value = ledger(max_seen=2)
    for n in range(1, 5):
        reserve(value, n)
        value.complete(response(value, n))
    value.undo("a")
    value.undo("b")
    with pytest.raises(RuntimeError, match="action capacity"):
        value.undo("c")
    value.undo("a")
    assert len(value.snapshot()["events"]) == 2


def test_cross_session_event_ids_are_unambiguous():
    first, second = ledger(session_id="a:b"), ledger(session_id="a")
    assert reserve(first, 1, segment_id="c") != reserve(second, 1, segment_id="b:c")
