import pickle

import pytest

from slbridge.slrt.slt_export import (
    SltExportError,
    convert_results_to_annotations,
    load_pickle,
    write_annotation_pickle,
)


def _meta():
    return [
        {"name": "S1", "gloss": "你 好", "text": "你好。", "num_frames": 20},
        {"name": "S2", "gloss": "再 见", "text": "再见。", "num_frames": 30},
    ]


def test_convert_replaces_gloss_and_keeps_text():
    results = {
        "S1": {"window_greedy_7_gls_hyp": "你 好", "gls_ref": "你 好"},
        "S2": {"naive_greedy_gls_hyp": "再 见", "gls_ref": "再 见"},
    }
    items, summary = convert_results_to_annotations(results, _meta())
    assert [item["name"] for item in items] == ["S1", "S2"]
    assert items[0]["gloss"] == "你 好"
    assert items[1]["gloss"] == "再 见"
    assert items[0]["text"] == "你好。"
    assert items[0]["gls_ref"] == "你 好"
    assert items[1]["prediction_decode_key"] == "naive_greedy_gls_hyp"
    assert summary["written"] == 2
    assert summary["empty_predictions"] == 0


def test_missing_prediction_fails_without_allow_missing():
    with pytest.raises(SltExportError):
        convert_results_to_annotations({"S1": {"window_greedy_7_gls_hyp": "你"}}, _meta())


def test_missing_prediction_is_empty_when_allowed():
    items, summary = convert_results_to_annotations(
        {"S1": {"window_greedy_7_gls_hyp": ""}}, _meta(), allow_missing=True
    )
    assert len(items) == 2
    assert items[0]["gloss"] == ""
    assert items[0]["prediction_decode_key"] == "window_greedy_7_gls_hyp"
    assert summary["missing_results"] == 1
    assert summary["empty_predictions"] == 2


def test_duplicate_names_are_rejected():
    with pytest.raises(SltExportError):
        convert_results_to_annotations({}, _meta() + [dict(_meta()[0])], allow_missing=True)


def test_pickle_round_trip(tmp_path):
    path = tmp_path / "csl_pred.dev"
    write_annotation_pickle(path, _meta())
    assert load_pickle(path) == _meta()
    with path.open("rb") as handle:
        assert pickle.load(handle) == _meta()
