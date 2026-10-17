"""Behavioral checks for the observation stream, without a recognition model."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.keypoints import KeypointExtractor, LandmarkFrame, POINT_COUNT, SCHEMA_ID
from slbridge.segmentation import MediaPipeActivityObserver, SegmentConfig
from slbridge.sources import FramePacket


def point(x, y, visibility=1):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def result(*, shift=0.0, left=True, right=True, pose=True):
    body = [point(0.5, 0.5) for _ in range(33)]
    body[11], body[12] = point(0.3, 0.3), point(0.7, 0.3)
    group = lambda x: SimpleNamespace(landmark=[point(x + shift, 0.5) for _ in range(21)])
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=body) if pose else None,
                           left_hand_landmarks=group(0.4) if left else None,
                           right_hand_landmarks=group(0.6) if right else None)


class Graph:
    def __init__(self, stream):
        self.stream = stream
        self.images = []
        self.closed = 0

    def process(self, rgb):
        assert not rgb.flags.writeable
        self.images.append(rgb.copy())
        value = next(self.stream)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self):
        self.closed += 1


class GraphFactory:
    def __init__(self, values):
        self.stream = iter(values)
        self.graphs = []
        self.options = []

    def __call__(self, **options):
        self.options.append(options)
        graph = Graph(self.stream)
        self.graphs.append(graph)
        return graph


def packet(timestamp=0.0, index=0, *, source="test", size=(8, 4)):
    return FramePacket(np.zeros((size[1], size[0], 3), dtype=np.uint8), timestamp, index, source)


def extract(value=None):
    factory = GraphFactory([value if value is not None else result()])
    with KeypointExtractor(session_id="test-session", source_epoch=2, holistic_factory=factory) as extractor:
        return extractor.process(packet())


def test_one_persistent_graph_call_per_frame_and_rgb_is_not_mirrored():
    factory = GraphFactory([result(), result(shift=0.04), result(shift=0.08)])
    extractor = KeypointExtractor(holistic_factory=factory)
    first_packet = packet()
    first_packet.image[:, :4] = [10, 20, 30]
    first_packet.image[:, 4:] = [40, 50, 60]
    first = extractor.process(first_packet)
    second = extractor.process(packet(0.1, 1))
    third = extractor.process(packet(0.3, 4))
    assert len(factory.graphs) == 1 and len(factory.graphs[0].images) == 3
    np.testing.assert_array_equal(factory.graphs[0].images[0][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(factory.graphs[0].images[0][0, -1], [60, 50, 40])
    assert factory.options[0]["static_image_mode"] is False
    assert factory.options[0]["smooth_landmarks"] is True
    assert first.image_size == (8, 4) and first.schema_id == SCHEMA_ID
    assert first.motion == 0 and not first.motion_valid and first.dt is None
    assert first.image_xy[33] == pytest.approx((0.4, 0.5))
    assert first.image_xy[54] == pytest.approx((0.6, 0.5))
    assert first.normalized_xy[33] == pytest.approx((-0.25, 0.25))
    assert second.motion_valid and second.motion == pytest.approx(1.0)
    assert third.motion == pytest.approx(0.5) and third.dt == pytest.approx(0.2)
    assert third.frames_skipped == 2
    extractor.close()
    extractor.close()
    assert factory.graphs[0].closed == 1


def test_outlet_returns_the_same_raw_result_without_changing_legacy_quality():
    partial = result()
    partial.left_hand_landmarks.landmark[0].x = float("nan")
    factory = GraphFactory([partial, result(right=False)])
    with MediaPipeActivityObserver(holistic_factory=factory) as observer:
        activity, raw = observer.observe_landmarks(packet())
        assert raw is partial
        assert not activity.quality_ok and not activity.left_hand_visible
        assert activity.right_hand_visible and activity.motion == 0
        old = observer.observe(packet(0.1, 1))
        assert old.quality_ok and old.left_hand_visible and not old.right_hand_visible
    assert len(factory.graphs[0].images) == 2
    # The stream has its own per-point masks; one bad finger does not erase
    # the other twenty observations or pretend the detector group is absent.
    stream = extract(partial)
    assert stream.left_present and stream.right_present
    assert not stream.point_observed[33] and stream.point_observed[34]
    assert stream.image_xy[33] == (0, 0) and stream.normalized_xy[33] == (0, 0)
    assert sum(stream.point_observed[33:54]) == 20


def test_missing_pose_keeps_image_hand_coordinates_but_does_not_invent_anchors():
    stream = extract(result(pose=False, right=False))
    assert stream.left_present and not stream.right_present
    assert stream.pose_visibility == (None,) * 33
    assert sum(stream.point_observed) == 21
    assert stream.image_xy[33] == (0.4, 0.5)
    assert not stream.anchor_valid and not any(stream.normalized_observed)
    assert stream.normalized_xy == ((0, 0),) * POINT_COUNT


def test_visibility_values_are_preserved_while_low_visibility_points_are_masked():
    raw = result()
    raw.pose_landmarks.landmark[0].visibility = 0.2
    raw.pose_landmarks.landmark[11].visibility = 0.1
    raw.pose_landmarks.landmark[1].visibility = float("nan")
    stream = extract(raw)
    assert stream.pose_visibility[0] == 0.2 and stream.pose_visibility[11] == 0.1
    assert stream.pose_visibility[1] is None
    assert not stream.point_observed[0] and not stream.point_observed[1]
    assert not stream.anchor_valid and not any(stream.normalized_observed)


def test_zero_image_coordinate_is_observed_and_zero_is_not_a_depth_estimate():
    raw = result()
    raw.left_hand_landmarks.landmark[0].x = 0
    raw.left_hand_landmarks.landmark[0].y = 0
    raw.left_hand_landmarks.landmark[0].z = 7  # z is deliberately ignored.
    stream = extract(raw)
    assert stream.point_observed[33] and stream.image_xy[33] == (0, 0)
    assert stream.normalized_observed[33] and len(stream.normalized_xy[33]) == 2


def test_landmark_frame_is_deeply_immutable_and_json_roundtrips_without_aliasing():
    frame = extract()
    with pytest.raises(FrozenInstanceError):
        frame.timestamp = 3
    with pytest.raises(TypeError):
        frame.image_xy[0] = (1, 1)
    with pytest.raises(TypeError):
        frame.image_xy[0][0] = 1
    payload = json.loads(json.dumps(frame.to_dict(), allow_nan=False))
    restored = LandmarkFrame.from_dict(payload)
    assert restored == frame
    payload["image_xy"][0][0] = 0.1
    assert restored.image_xy[0] == (0.5, 0.5)
    with pytest.raises(ValueError, match="immutable"):
        replace(frame, image_xy=list(frame.image_xy))


@pytest.mark.parametrize("field,value", [
    ("schema_id", "pose_hands_xyz225"), ("source_id", ""), ("session_id", " "),
    ("source_epoch", True), ("tracking_epoch", -1), ("frame_index", 0.5),
    ("timestamp", float("nan")), ("received_monotonic", -1),
    ("extracted_monotonic", float("inf")), ("time_domain", "wall_clock"),
    ("frames_skipped", -1), ("image_size", [8, 0]), ("image_size", [8, 4, 2]),
    ("point_observed", [True] * 74), ("point_observed", [1] * 75),
    ("normalized_observed", [False] * 75), ("pose_visibility", [1] * 32),
    ("pose_visibility", [None] * 33), ("left_present", 1), ("anchor_valid", 1),
    ("motion_valid", True), ("motion", 0.1), ("dt", 0), ("reset_reason", ""),
    ("image_xy", [[0, 0]] * 74), ("normalized_xy", [[float("inf"), 0]] * 75),
])
def test_json_rejects_bad_types_dimensions_masks_and_nonfinite_fields(field, value):
    payload = extract().to_dict()
    payload[field] = value
    with pytest.raises(ValueError):
        LandmarkFrame.from_dict(payload)


def test_json_rejects_unknown_missing_mutable_nested_and_inconsistent_fields():
    frame = extract()
    payload = frame.to_dict()
    payload["recognition"] = "hello"
    with pytest.raises(ValueError, match="fields"):
        LandmarkFrame.from_dict(payload)
    payload = frame.to_dict()
    del payload["dt"]
    with pytest.raises(ValueError, match="fields"):
        LandmarkFrame.from_dict(payload)
    with pytest.raises(ValueError, match="JSON array"):
        LandmarkFrame.from_dict({**frame.to_dict(), "image_xy": frame.image_xy})
    payload = frame.to_dict()
    payload["left_present"] = False
    with pytest.raises(ValueError, match="absent hand"):
        LandmarkFrame.from_dict(payload)
    payload = frame.to_dict()
    payload["point_observed"][0] = False
    with pytest.raises(ValueError, match="must be zero"):
        LandmarkFrame.from_dict(payload)
    payload = frame.to_dict()
    payload["image_xy"][0] = [2, 0]
    with pytest.raises(ValueError, match="lie in"):
        LandmarkFrame.from_dict(payload)
    payload = frame.to_dict()
    payload["extracted_monotonic"] = frame.received_monotonic - 0.1
    with pytest.raises(ValueError, match="precede"):
        LandmarkFrame.from_dict(payload)


@pytest.mark.parametrize("invalid", [
    packet(0, 1), packet(0.1, 0), packet(0.1, 1, source="another"),
])
def test_bad_order_or_source_does_not_process_or_corrupt_the_next_frame(invalid):
    factory = GraphFactory([result(), result(shift=0.04)])
    with KeypointExtractor(holistic_factory=factory) as extractor:
        first = extractor.process(packet())
        with pytest.raises(ValueError):
            extractor.process(invalid)
        next_frame = extractor.process(packet(0.1, 1))
    assert len(factory.graphs) == 1 and len(factory.graphs[0].images) == 2
    assert next_frame.tracking_epoch == first.tracking_epoch
    assert next_frame.motion_valid and next_frame.motion == pytest.approx(1)


@pytest.mark.parametrize("next_packet,reason", [
    (packet(1, 4), "timestamp_gap"),
    (packet(0.1, 1, size=(16, 4)), "image_size_changed"),
])
def test_gap_or_image_resize_reopens_tracking_and_invalidates_cross_reset_motion(next_packet, reason):
    factory = GraphFactory([result(), result(shift=0.04), result(shift=0.08)])
    with KeypointExtractor(source_epoch=3, holistic_factory=factory) as extractor:
        first = extractor.process(packet())
        reset = extractor.process(next_packet)
        after = extractor.process(packet(next_packet.timestamp + 0.1, next_packet.frame_index + 1,
                                         size=next_packet.image.shape[1::-1]))
    assert len(factory.graphs) == 2 and all(graph.closed == 1 for graph in factory.graphs)
    assert reset.source_epoch == first.source_epoch == 3
    assert reset.tracking_epoch == first.tracking_epoch + 1
    assert reset.reset_reason == reason and not reset.motion_valid
    assert reset.dt == next_packet.timestamp and reset.frames_skipped == next_packet.frame_index - 1
    assert after.reset_reason is None and after.motion_valid


def test_manual_reset_retains_external_source_epoch_and_accepts_a_fresh_local_timeline():
    factory = GraphFactory([result(), result()])
    extractor = KeypointExtractor(session_id="same-session", source_epoch=7, holistic_factory=factory)
    first = extractor.process(packet(2, 20))
    extractor.reset("operator_reset")
    assert extractor.metadata()["source_id"] is None
    next_frame = extractor.process(packet())
    assert next_frame.session_id == first.session_id and next_frame.source_epoch == 7
    assert next_frame.tracking_epoch == 1 and next_frame.dt is None
    assert next_frame.reset_reason == "operator_reset" and not next_frame.motion_valid
    extractor.close()
    with pytest.raises(RuntimeError, match="closed"):
        extractor.process(packet(0.1, 1))
    with pytest.raises(RuntimeError, match="closed"):
        extractor.reset()


def test_extraction_error_reopens_the_native_graph_and_does_not_advance_source_identity():
    factory = GraphFactory([result(), RuntimeError("native graph failed"), result(shift=0.04)])
    with KeypointExtractor(holistic_factory=factory) as extractor:
        extractor.process(packet())
        with pytest.raises(RuntimeError, match="native graph"):
            extractor.process(packet(0.1, 1))
        retried = extractor.process(packet(0.1, 1))
    assert retried.frame_index == 1 and retried.dt == 0.1
    assert retried.tracking_epoch == 1 and retried.reset_reason == "extraction_error"
    assert not retried.motion_valid and retried.motion == 0
    assert len(factory.graphs) == 2 and all(graph.closed == 1 for graph in factory.graphs)


def test_invalid_receive_time_is_rejected_before_the_native_call(monkeypatch):
    factory = GraphFactory([result()])
    monkeypatch.setattr("slbridge.keypoints.time.perf_counter", lambda: 10.0)
    with KeypointExtractor(holistic_factory=factory) as extractor:
        for value in (-1, float("nan"), 11, True):
            with pytest.raises(ValueError):
                extractor.process(packet(), received_monotonic=value)
        frame = extractor.process(packet(), received_monotonic=9.5)
    assert len(factory.graphs[0].images) == 1
    assert frame.received_monotonic == 9.5 and frame.extracted_monotonic == 10


def test_motion_does_not_bridge_missing_observations():
    factory = GraphFactory([result(), result(left=False, right=False), result(shift=0.04)])
    with KeypointExtractor(holistic_factory=factory) as extractor:
        extractor.process(packet())
        missing = extractor.process(packet(0.1, 1))
        restored = extractor.process(packet(0.2, 2))
    assert not missing.motion_valid and not restored.motion_valid
    assert missing.motion == restored.motion == 0


@pytest.mark.parametrize("mode", ["tracking_reset", "missing_hands"])
def test_serialized_motion_cannot_claim_validity_across_reset_or_missing_hands(mode):
    factory = GraphFactory([result(), result(shift=0.04)])
    with KeypointExtractor(holistic_factory=factory) as extractor:
        extractor.process(packet())
        valid = extractor.process(packet(0.1, 1))
    payload = valid.to_dict()
    if mode == "tracking_reset":
        payload["reset_reason"] = "manual"
    else:
        payload["left_present"] = payload["right_present"] = False
        for index in range(33, 75):
            payload["point_observed"][index] = payload["normalized_observed"][index] = False
            payload["image_xy"][index] = payload["normalized_xy"][index] = [0, 0]
    with pytest.raises(ValueError, match="valid motion requires observed"):
        LandmarkFrame.from_dict(payload)


def test_metadata_is_serializable_and_protocol_hash_excludes_stream_identity():
    first = KeypointExtractor(session_id="a", source_epoch=0, holistic_factory=GraphFactory([result()]))
    second = KeypointExtractor(session_id="b", source_epoch=7, time_domain="application_monotonic",
                               holistic_factory=GraphFactory([]))
    changed = KeypointExtractor(SegmentConfig(pose_visibility=0.6), holistic_factory=GraphFactory([]))
    before = first.metadata()
    first.process(packet())
    first.reset()
    after = first.metadata()
    assert before["protocol_hash"] == after["protocol_hash"] == second.metadata()["protocol_hash"]
    assert changed.metadata()["protocol_hash"] != before["protocol_hash"]
    assert before["coordinate_dimensions"] == 2
    assert before["protocol"]["point_order"] == [
        {"group": "pose", "start": 0, "count": 33},
        {"group": "left_hand", "start": 33, "count": 21},
        {"group": "right_hand", "start": 54, "count": 21}]
    encoded = json.dumps(before["protocol"], sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == before["protocol_hash"]
    json.dumps(after, allow_nan=False)
    first.close()
    second.close()
    changed.close()


@pytest.mark.parametrize("options", [
    {"config": {}}, {"session_id": ""}, {"source_epoch": True},
    {"time_domain": "video"}, {"holistic_factory": 1},
])
def test_extractor_rejects_invalid_configuration(options):
    with pytest.raises((TypeError, ValueError)):
        KeypointExtractor(**options)
