"""Action boundary behavior, without assuming recognition or heuristic accuracy."""

from dataclasses import replace
import subprocess
import sys
import re
from types import SimpleNamespace

import numpy as np
import pytest

from slbridge.segmentation import (
    ActivityObservation, AutoSegmenter, MediaPipeActivityObserver, SegmentConfig,
)
from slbridge.sources import FramePacket


def packet(timestamp, index, *, source="test", width=8, height=4):
    return FramePacket(np.full((height, width, 3), index % 256, dtype=np.uint8),
                       timestamp, index, source)


def feed(segmenter, timestamp, index, *, motion=1.0, active=True, hands=True,
         quality=True, left=None, right=None, **packet_options):
    return segmenter.feed(packet(timestamp, index, **packet_options), ActivityObservation(
        timestamp, hands, quality, motion, active, left, right))


def ready(notices):
    return [notice.segment for notice in notices if notice.kind == "ready"]


def short_config(**options):
    config = SegmentConfig(start_candidate_seconds=0.1, end_candidate_seconds=0.3,
                           min_segment_seconds=0.2, pre_buffer_seconds=0.1,
                           post_buffer_seconds=0.1, rearm_seconds=0.1,
                           quality_gap_seconds=0.2)
    return replace(config, **options)


def complete_action(segmenter, *, origin=0, first_index=0, static=False):
    notices = []
    for index, t in enumerate((0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)):
        notices.extend(feed(segmenter, origin + t, first_index + index,
                            motion=0.0 if static or t >= 0.3 else 1.0))
    return notices


def test_module_import_is_lazy():
    subprocess.run([sys.executable, "-c", "import sys; import slbridge.segmentation; "
                    "assert not ({'numpy','cv2','mediapipe'} & set(sys.modules))"], check=True)


@pytest.mark.parametrize("options", [
    {"config_version": 2}, {"start_candidate_seconds": 0},
    {"end_candidate_seconds": float("nan")}, {"pre_buffer_seconds": -1},
    {"min_segment_seconds": 9}, {"max_bytes": True}, {"max_frames": 1},
    {"end_motion_threshold": 1}, {"pose_visibility": 0}, {"static_active_x": 0},
])
def test_config_rejects_invalid_ranges(options):
    with pytest.raises(ValueError):
        SegmentConfig(**options)


def test_idle_without_hands_or_resting_hands_never_yields_a_clip():
    segmenter = AutoSegmenter(short_config(max_frames=4, max_bytes=4 * 8 * 4 * 3,
                                          max_buffer_seconds=0.25))
    for index in range(50):
        notices = feed(segmenter, index / 10, index, motion=0, active=False,
                       hands=index % 2 == 0, quality=index % 2 == 0)
        assert not ready(notices)
        assert segmenter.state == "WAITING"
        assert segmenter.buffered_frames <= 4
        assert segmenter.buffered_bytes <= 4 * 8 * 4 * 3
        assert segmenter.buffered_seconds <= 0.25 + 1e-9
    assert segmenter.flush() == ()


def test_complete_action_preserves_source_times_and_uses_real_pre_post_buffers():
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 10.0, 0, motion=0, active=False)
    feed(segmenter, 10.05, 1, motion=0, active=False)
    notices = complete_action(segmenter, origin=10.1, first_index=2)
    segment, = ready(notices)
    assert segment.start_time == 10.0
    assert segment.action_end_time == pytest.approx(10.4)
    assert segment.end_time == pytest.approx(10.5)
    assert segment.sequence == 1 and segment.source_id == "test"
    assert re.fullmatch(r"segment:[0-9a-f]{32}:1", segment.segment_id)
    assert segment.clip.frames[0].timestamp == 0
    assert segment.clip.frames[-1].timestamp == pytest.approx(0.5)
    assert [item.frame_index for item in segment.clip.frames] == list(range(7))
    assert all(not item.image.flags.writeable for item in segment.clip.frames)
    assert segmenter.state == "REARM" and segmenter.buffered_bytes == 0


def test_static_active_hand_can_start_without_movement_and_held_pose_commits_once():
    segmenter = AutoSegmenter(short_config())
    segment, = ready(complete_action(segmenter, static=True))
    assert segment.action_end_time == pytest.approx(0.2)
    for index in range(7, 60):
        assert not ready(feed(segmenter, index / 10, index, motion=0, active=True))
        assert segmenter.state == "REARM"


def test_internal_pause_resuming_activity_does_not_split_the_action():
    segmenter = AutoSegmenter(short_config())
    output = []
    for index, motion in enumerate((1, 1, 0, 0, 0.2, 1, 0, 0, 0, 0)):
        output.extend(feed(segmenter, index / 10, index, motion=motion))
    segment, = ready(output)
    assert segment.action_end_time == pytest.approx(0.6)
    assert any(notice.reason == "activity_resumed" for notice in output)


def test_short_start_glitch_cancels_and_short_action_is_explicitly_rejected():
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0)
    assert any(notice.reason == "start_cancelled" for notice in feed(
        segmenter, 0.05, 1, motion=0, active=False))
    assert segmenter.state == "WAITING"
    output = []
    for index, t in enumerate((0.1, 0.2, 0.3, 0.4, 0.5), start=2):
        output.extend(feed(segmenter, t, index, motion=1 if t == 0.1 else 0))
    assert not ready(output)
    assert any(notice.reason == "segment_too_short" for notice in output)


@pytest.mark.parametrize("hands,quality", [(False, False), (True, False)])
def test_quality_loss_or_leaving_never_counts_as_successful_pause(hands, quality):
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0)
    feed(segmenter, 0.1, 1)
    feed(segmenter, 0.2, 2, motion=0)
    output = []
    for index, t in enumerate((0.3, 0.4, 0.5), start=3):
        output.extend(feed(segmenter, t, index, motion=0, hands=hands, quality=quality))
    assert not ready(output)
    assert any(notice.reason == "quality_lost" for notice in output)


def test_brief_occlusion_restarts_end_timer_and_long_occlusion_rejects_on_recovery():
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0)
    feed(segmenter, 0.1, 1)
    feed(segmenter, 0.2, 2, motion=0)
    feed(segmenter, 0.3, 3, motion=0, quality=False)
    assert not ready(feed(segmenter, 0.4, 4, motion=0))
    assert not ready(feed(segmenter, 0.5, 5, motion=0))
    assert not ready(feed(segmenter, 0.6, 6, motion=0))
    segment, = ready(feed(segmenter, 0.7, 7, motion=0))
    assert segment.action_end_time == 0.4
    segmenter.reset()
    feed(segmenter, 1.0, 0)
    feed(segmenter, 1.1, 1)
    feed(segmenter, 1.2, 2, quality=False)
    result = feed(segmenter, 1.45, 3, motion=0)
    assert any(item.reason == "quality_lost" for item in result)
    assert not ready(result)


def test_losing_one_of_two_observed_hands_is_quality_loss():
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0, left=True, right=True)
    feed(segmenter, 0.1, 1, left=True, right=True)
    output = []
    for index, t in enumerate((0.2, 0.3, 0.4), start=2):
        output.extend(feed(segmenter, t, index, motion=0, left=True, right=False))
    assert not ready(output)
    assert any(item.reason == "quality_lost" for item in output)


@pytest.mark.parametrize("repeat_mode", ["release", "motion"])
def test_rearming_allows_two_distinct_actions_without_text_deduplication(repeat_mode):
    segmenter = AutoSegmenter(short_config())
    first, = ready(complete_action(segmenter))
    output = []
    if repeat_mode == "release":
        feed(segmenter, 0.7, 7, motion=0, active=False, hands=False, quality=False)
        feed(segmenter, 0.8, 8, motion=0, active=False, hands=False, quality=False)
        output = complete_action(segmenter, origin=0.9, first_index=9)
    else:
        for index in range(7, 15):
            output.extend(feed(segmenter, index / 10, index, motion=1 if index <= 10 else 0))
    second, = ready(output)
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.segment_id != second.segment_id
    assert second.start_time > first.end_time


def test_overlong_action_rejects_whole_clip_and_cannot_restart_while_continuing():
    segmenter = AutoSegmenter(short_config(max_segment_seconds=0.4))
    output = []
    for index in range(15):
        output.extend(feed(segmenter, index / 10, index))
    assert not ready(output)
    assert len([item for item in output if item.kind == "rejected"]) == 1
    assert any(item.reason == "segment_too_long" for item in output)
    assert segmenter.state == "REARM"


@pytest.mark.parametrize("options,reason", [
    ({"max_frames": 3}, "buffer_frame_limit"),
    ({"max_bytes": 3 * 8 * 4 * 3}, "buffer_byte_limit"),
    ({"max_buffer_seconds": 0.25}, "buffer_time_limit"),
])
def test_active_buffer_limits_reject_instead_of_evicting_part_of_the_action(options, reason):
    segmenter = AutoSegmenter(short_config(**options))
    output = []
    for index in range(7):
        output.extend(feed(segmenter, index / 10, index))
        assert segmenter.buffered_frames <= segmenter.config.max_frames
        assert segmenter.buffered_bytes <= segmenter.config.max_bytes
        assert segmenter.buffered_seconds <= segmenter.config.max_buffer_seconds + 1e-9
    assert not ready(output)
    assert any(item.kind == "rejected" and item.reason == reason for item in output)


def test_resized_owned_images_are_unmirrored_and_not_changed_by_source_mutation():
    segmenter = AutoSegmenter(short_config(max_width=4))
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :4] = [10, 20, 30]
    image[:, 4:] = [40, 50, 60]
    source = FramePacket(image, 0, 0, "test")
    segmenter.feed(source, ActivityObservation(0, True, True, 1, True))
    image[:] = 255
    output = []
    for index in range(1, 7):
        output.extend(feed(segmenter, index / 10, index, motion=1 if index < 3 else 0))
    segment, = ready(output)
    first = segment.clip.frames[0].image
    assert first.shape == (2, 4, 3)
    np.testing.assert_array_equal(first[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(first[0, -1], [40, 50, 60])


@pytest.mark.parametrize("bad_packet,bad_observation", [
    (packet(0, 1), ActivityObservation(0, True, True, 1)),
    (packet(0.1, 0), ActivityObservation(0.1, True, True, 1)),
    (packet(0.1, 1, source="another"), ActivityObservation(0.1, True, True, 1)),
    (packet(0.1, 1), ActivityObservation(0.2, True, True, 1)),
    (packet(0.1, 1, width=9), ActivityObservation(0.1, True, True, 1)),
])
def test_timestamps_indices_source_dimensions_and_observation_time_are_strict(bad_packet, bad_observation):
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0)
    with pytest.raises(ValueError):
        segmenter.feed(bad_packet, bad_observation)
    assert segmenter.buffered_frames == 1


def test_sparse_timestamps_cannot_confirm_a_pause_and_reset_changes_source():
    segmenter = AutoSegmenter(short_config())
    feed(segmenter, 0, 0)
    feed(segmenter, 0.1, 1)
    feed(segmenter, 0.2, 2, motion=0)
    output = feed(segmenter, 3, 3, motion=0)
    assert not ready(output)
    assert any(item.reason == "timestamp_gap" for item in output)
    segmenter.reset()
    assert segmenter.state == "WAITING" and segmenter.buffered_frames == 0
    feed(segmenter, 0, 0, source="new-source")


def test_segment_ids_are_ascii_and_unique_across_reset_and_new_instances():
    source = "video:E:/素材/105_Árvore/INFORMANTE 3/IMG_2971.mp4"
    segmenter = AutoSegmenter(short_config())
    segments = []
    for cycle in range(3):
        if cycle == 1:
            segmenter.reset()
        elif cycle == 2:
            segmenter = AutoSegmenter(short_config())
        output = []
        for index in range(7):
            output.extend(feed(segmenter, index / 10, index,
                               motion=1 if index < 3 else 0, source=source))
        segment, = ready(output)
        assert re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", segment.segment_id)
        assert segment.source_id == source and segment.clip.source_id == source
        segments.append(segment)
    assert len({segment.segment_id for segment in segments}) == 3
    assert [segment.sequence for segment in segments] == [1, 2, 1]


@pytest.mark.parametrize("frames", [1, 2, 4])
def test_eof_does_not_force_an_unfinished_candidate_or_action_into_ready(frames):
    segmenter = AutoSegmenter(short_config())
    for index in range(frames):
        feed(segmenter, index / 10, index, motion=0 if index >= 2 else 1)
    output = segmenter.flush()
    assert not ready(output)
    assert any(item.reason == "incomplete_at_eof" for item in output)
    assert segmenter.buffered_bytes == 0 and segmenter.flush() == ()


def landmark(x, y, visibility=1):
    return SimpleNamespace(x=x, y=y, visibility=visibility)


def results(*, left_y=0.5, right_y=0.5, shift=0, bad_anchor=False, invalid_hand=False):
    pose = [landmark(0.5, 0.5) for _ in range(33)]
    pose[11] = landmark(0.3, 0.3, visibility=0 if bad_anchor else 1)
    pose[12] = landmark(0.7, 0.3)
    group = lambda y: None if y is None else SimpleNamespace(landmark=[
        landmark(0.4 + shift, y) for _ in range(21)])
    left, right = group(left_y), group(right_y)
    if invalid_hand and left is not None:
        left.landmark[0].x = float("nan")
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=pose),
                           left_hand_landmarks=left, right_hand_landmarks=right)


class Graph:
    def __init__(self, result_sequence):
        self.results = iter(result_sequence)
        self.images = []
        self.close_count = 0

    def process(self, image):
        self.images.append(image.copy())
        assert not image.flags.writeable
        return next(self.results)

    def close(self):
        self.close_count += 1


def test_observer_tracks_both_real_landmark_groups_and_motion_uses_timestamp_seconds():
    graph = Graph([results(), results(shift=0.04), results(shift=0.08),
                   results(left_y=0.95, right_y=0.95, shift=0.08)])
    options = []
    observer = MediaPipeActivityObserver(holistic_factory=lambda **kw: options.append(kw) or graph)
    first = observer.observe(packet(0, 0, height=8))
    second = observer.observe(packet(0.1, 1, height=8))
    third = observer.observe(packet(0.3, 2, height=8))
    resting = observer.observe(packet(0.4, 3, height=8))
    assert first.quality_ok and first.pose_active and first.motion == 0
    assert first.left_hand_visible and first.right_hand_visible
    assert second.motion == pytest.approx(1.0)
    assert third.motion == pytest.approx(0.5)
    assert resting.hands_visible and not resting.pose_active
    assert options[0]["model_complexity"] == 1 and options[0]["static_image_mode"] is False
    observer.close()
    observer.close()
    assert graph.close_count == 1
    with pytest.raises(RuntimeError, match="closed"):
        observer.observe(packet(0.5, 4))


@pytest.mark.parametrize("result", [results(bad_anchor=True), results(invalid_hand=True),
                                         results(left_y=None, right_y=None)])
def test_observer_marks_anchor_or_hand_failure_as_bad_quality(result):
    observer = MediaPipeActivityObserver(holistic_factory=lambda **options: Graph([result]))
    observed = observer.observe(packet(0, 0))
    assert not observed.quality_ok and not observed.pose_active and observed.motion == 0
    observer.close()


def test_observer_converts_bgr_to_rgb_without_mirroring_and_one_hand_is_sufficient():
    graph = Graph([results(right_y=None)])
    observer = MediaPipeActivityObserver(holistic_factory=lambda **options: graph)
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, :4] = [11, 22, 33]
    image[:, 4:] = [44, 55, 66]
    result = observer.observe(FramePacket(image, 0, 0, "test"))
    assert result.quality_ok and result.left_hand_visible and not result.right_hand_visible
    np.testing.assert_array_equal(graph.images[0][0, 0], [33, 22, 11])
    np.testing.assert_array_equal(graph.images[0][0, -1], [66, 55, 44])
    observer.close()
