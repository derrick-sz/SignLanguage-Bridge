"""Manual rotation fixtures for the published SignSparK stream boundary."""

import numpy as np
import pytest

from slbridge.animation.signspark_codec import decode_signspark_streams


IDENTITY = [1, 0, 0, 0, 1, 0]


def streams(frames=2):
    body = np.tile(IDENTITY, (frames, 10)).astype(np.float32)
    left = np.tile(IDENTITY, (frames, 15)).astype(np.float32)
    right = left.copy()
    face = np.zeros((frames, 56), dtype=np.float64)
    face[:, :6] = IDENTITY
    return body, left, right, face


def test_flat_hand_identity_has_no_added_mean_and_expression_is_preserved():
    body, left, right, face = streams()
    # Non-normalized, signed coefficients beyond the old ten-expression schema.
    face[0, 6:] = np.linspace(-4.125, 8.375, 50)
    face[1, 55] = 1.123456789012345
    result = decode_signspark_streams(
        body, left, right, face, left_hand_convention="model-right"
    )
    np.testing.assert_array_equal(result.poses, np.zeros((2, 55, 3)))
    assert result.poses.dtype == np.float32
    np.testing.assert_array_equal(result.expression, face[:, 6:])
    assert result.expression.dtype == face.dtype
    assert not np.shares_memory(result.expression, face)


def test_manual_row_rotations_map_to_correct_joints_and_left_handedness():
    body, left, right, face = streams(1)
    # Literal first TWO ROWS of active rotations, not columns.
    z90 = [0, -1, 0, 1, 0, 0]
    y90 = [0, 0, 1, 0, 1, 0]
    x90 = [1, 0, 0, 0, 0, -1]
    x180 = [1, 0, 0, 0, -1, 0]
    z180 = [-1, 0, 0, 0, -1, 0]
    body[0, :6] = z90
    body[0, -6:] = z180
    left[0, :18] = x90 + y90 + z90
    right[0, :6] = y90
    face[0, :6] = x180
    originals = [value.copy() for value in (body, left, right, face)]
    for value in (body, left, right, face):
        value.setflags(write=False)
    result = decode_signspark_streams(
        body, left, right, face, left_hand_convention="model-right"
    )
    expected = np.zeros((1, 55, 3))
    expected[0, 12] = [0, 0, np.pi / 2]
    expected[0, 21] = [0, 0, np.pi]
    expected[0, 22] = [np.pi, 0, 0]
    # Conjugation preserves X rotations, reverses Y/Z; it does not swap hands.
    expected[0, 25] = [np.pi / 2, 0, 0]
    expected[0, 26] = [0, -np.pi / 2, 0]
    expected[0, 27] = [0, 0, -np.pi / 2]
    expected[0, 40] = [0, np.pi / 2, 0]
    np.testing.assert_allclose(result.poses, expected, atol=2e-7)
    for value, original in zip((body, left, right, face), originals):
        np.testing.assert_array_equal(value, original)


def test_native_left_is_not_flipped_again_and_scaled_rows_are_orthogonalized():
    body, left, right, face = streams(1)
    # First row scaled by 2; second row has a parallel component.
    left[0, :6] = [0, -2, 0, 4, -3, 0]
    result = decode_signspark_streams(
        body, left, right, face, left_hand_convention="smplx-left"
    )
    np.testing.assert_allclose(result.poses[0, 25], [0, 0, np.pi / 2], atol=1e-7)


@pytest.mark.parametrize("convention", ["left", "right", "", None, True])
def test_unknown_left_hand_convention_is_rejected(convention):
    with pytest.raises(ValueError, match="left_hand_convention"):
        decode_signspark_streams(*streams(), left_hand_convention=convention)


@pytest.mark.parametrize("index,name", enumerate(["body", "left", "right", "face"]))
@pytest.mark.parametrize("fault", ["width", "rank", "frames", "nan", "inf", "object", "complex", "bool"])
def test_invalid_stream_is_rejected_without_padding_or_type_coercion(index, name, fault):
    values = list(streams())
    source = values[index]
    if fault == "width":
        values[index] = source[:, :-1]
    elif fault == "rank":
        values[index] = source[None, ...]
    elif fault == "frames":
        values[index] = source[:1]
    elif fault in ("nan", "inf"):
        # For face this also verifies that all 50 expression channels are checked.
        values[index][0, -1] = np.nan if fault == "nan" else np.inf
    else:
        values[index] = source.astype({"object": object, "complex": complex, "bool": bool}[fault])
    with pytest.raises(ValueError, match="frames" if fault == "frames" else name):
        decode_signspark_streams(*values, left_hand_convention="model-right")


def test_empty_sequences_are_rejected():
    with pytest.raises(ValueError, match="frames"):
        decode_signspark_streams(*streams(0), left_hand_convention="model-right")


@pytest.mark.parametrize("index,name", enumerate(["body", "left", "right", "face"]))
@pytest.mark.parametrize("bad_rotation", [
    [0, 0, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 2, 0, 0],
    [1, 0, 0, -2, 0, 0],
    [1, 0, 0, 1, 1e-9, 0],
    [1e-12, 0, 0, 0, 1, 0],
])
def test_degenerate_rotation_cannot_silently_become_identity(index, name, bad_rotation):
    values = list(streams())
    values[index][1, :6] = bad_rotation
    with pytest.raises(ValueError, match=f"{name}.*degenerate.*frame 1"):
        decode_signspark_streams(*values, left_hand_convention="model-right")


def test_large_finite_rows_do_not_overflow_normalization():
    body, left, right, face = streams(1)
    body = body.astype(np.float64)
    body[0, :6] = [0, -1e200, 0, 1e200, 0, 0]
    with np.errstate(all="raise"):
        result = decode_signspark_streams(
            body, left, right, face, left_hand_convention="model-right"
        )
    np.testing.assert_allclose(result.poses[0, 12], [0, 0, np.pi / 2], atol=1e-7)
