import base64
from io import BytesIO
import math

import pytest
from PIL import Image

from slbridge import realtime_protocol as p


def jpeg(size=(64, 64)):
    output = BytesIO()
    Image.new('RGB', size, 'blue').save(output, format='JPEG')
    return base64.b64encode(output.getvalue()).decode('ascii')


def frame(index=0, timestamp=0, epoch=3):
    return dict(type='frame', index=index, timestamp=timestamp, source_epoch=epoch,
                capture_ms=123.4, jpeg=jpeg())


def test_stream_start_is_explicit_and_csl_only():
    s = p.validate_start(dict(type='start', protocol=p.PROTOCOL, sign_language='csl', source_epoch=3, fps=30))
    assert s['source_epoch'] == 3 and s['fps'] == 30
    with pytest.raises(ValueError):
        p.validate_start(dict(s, sign_language='libras'))
    with pytest.raises(ValueError):
        p.validate_start(dict(s, endpoint='http://elsewhere'))


def test_frame_decoder_keeps_real_time_and_bgr_pixels():
    d = p.FrameDecoder(source_epoch=3, source_id='session-test')
    first = d.decode(frame())
    later = d.decode(frame(2, .1))
    assert later.packet.frame_index == 2 and later.packet.timestamp == .1
    assert first.packet.image.shape == (64, 64, 3)
    assert first.packet.image[0, 0, 0] > 240 and first.packet.image[0, 0, 2] < 10
    assert first.capture_ms == 123.4


@pytest.mark.parametrize('update', [dict(index=True), dict(timestamp=math.nan), dict(timestamp=-1),
                                  dict(source_epoch=4), dict(capture_ms=math.inf), dict(jpeg='not base64'),
                                  dict(jpeg='A'*400000), dict(jpeg=jpeg((513, 16))), dict(gt_text='reference')])
def test_bad_or_oversized_frames_are_rejected_before_inference(update):
    d = p.FrameDecoder(source_epoch=3, source_id='test')
    with pytest.raises(ValueError):
        d.decode(dict(frame(), **update))


def test_replayed_frames_or_nonmonotonic_time_are_rejected():
    d = p.FrameDecoder(source_epoch=3, source_id='test')
    d.decode(frame())
    with pytest.raises(ValueError):
        d.decode(frame())
    with pytest.raises(ValueError):
        d.decode(frame(1, 0))


def test_wire_parser_rejects_large_or_duplicate_key_messages():
    with pytest.raises(ValueError):
        p.parse_message('{"type":"stop","type":"start"}')
    with pytest.raises(ValueError):
        p.parse_message(' '*400000)
    with pytest.raises(ValueError):
        p.parse_message('[1,2]')


def test_assistance_is_an_explicit_optional_boolean():
    start=dict(type='start',protocol=p.PROTOCOL,sign_language='csl',source_epoch=3,fps=30,assistance=True)
    assert p.validate_start(start)['assistance'] is True
    with pytest.raises(ValueError):p.validate_start(dict(start,assistance='true'))
