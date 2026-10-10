"""Real RGB observation -> timestamp endpoint -> localhost model API -> Qt history.

Uses a trusted existing model and legal local video. No new training. Replaying
recorded activity observations is evidence of the engineering chain, not live
camera quality. Video segments are losslessly encoded for the HTTP upload.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

from slbridge._io import configure_utf8_streams, write_json_atomic


def controlled_pause(video, output, pause_at, hold_seconds):
    """Create an explicitly synthetic pause fixture from actual RGB frames."""
    import cv2
    from slbridge.sources import VideoSource
    if not .2 <= pause_at <= 10 or not .8 <= hold_seconds <= 2:
        raise ValueError("Controlled pause requires pause-at .2..10 and hold-seconds .8..2")
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    with VideoSource(video) as source:
        while (packet := source.read()) is not None and packet.timestamp <= pause_at:
            frames.append(packet)
    if len(frames) < 2:
        raise ValueError("Insufficient original frames for pause fixture")
    fps = (len(frames) - 1) / (frames[-1].timestamp - frames[0].timestamp)
    h, w = frames[0].image.shape[:2]
    path = output / "controlled-pause.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("Lossless FFV1 encoder unavailable")
    try:
        for frame in frames:
            writer.write(frame.image)
        for _ in range(round(hold_seconds * fps)):
            writer.write(frames[-1].image)
    finally:
        writer.release()
    write_json_atomic(output / "fixture.json", dict(kind="synthetic_static_pause_on_real_rgb",
                      original_video=str(video.resolve()), original_frames=len(frames),
                      pause_at=pause_at, hold_seconds=hold_seconds, fps=fps,
                      limitation="Engineering replay only; no live endpoint or recognition accuracy claim"))
    return path


def validate(model, video, catalog, output):
    import cv2
    import httpx
    from PySide6.QtWidgets import QApplication
    from slbridge.desktop import SignLanguageApp
    from slbridge.segmentation import AutoSegmenter, MediaPipeActivityObserver, SegmentConfig
    from slbridge.session import RecognitionSession
    from slbridge.sources import VideoSource
    from slbridge.runtime_options import RuntimeOptions

    output.mkdir(parents=True, exist_ok=True)
    config = SegmentConfig()
    observer, segmenter = MediaPipeActivityObserver(config), AutoSegmenter(config)
    observations, segments, notices, timings = {}, [], [], {}
    started = time.monotonic()
    try:
        with VideoSource(video) as source:
            while (packet := source.read()) is not None:
                observation = observer.observe(packet)
                observations[packet.frame_index] = observation
                for notice in segmenter.feed(packet, observation):
                    notices.append(dict(kind=notice.kind, state=notice.state, reason=notice.reason,
                                        timestamp=packet.timestamp))
                    if notice.segment:
                        segments.append(notice.segment)
        notices.extend(dict(kind=n.kind, state=n.state, reason=n.reason) for n in segmenter.flush())
    finally:
        observer.close()
    timings["observation_replay_wall_ms"] = (time.monotonic() - started) * 1000
    evidence = dict(video=str(video.resolve()), model=str(model.resolve()), config=config.to_dict(),
                    source_frames=len(observations), activity_observations=[asdict(o) for o in observations.values()],
                    notices=notices, ready_count=len(segments), timing_ms=timings,
                    camera_status="skipped", accuracy_status="not_evaluated",
                    end_to_end_latency_status="not_live_camera_measurement")
    write_json_atomic(output / "replay.json", evidence)
    if not segments:
        raise RuntimeError("No completed segment under defaults; see replay.json. This is not a pass.")

    # Encode the actual selected RGB frames with a lossless codec; record their
    # original capture times separately. The fixture is an actual video replay.
    encoded = []
    for index, segment in enumerate(segments):
        path = output / f"segment-{index+1}.avi"
        frames = segment.clip.frames
        seconds = frames[-1].timestamp - frames[0].timestamp
        fps = (len(frames) - 1) / seconds
        h, w = frames[0].image.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError("Lossless FFV1 encoder unavailable")
        try:
            for frame in frames:
                writer.write(frame.image)
        finally:
            writer.release()
        encoded.append(path)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    api_catalog = output / "model-catalog.json"
    write_json_atomic(api_catalog, {"schema_version": 1, "models": {"smoke": str(model.resolve())}})
    server_log = (output / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", "slbridge.api_cli", "--catalog", str(api_catalog),
                                "--model", "smoke", "--device", "cpu", "--port", str(port)],
                               stdout=server_log, stderr=subprocess.STDOUT,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    responses = []
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=45, trust_env=False) as client:
            deadline = time.monotonic() + 40
            while True:
                if process.poll() is not None:
                    raise RuntimeError("Model server exited; see server.log")
                try:
                    health = client.get("/v1/health")
                    if health.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model server startup timeout")
                time.sleep(.1)
            descriptor = health.json()["status"]["active"]
            language, generation = descriptor["sign_language"], descriptor["generation"]
            caps = client.get("/v3/models")
            caps.raise_for_status()
            opened = client.post("/v3/caption-sessions", json={"session_id": "real-rgb-replay", "sign_language": language})
            opened.raise_for_status()
            for index, (segment, path) in enumerate(zip(segments, encoded), 1):
                fields = dict(request_id=f"replay-{index}", session_id="real-rgb-replay",
                              segment_id=segment.segment_id, sequence=str(index), source_epoch="0",
                              expected_generation=str(generation), sign_language=language,
                              representation_id="mediapipe225_rgb_v1", task_type="classification",
                              start_time=str(segment.start_time), end_time=str(segment.end_time),
                              source_id="video-replay")
                began = time.monotonic()
                with path.open("rb") as stream:
                    response = client.post("/v3/recognitions", data=fields,
                                           files={"file": (path.name, stream, "video/x-msvideo")})
                response.raise_for_status()
                body = response.json()
                body["measured_http_wall_ms"] = (time.monotonic() - began) * 1000
                responses.append(body)
                if body["result"]["status"] != "recognized" or not body["caption_committed"]:
                    raise AssertionError(body)
                with path.open("rb") as stream:
                    duplicate = client.post("/v3/recognitions", data=fields,
                                            files={"file": (path.name, stream, "video/x-msvideo")})
                duplicate.raise_for_status()
                assert duplicate.json()["caption_event_id"] == body["caption_event_id"]
            history = client.get("/v3/caption-sessions/real-rgb-replay").json()
            assert len(history["snapshot"]["events"]) == len(responses)
            undo = client.post("/v3/caption-sessions/real-rgb-replay/actions",
                               json={"action_id": "undo-once", "action": "undo"})
            undo.raise_for_status()
            assert len(undo.json()["snapshot"]["events"]) == len(responses) - 1
            clear = client.post("/v3/caption-sessions/real-rgb-replay/actions",
                                json={"action_id": "clear-once", "action": "clear"})
            clear.raise_for_status()
            assert not clear.json()["snapshot"]["events"]
            evidence.update(http_responses=responses, http_history=history, http_capabilities=caps.json())
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        server_log.close()
        write_json_atomic(output / "replay.json", evidence)

    # Cached observations were computed above by a real MediaPipe graph. The
    # mounted checkbox drives production capture/segmentation/inference/event
    # code; RGB clip preparation still runs the saved extractor afresh.
    class ReplayObserver:
        def __init__(self, _config):
            pass
        def observe(self, packet):
            return observations[packet.frame_index]
        def close(self):
            pass

    class GatedVideo(VideoSource):
        def read(self):
            while not gate[0]:
                time.sleep(.005)
            return super().read()

    gate = [False]
    application = QApplication.instance() or QApplication([])
    session = RecognitionSession(options=RuntimeOptions(device="cpu"), observer_factory=ReplayObserver)
    window = SignLanguageApp(session=session, segment_config=config)
    window.load_catalog(catalog)
    window.load_model(model)
    window.show()

    def pump(predicate, seconds=40):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            application.processEvents()
            if session.state == "ERROR":
                raise AssertionError(window._error_detail)
            if predicate():
                return
            time.sleep(.005)
        raise TimeoutError("Mounted Qt replay did not reach expected behavior")

    try:
        pump(lambda: session.recognizer is not None and not session.busy)
        session.open_source(GatedVideo(video), kind="video", video_path=video)
        pump(lambda: session.source_active and window.auto_check.isEnabled())
        assert window.auto_check.isEnabled()
        window.auto_check.click()
        assert session.automatic
        gate[0] = True
        pump(lambda: window.caption_history.count() == len(segments))
        evidence["qt_history"] = session.captions.snapshot()
        evidence["qt_text"] = [window.caption_history.item(i).text() for i in range(window.caption_history.count())]
        evidence["qt_automatic_status"] = session.automatic_status()
        window.grab().save(str(output / "mounted-history.png"))
        before_undo = window.caption_history.count()
        window.undo_caption_button.click()
        assert window.caption_history.count() == before_undo - 1
        evidence["qt_undo_passed"] = True
        if window.caption_history.count() == 0:
            # Clear must act on a nonempty history; clicking a disabled button
            # after undo is not evidence of the mounted clear action.
            pump(lambda: not session.source_active and not session.busy)
            gate[0] = False
            session.open_source(GatedVideo(video), kind="video", video_path=video)
            pump(lambda: session.source_active and window.auto_check.isEnabled()
                 and not window.auto_check.isChecked())
            window.auto_check.click()
            assert session.automatic
            gate[0] = True
            pump(lambda: window.caption_history.count() > 0)
            evidence["qt_replay_after_undo"] = session.captions.snapshot()
        assert window.clear_history_button.isEnabled()
        window.clear_history_button.click()
        assert window.caption_history.count() == 0
        assert not session.captions.snapshot()["events"]
        evidence["qt_clear_passed"] = True
        evidence["status"] = "passed_engineering_smoke"
    finally:
        gate[0] = True
        window.close()
        application.processEvents()
        write_json_atomic(output / "replay.json", evidence)
    return evidence


def main():
    configure_utf8_streams()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("configs/libras_caption_catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/auto-captions-smoke"))
    parser.add_argument("--pause-at", type=float, help="显式创建合成停顿：截至此源秒，随后保持末帧；不是自然长流验收")
    parser.add_argument("--hold-seconds", type=float, default=1.2)
    args = parser.parse_args()
    video = controlled_pause(args.video, args.output, args.pause_at, args.hold_seconds) if args.pause_at else args.video
    result = validate(args.model, video, args.catalog, args.output)
    print(json.dumps({"status": result["status"], "segments": result["ready_count"],
                      "http_results": len(result["http_responses"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
