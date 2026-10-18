"""Wake word: the session logic and WS /ws/wake with a fake detector, plus the real model when it is present."""
import struct

import pytest

from app import wake

FRAME = bytes(wake.FRAME_BYTES)          # 80 ms of silence


class FakeEngine:
    """Scores frames from a script (default 0.0) and counts resets."""

    def __init__(self, scores=()):
        self.scores, self.frames, self.resets = list(scores), 0, 0

    def score(self, frame):
        assert len(frame) == wake.FRAME_BYTES
        self.frames += 1
        return self.scores.pop(0) if self.scores else 0.0

    def reset(self):
        self.resets += 1


# ---- session -----------------------------------------------------------------------------

def test_session_scores_whole_frames_and_keeps_the_remainder():
    eng = FakeEngine()
    s = wake.WakeSession(eng, 0.6)
    assert s.process(FRAME[:1000], 0) is None and eng.frames == 0       # not a whole frame yet
    assert s.process(FRAME[:1000], 0) is None and eng.frames == 0       # 2000 bytes < 2560
    assert s.process(FRAME[:1000], 0) is None and eng.frames == 1       # 3000 bytes -> one frame, 440 left over
    assert s.process(FRAME * 2, 0) is None and eng.frames == 3          # 440 + 5120 = two more frames


def test_wakes_at_or_above_the_threshold_only():
    s = wake.WakeSession(FakeEngine([0.59, 0.6]), 0.6)
    assert s.process(FRAME, 0) is None
    assert s.process(FRAME, 1) == {"type": "wake", "score": 0.6}


def test_one_phrase_wakes_once_during_the_cooldown_then_again_after_it():
    eng = FakeEngine([0.9, 0.95, 0.99, 0.9])
    s = wake.WakeSession(eng, 0.6, cooldown_s=3.0)
    assert s.process(FRAME, 10.0)["type"] == "wake"
    assert eng.resets == 1                                  # detector state cleared after a wake
    assert s.process(FRAME, 10.1) is None                   # still the same phrase
    assert s.process(FRAME, 12.9) is None
    assert s.process(FRAME, 13.0)["type"] == "wake"         # cooldown over


def test_the_best_score_in_a_chunk_counts():
    s = wake.WakeSession(FakeEngine([0.1, 0.8, 0.2]), 0.6)
    assert s.process(FRAME * 3, 0)["score"] == 0.8


def test_reset_drops_buffered_audio_and_resets_the_detector():
    eng = FakeEngine()
    s = wake.WakeSession(eng, 0.6)
    s.process(FRAME[:1000], 0)
    s.reset()
    assert eng.resets == 1
    s.process(FRAME[:1560], 0)                              # would have completed a frame without the reset
    assert eng.frames == 0


def test_odd_byte_chunks_are_rejected():
    with pytest.raises(ValueError):
        wake.WakeSession(FakeEngine(), 0.6).process(b"\x00\x01\x02", 0)


# ---- settings ----------------------------------------------------------------------------

def test_off_by_default():
    ok, reason = wake.availability()
    assert not ok and "WAKE_ENABLED=true" in reason


def test_threshold_setting(monkeypatch):
    assert wake.threshold() == 0.6
    for value, expected in (("0.8", 0.8), ("", 0.6), ("abc", 0.6), ("0", 0.6), ("1.5", 0.6), ("1", 1.0)):
        monkeypatch.setenv("WAKE_THRESHOLD", value)
        assert wake.threshold() == expected


def test_missing_models_say_how_to_get_them(monkeypatch, tmp_path):
    import sys
    import types
    # pretend the optional packages are installed, so this test does not depend on the machine
    monkeypatch.setitem(sys.modules, "openwakeword", types.ModuleType("openwakeword"))
    monkeypatch.setenv("WAKE_ENABLED", "true")
    monkeypatch.setenv("WAKE_MODEL_DIR", str(tmp_path))
    ok, reason = wake.availability()
    assert not ok and "setup_wake" in reason and wake.KEYWORD_FILE in reason


def test_missing_packages_say_how_to_install_them(monkeypatch, tmp_path):
    import sys
    monkeypatch.setitem(sys.modules, "openwakeword", None)          # makes `import openwakeword` fail
    monkeypatch.setenv("WAKE_ENABLED", "true")
    monkeypatch.setenv("WAKE_MODEL_DIR", str(tmp_path))
    ok, reason = wake.availability()
    assert not ok and "requirements-wake.txt" in reason
    with pytest.raises(wake.WakeUnavailable, match="requirements-wake.txt"):
        wake.new_session()


def test_a_broken_model_file_is_unavailable_not_a_crash(monkeypatch, tmp_path):
    pytest.importorskip("openwakeword")
    monkeypatch.setenv("WAKE_ENABLED", "true")
    monkeypatch.setenv("WAKE_MODEL_DIR", str(tmp_path))
    for name in (wake.KEYWORD_FILE, *wake.SUPPORT_FILES):
        (tmp_path / name).write_bytes(b"not an onnx model")
    assert wake.availability()[0] is True
    with pytest.raises(wake.WakeUnavailable, match="Could not load the wake model"):
        wake.new_session()


# ---- WebSocket ---------------------------------------------------------------------------

@pytest.fixture()
def fake_wake(monkeypatch):
    """Make /ws/wake use a scripted engine; returns the engine."""
    engine = FakeEngine()
    monkeypatch.setattr(wake, "new_session", lambda: wake.WakeSession(engine, 0.6, cooldown_s=0.0))
    return engine


def test_ws_ready_then_wake_event(client, fake_wake):
    fake_wake.scores = [0.0, 0.97]
    with client.websocket_connect("/ws/wake") as ws:
        assert ws.receive_json() == {"type": "ready"}
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        assert ws.receive_json() == {"type": "wake", "score": 0.97}


def test_ws_reset_message_clears_the_detector(client, fake_wake):
    with client.websocket_connect("/ws/wake") as ws:
        ws.receive_json()
        ws.send_text("reset")
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        fake_wake.scores = [0.9]
        ws.send_bytes(FRAME)
        assert ws.receive_json()["type"] == "wake"
    assert fake_wake.resets >= 2                            # the explicit reset, and the one after the wake


def test_ws_bad_audio_is_rejected_and_closed(client, fake_wake):
    for bad in (b"\x00" * 3, b"\x00" * (wake.MAX_CHUNK_BYTES + 2)):
        with client.websocket_connect("/ws/wake") as ws:
            ws.receive_json()
            ws.send_bytes(bad)
            assert ws.receive_json()["type"] == "error"


def test_ws_unavailable_explains_itself(client):
    with client.websocket_connect("/ws/wake") as ws:        # WAKE_ENABLED is off in tests
        msg = ws.receive_json()
        assert msg["type"] == "error" and "WAKE_ENABLED=true" in msg["reason"]


def test_ws_limits_the_number_of_listeners(client, fake_wake):
    assert wake.acquire_slot() and wake.acquire_slot()
    try:
        with client.websocket_connect("/ws/wake") as ws:
            assert "Too many" in ws.receive_json()["reason"]
    finally:
        wake.release_slot()
        wake.release_slot()
    with client.websocket_connect("/ws/wake") as ws:        # slots are free again
        assert ws.receive_json() == {"type": "ready"}


def test_slot_is_released_when_the_phone_disconnects(client, fake_wake):
    for _ in range(wake.MAX_SESSIONS + 2):
        with client.websocket_connect("/ws/wake") as ws:
            assert ws.receive_json() == {"type": "ready"}


def test_health_reports_whether_wake_can_run(client, monkeypatch):
    assert client.get("/api/health").json()["wake"] is False
    monkeypatch.setattr(wake, "availability", lambda: (True, ""))
    assert client.get("/api/health").json()["wake"] is True


def test_no_audio_or_scores_in_the_logs(client, fake_wake, caplog):
    import logging
    fake_wake.scores = [0.9]
    with caplog.at_level(logging.DEBUG):
        with client.websocket_connect("/ws/wake") as ws:
            ws.receive_json()
            ws.send_bytes(struct.pack("<1280h", *range(1280)))
            ws.receive_json()
    assert "0.9" not in caplog.text and "wake" not in caplog.text.lower().replace("sightline.wake", "")


# ---- the real model (skipped unless the packages and model files are present) -------------

def test_real_model_loads_and_silence_never_wakes(monkeypatch):
    pytest.importorskip("openwakeword")
    monkeypatch.setenv("WAKE_ENABLED", "true")
    ok, reason = wake.availability()
    if not ok:
        pytest.skip(reason)
    session = wake.new_session()
    for i in range(60):                                      # about 5 seconds of silence
        assert session.process(FRAME, float(i)) is None
