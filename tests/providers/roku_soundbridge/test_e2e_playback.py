"""
End-to-end hardware test: play a real track on a physical SoundBridge through Music Assistant.

Skipped automatically unless ``MA_TOKEN`` is set in the environment.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

import pytest
import websockets

_LOGGER = logging.getLogger(__name__)

MA_TOKEN = os.environ.get("MA_TOKEN", "")
MA_HOST = os.environ.get("MA_HOST", "localhost")
MA_PORT = int(os.environ.get("MA_PORT", "8095"))
PLAYER_ID = os.environ.get("MA_PLAYER_ID", "soundbridge_roku_fivesevenfive_org")
RCP_HOST = os.environ.get("RCP_HOST") or PLAYER_ID.removeprefix("soundbridge_").replace("_", ".")
RCP_PORT = 5555

AUDIO_BUDGET_S = 6.0

pytestmark = pytest.mark.skipif(not MA_TOKEN, reason="MA_TOKEN not set")


class WSClient:
    """Minimal Music Assistant WebSocket API client for hardware e2e testing."""

    def __init__(self, ws: Any) -> None:
        """Initialize client wrapper."""
        self._ws = ws
        self._n = 0

    async def call(self, command: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a WebSocket command and return the matching response."""
        self._n += 1
        msg_id = f"t{self._n}"
        await self._ws.send(
            json.dumps({"message_id": msg_id, "command": command, "args": args or {}})
        )
        while True:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=15)
            msg: dict[str, Any] = json.loads(raw)
            if msg.get("message_id") == msg_id:
                return msg


async def _rcp_query(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    command: str,
) -> str:
    """Send a single RCP command and read one response line."""
    writer.write(f"{command}\r\n".encode())
    await writer.drain()
    return (await asyncio.wait_for(reader.readline(), timeout=2)).decode().strip()


async def _await_stop(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> str:
    """Ensure the device is idle and return ``"standby"`` or ``"stop"``."""
    state = await _rcp_query(reader, writer, "GetTransportState")
    if "Standby" in state:
        return "standby"
    await _rcp_query(reader, writer, "Stop")
    for _ in range(30):
        state = await _rcp_query(reader, writer, "GetTransportState")
        if "Standby" in state:
            return "standby"
        if "Stop" in state:
            return "stop"
        await asyncio.sleep(0.1)
    msg = f"device did not reach Stop/Standby within 3s (last={state!r})"
    raise AssertionError(msg)


async def _poll_song_info_for(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    needle: str,
    timeout: float,
) -> str:
    """Poll GetCurrentSongInfo until response contains ``needle`` or timeout expires."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        writer.write(b"GetCurrentSongInfo\r\n")
        await writer.drain()
        lines: list[str] = []
        while True:
            line = (await asyncio.wait_for(reader.readline(), timeout=2)).decode().strip()
            lines.append(line)
            if line.endswith(": OK"):
                break
        last = "\n".join(lines)
        if needle in last:
            return last
        await asyncio.sleep(0.5)
    return last


async def _wait_for_audio(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    t0: float,
    timeout: float,
) -> float:
    """Return wall-clock seconds until GetElapsedTime advances past 0:00:00."""
    seen_reset = False
    while time.monotonic() - t0 < timeout:
        elapsed = await _rcp_query(reader, writer, "GetElapsedTime")
        val = elapsed.partition(":")[2].strip()
        if val == "0:00:00":
            seen_reset = True
        elif seen_reset and ":" in val and "error" not in val.lower():
            return time.monotonic() - t0
        await asyncio.sleep(0.05)
    msg = f"elapsed never advanced past 0 within {timeout}s"
    raise AssertionError(msg)


async def _connect_ws() -> WSClient:
    """Connect and authenticate to the local Music Assistant WebSocket server."""
    ws = await websockets.connect(f"ws://{MA_HOST}:{MA_PORT}/ws", open_timeout=5)
    await asyncio.wait_for(ws.recv(), timeout=3)
    client = WSClient(ws)
    auth = await client.call("auth", {"token": MA_TOKEN})
    if auth.get("error_code"):
        msg = f"auth failed: {auth}"
        raise AssertionError(msg)
    return client


async def test_player_registered() -> None:
    """Verify the SoundBridge player is registered and available."""
    client = await _connect_ws()
    res = await client.call("players/get", {"player_id": PLAYER_ID})
    assert res.get("error_code") is None, res
    player = res["result"]
    assert player["available"], f"player not available: {player}"


async def _pick_playable_track(client: WSClient) -> dict[str, Any]:
    """Return a playable library track, preferring local filesystem items."""
    prov_res = await client.call("providers")
    loaded_instances = {
        p["instance_id"] for p in (prov_res.get("result") or []) if p.get("available", True)
    }
    fallback: dict[str, Any] | None = None
    for command in ("music/recently_added_tracks", "music/tracks/library_items"):
        res = await client.call(command, {"limit": 25})
        for track in res.get("result") or []:
            usable = [
                m
                for m in (track.get("provider_mappings") or [])
                if m.get("available", True)
                and (not loaded_instances or m.get("provider_instance") in loaded_instances)
            ]
            if any(str(m.get("provider_domain", "")).startswith("filesystem_") for m in usable):
                return dict(track)
            if usable and fallback is None:
                fallback = dict(track)
    if fallback is not None:
        return fallback
    msg = "no playable tracks available on any loaded provider"
    raise AssertionError(msg)


async def test_e2e_play_a_track() -> None:
    """Play a real recently-added track and verify audio start and ICY metadata."""
    client = await _connect_ws()

    track = await _pick_playable_track(client)
    expected_title = track["name"]
    expected_artist = (track.get("artists") or [{}])[0].get("name", "")

    rcp_r, rcp_w = await asyncio.open_connection(RCP_HOST, RCP_PORT)
    was_standby = False
    try:
        await asyncio.wait_for(rcp_r.readline(), timeout=2)
        was_standby = (await _await_stop(rcp_r, rcp_w)) == "standby"

        t0 = time.monotonic()
        watcher = asyncio.create_task(_wait_for_audio(rcp_r, rcp_w, t0, AUDIO_BUDGET_S + 2))
        play = await client.call(
            "player_queues/play_media",
            {"queue_id": PLAYER_ID, "media": track["uri"], "option": "play"},
        )
        assert play.get("error_code") is None, play
        audio_at = await watcher
        _LOGGER.info(
            "Audio started at %.2fs (was_standby=%s, title=%r)",
            audio_at,
            was_standby,
            expected_title,
        )
        assert audio_at <= AUDIO_BUDGET_S, f"audio took {audio_at:.2f}s"

        info = await _poll_song_info_for(rcp_r, rcp_w, expected_title, timeout=8.0)
        assert "format: MP3" in info, f"expected MP3 mode, got:\n{info}"
        assert "status: playable" in info, f"device not playable:\n{info}"
        assert expected_title in info, f"expected title {expected_title!r} in:\n{info}"
        if expected_artist:
            assert expected_artist in info, f"expected artist {expected_artist!r} in:\n{info}"
    finally:
        try:
            rcp_w.write(b"Stop\r\n")
            if was_standby:
                rcp_w.write(b"SetPowerState standby\r\n")
            await rcp_w.drain()
        finally:
            rcp_w.close()
