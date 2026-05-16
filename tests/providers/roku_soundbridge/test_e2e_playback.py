"""End-to-end test: play a real track on a real SoundBridge through MA.

This drives Music Assistant's WebSocket API and queries the SoundBridge over
RCP. It verifies the full provider path: MA queue -> provider.play_media ->
RCP working-song flow -> SoundBridge HTTP-fetches the MA flow stream ->
audio actually playing on the device, with ICY metadata visible.

Required env vars:
  MA_TOKEN       - long-lived MA API token
  MA_PLAYER_ID   - SoundBridge player id, e.g. soundbridge_roku_fivesevenfive_org
  MA_HOST        - MA server host (default: localhost)
  MA_PORT        - MA WS port (default: 8095)
  RCP_HOST       - SoundBridge hostname/IP (default: derived from player id)

Run with:
  MA_TOKEN=... pytest tests/test_e2e_playback.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import pytest
import websockets

MA_TOKEN = os.environ.get("MA_TOKEN", "")
MA_HOST = os.environ.get("MA_HOST", "localhost")
MA_PORT = int(os.environ.get("MA_PORT", "8095"))
PLAYER_ID = os.environ.get("MA_PLAYER_ID", "soundbridge_roku_fivesevenfive_org")
RCP_HOST = os.environ.get("RCP_HOST") or PLAYER_ID.removeprefix("soundbridge_").replace("_", ".")
RCP_PORT = 5555

# Click-to-audio budget. With pipelined RCP commands and MP3+ICY mode, an MA
# library track plays in ~3.1s on local hardware. Allow some slack for slower
# hosts; if this regresses much higher we want to know.
AUDIO_BUDGET_S = 6.0


pytestmark = pytest.mark.skipif(not MA_TOKEN, reason="MA_TOKEN not set")


class WSClient:
    """Tiny MA WebSocket client just for what this test needs."""

    def __init__(self, ws):
        self._ws = ws
        self._n = 0

    async def call(self, command: str, args: dict | None = None) -> dict:
        self._n += 1
        msg_id = f"t{self._n}"
        await self._ws.send(json.dumps({"message_id": msg_id, "command": command, "args": args or {}}))
        while True:
            msg = json.loads(await asyncio.wait_for(self._ws.recv(), timeout=15))
            if msg.get("message_id") == msg_id:
                return msg


async def _rcp_query(reader, writer, command: str) -> str:
    writer.write(f"{command}\r\n".encode())
    await writer.drain()
    return (await asyncio.wait_for(reader.readline(), timeout=2)).decode().strip()


async def _await_stop(reader, writer) -> None:
    await _rcp_query(reader, writer, "Stop")
    for _ in range(30):
        if "Stop" in await _rcp_query(reader, writer, "GetTransportState"):
            return
        await asyncio.sleep(0.1)
    raise AssertionError("device did not reach Stop within 3s")


async def _poll_song_info_for(reader, writer, needle: str, timeout: float) -> str:
    """Poll GetCurrentSongInfo until the response contains `needle` or timeout.

    The SoundBridge updates CurrentSongInfo when ICY metadata is received and
    consumed by the decoder; that lag is non-deterministic so we poll.
    """
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


async def _wait_for_audio(reader, writer, t0: float, timeout: float) -> float:
    """Return wall-clock seconds until elapsed counter advances past 0."""
    seen_reset = False
    while time.monotonic() - t0 < timeout:
        e = await _rcp_query(reader, writer, "GetElapsedTime")
        if "0:00:00" in e:
            seen_reset = True
        elif seen_reset:
            return time.monotonic() - t0
        await asyncio.sleep(0.05)
    raise AssertionError(f"elapsed never advanced past 0 within {timeout}s")


async def _connect_ws() -> WSClient:
    ws = await websockets.connect(f"ws://{MA_HOST}:{MA_PORT}/ws", open_timeout=5)
    await asyncio.wait_for(ws.recv(), timeout=3)  # server hello
    client = WSClient(ws)
    auth = await client.call("auth", {"token": MA_TOKEN})
    if auth.get("error_code"):
        raise AssertionError(f"auth failed: {auth}")
    return client


@pytest.mark.asyncio
async def test_player_registered() -> None:
    """The SoundBridge player should be registered and available."""
    client = await _connect_ws()
    r = await client.call("players/get", {"player_id": PLAYER_ID})
    assert r.get("error_code") is None, r
    player = r["result"]
    assert player["available"], f"player not available: {player}"


@pytest.mark.asyncio
async def test_e2e_play_a_track() -> None:
    """Play a real recently-added track and verify it actually plays + shows metadata."""
    client = await _connect_ws()

    # Find a track to play
    r = await client.call("music/recently_added_tracks", {"limit": 1})
    tracks = r.get("result", [])
    assert tracks, "no recently-added tracks available to test with"
    track = tracks[0]
    expected_title = track["name"]
    expected_artist = (track.get("artists") or [{}])[0].get("name", "")

    # Open RCP socket and force device to a known stopped state
    rcp_r, rcp_w = await asyncio.open_connection(RCP_HOST, RCP_PORT)
    try:
        await asyncio.wait_for(rcp_r.readline(), timeout=2)  # banner
        await _await_stop(rcp_r, rcp_w)

        # Trigger play and concurrently watch for audio start
        t0 = time.monotonic()
        watcher = asyncio.create_task(_wait_for_audio(rcp_r, rcp_w, t0, AUDIO_BUDGET_S + 2))
        play = await client.call(
            "player_queues/play_media",
            {"queue_id": PLAYER_ID, "media": track["uri"], "option": "play"},
        )
        assert play.get("error_code") is None, play
        audio_at = await watcher

        assert audio_at <= AUDIO_BUDGET_S, (
            f"audio took {audio_at:.2f}s, expected <= {AUDIO_BUDGET_S}s"
        )

        # Verify the device shows the track via ICY metadata.
        # ICY blocks arrive every 16384 bytes (~700ms at typical MP3 bitrates),
        # but the device may take additional time to surface them in
        # GetCurrentSongInfo, so poll for several seconds.
        info = await _poll_song_info_for(
            rcp_r, rcp_w, expected_title, timeout=8.0
        )
        assert "format: MP3" in info, f"expected MP3 mode, got:\n{info}"
        assert "status: playable" in info, f"device not playable:\n{info}"
        assert expected_title in info, (
            f"expected track title {expected_title!r} in song info:\n{info}"
        )
        if expected_artist:
            assert expected_artist in info, (
                f"expected artist {expected_artist!r} in song info:\n{info}"
            )
    finally:
        # Leave the device in Stop so subsequent runs start clean
        try:
            rcp_w.write(b"Stop\r\n")
            await rcp_w.drain()
        finally:
            rcp_w.close()
