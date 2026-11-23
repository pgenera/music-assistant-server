"""Unit tests for the Roku SoundBridge RCP TCP client."""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING

import pytest

from music_assistant.providers.roku_soundbridge.rcp_client import RcpClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


class FakeRcpServer:
    """In-process TCP server speaking the Roku SoundBridge RCP line protocol."""

    def __init__(self, banner: bytes = b"roku: ready\r\n") -> None:
        """Initialize fake server state."""
        self.banner = banner
        self.power_state = "on"
        self.transport_state = "stop"
        self.volume = 35
        self.mac_address = "00-0d-4b-11-22-33"
        self.received_commands: list[str] = []
        self._server: asyncio.Server | None = None
        self._writers: set[asyncio.StreamWriter] = set()
        self.port: int = 0

    async def start(self) -> None:
        """Start listening on an ephemeral localhost port."""
        self._server = await asyncio.start_server(self._handle_client, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        """Close all active client sockets and stop listening."""
        for writer in list(self._writers):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        self._writers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve RCP commands on a client connection."""
        self._writers.add(writer)
        try:
            writer.write(self.banner)
            await writer.drain()
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                cmd = raw.decode().strip()
                if not cmd:
                    continue
                self.received_commands.append(cmd)
                await self._respond(writer, cmd)
        except asyncio.CancelledError, OSError:
            pass
        finally:
            self._writers.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def _respond(self, writer: asyncio.StreamWriter, cmd: str) -> None:
        """Send canned RCP responses matching real SoundBridge firmware."""
        lower = cmd.lower()
        canned: dict[str, list[str]] = {
            "getpowerstate": [f"GetPowerState: {self.power_state}"],
            "getmacaddress": [f"GetMACAddress: {self.mac_address}"],
            "gettransportstate": [f"GetTransportState: {self.transport_state}"],
            "getvolume": [f"GetVolume: {self.volume}"],
            "getelapsedtime": ["GetElapsedTime: 0:01:15"],
            "gettotaltime": ["GetTotalTime: 0:04:05"],
            "getcurrentsonginfo": [
                "GetCurrentSongInfo: title: Radio Song",
                "GetCurrentSongInfo: artist: Test Artist",
                "GetCurrentSongInfo: album: Test Album",
                "GetCurrentSongInfo: resource[0] url: http://127.0.0.1:8095/flow.mp3",
                "GetCurrentSongInfo: OK",
            ],
        }
        if lower in canned:
            lines = canned[lower]
        elif lower == "playpreset 0":
            self.power_state = "on"
            lines = ["PlayPreset: OK"]
        elif lower == "setpowerstate standby":
            self.power_state = "standby"
            lines = ["SetPowerState: OK"]
        elif lower.startswith("setvolume "):
            self.volume = int(cmd.split(None, 1)[1])
            lines = ["SetVolume: OK"]
        elif lower == "queueandplayone working":
            self.transport_state = "play"
            lines = ["QueueAndPlayOne: OK"]
        else:
            verb = cmd.split(None, 1)[0]
            lines = [f"{verb}: OK"]

        writer.write("".join(f"{line}\r\n" for line in lines).encode())
        await writer.drain()


@pytest.fixture
async def rcp_server() -> AsyncIterator[FakeRcpServer]:
    """Yield a started FakeRcpServer and shut it down after the test."""
    server = FakeRcpServer()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


async def test_connect_and_initial_poll_populates_state(rcp_server: FakeRcpServer) -> None:
    """Connecting to an RCP server reads the banner and populates state from the poll loop."""
    callbacks = 0

    def on_change() -> None:
        nonlocal callbacks
        callbacks += 1

    client = RcpClient("127.0.0.1", rcp_server.port, on_change)
    try:
        assert await client.connect() is True
        assert client.is_connected is True

        for _ in range(30):
            if client.duration == 245 and client.title == "Radio Song":
                break
            await asyncio.sleep(0.05)

        assert client.power_state == "on"
        assert client.mac_address == "00-0d-4b-11-22-33"
        assert client.volume == 35
        assert client.position == 75
        assert client.duration == 245
        assert client.title == "Radio Song"
        assert client.artist == "Test Artist"
        assert client.album == "Test Album"
        assert client.url == "http://127.0.0.1:8095/flow.mp3"
        assert callbacks > 0
    finally:
        await client.disconnect()


async def test_connect_rejects_invalid_banner() -> None:
    """A server that does not greet with 'roku: ready' fails connection cleanly."""
    bad_server = FakeRcpServer(banner=b"HTTP/1.1 400 Bad Request\r\n")
    await bad_server.start()
    client = RcpClient("127.0.0.1", bad_server.port, lambda: None)
    try:
        assert await client.connect() is False
        assert client.is_connected is False
    finally:
        await client.disconnect()
        await bad_server.stop()


async def test_play_url_pipelines_working_song_commands(rcp_server: FakeRcpServer) -> None:
    """play_url sends ClearWorkingSong, SetWorkingSongInfo, and QueueAndPlayOne in order."""
    client = RcpClient("127.0.0.1", rcp_server.port, lambda: None)
    try:
        assert await client.connect() is True
        rcp_server.received_commands.clear()

        await client.play_url(
            "http://192.168.1.10:8097/flow/stream.wav",
            title="Blue Train",
            artist="John Coltrane",
            fmt="WAV",
            length_ms=642000,
        )
        for _ in range(20):
            if "QueueAndPlayOne working" in rcp_server.received_commands:
                break
            await asyncio.sleep(0.05)

        assert "ClearWorkingSong" in rcp_server.received_commands
        assert (
            "SetWorkingSongInfo url http://192.168.1.10:8097/flow/stream.wav"
            in rcp_server.received_commands
        )
        assert "SetWorkingSongInfo format WAV" in rcp_server.received_commands
        assert "SetWorkingSongInfo remoteStream 1" in rcp_server.received_commands
        assert "SetWorkingSongInfo title Blue Train" in rcp_server.received_commands
        assert "SetWorkingSongInfo artist John Coltrane" in rcp_server.received_commands
        assert "SetWorkingSongInfo trackLength 642000" in rcp_server.received_commands
        assert "QueueAndPlayOne working" in rcp_server.received_commands
        assert client.transport_state == "play"
    finally:
        await client.disconnect()


async def test_standby_wake_and_volume_controls(rcp_server: FakeRcpServer) -> None:
    """Standby wake, power off, volume, and mute issue the expected RCP commands."""
    rcp_server.power_state = "standby"
    client = RcpClient("127.0.0.1", rcp_server.port, lambda: None)
    try:
        assert await client.connect() is True
        await client.turn_on()
        assert await client.wait_for_power_on(timeout=2.0) is True
        assert client.power_state == "on"

        await client.set_volume(62)
        assert client.volume == 62

        await client.set_mute(muted=True)
        assert client.muted is True

        await client.turn_off()
        assert client.power_state == "standby"

        for _ in range(20):
            if "SetPowerState standby" in rcp_server.received_commands:
                break
            await asyncio.sleep(0.05)

        assert "SetVolume 62" in rcp_server.received_commands
        assert "Mute on" in rcp_server.received_commands
        assert "SetPowerState standby" in rcp_server.received_commands
    finally:
        await client.disconnect()
