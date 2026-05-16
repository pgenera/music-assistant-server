"""Roku SoundBridge RCP protocol client."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

_LOGGER = logging.getLogger(__name__)

RCP_PORT = 5555


class RcpClient:
    """Async client for the Roku SoundBridge RCP protocol (port 5555)."""

    def __init__(self, host: str, port: int, on_state_change: Callable[[], None]) -> None:
        """Initialize the client."""
        self.host = host
        self.port = port
        self._on_state_change = on_state_change

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected = False
        self._connecting = False
        self._closing = False
        self._lock = asyncio.Lock()
        self._pending_responses: dict[str, deque[asyncio.Future]] = {}

        self._read_task: asyncio.Task | None = None
        self._poll_task: asyncio.Task | None = None
        self._reconnect_task: asyncio.Task | None = None

        # Device state
        self.power_state = "on"
        self.transport_state = "stop"
        self.volume = 0
        self.muted = False
        self.title = ""
        self.artist = ""
        self.album = ""
        self.genre = ""
        self.url = ""
        self.duration = 0
        self.position = 0
        self.position_updated_at = 0.0
        self.mac_address = ""
        self.version = ""

        self._list_future: asyncio.Future | None = None
        self._current_list: list[str] = []

    @property
    def is_connected(self) -> bool:
        """Return True if connected."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to the SoundBridge."""
        if self._connected:
            return True
        if self._connecting:
            for _ in range(50):
                if self._connected:
                    return True
                if not self._connecting:
                    break
                await asyncio.sleep(0.1)
            if self._connected:
                return True

        self._connecting = True
        try:
            _LOGGER.debug("Connecting to %s:%d", self.host, self.port)
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.host, self.port), timeout=5.0
            )
            self._connected = True
            self._closing = False

            line = b""
            for _ in range(3):
                line = await asyncio.wait_for(self._reader.readline(), timeout=5.0)
                if line.strip():
                    break
            if not line.startswith(b"roku: ready"):
                raise ConnectionError(f"Invalid RCP banner: {line!r}")

            if not self._read_task or self._read_task.done():
                self._read_task = asyncio.create_task(self._read_loop())
            if not self._poll_task or self._poll_task.done():
                self._poll_task = asyncio.create_task(self._poll_loop())
            return True
        except (TimeoutError, ConnectionRefusedError, OSError) as err:
            _LOGGER.debug("Failed to connect to %s:%d: %s", self.host, self.port, err)
            self._handle_disconnect()
            return False
        finally:
            self._connecting = False

    async def disconnect(self) -> None:
        """Disconnect from the SoundBridge."""
        self._closing = True
        self._connected = False

        for task in (self._read_task, self._poll_task, self._reconnect_task):
            if task:
                task.cancel()

        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None

    def _handle_disconnect(self) -> None:
        """Handle an unexpected disconnection."""
        if self._closing:
            return
        was_connected = self._connected
        self._connected = False
        self._clear_pending_responses()
        asyncio.create_task(self._cleanup_connection(was_connected))
        self._ensure_reconnect()

    async def _cleanup_connection(self, was_connected: bool) -> None:
        """Clean up tasks after disconnect."""
        current = asyncio.current_task()
        for task in (self._read_task, self._poll_task):
            if task and not task.done() and task is not current:
                task.cancel()
        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
        if was_connected:
            _LOGGER.info("Disconnected from %s:%d", self.host, self.port)
            self._on_state_change()

    def _ensure_reconnect(self) -> None:
        """Schedule reconnect if not already running."""
        if self._closing or (self._reconnect_task and not self._reconnect_task.done()):
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Exponential backoff reconnect loop."""
        delay = 5
        while not self._connected and not self._closing:
            await asyncio.sleep(delay)
            if await self.connect():
                _LOGGER.info("Reconnected to %s:%d", self.host, self.port)
                self._on_state_change()
                break
            delay = min(delay * 2, 60)

    def _clear_pending_responses(self) -> None:
        """Cancel all pending response futures."""
        for deq in self._pending_responses.values():
            while deq:
                future = deq.popleft()
                if not future.done():
                    future.set_result(None)
        self._pending_responses.clear()

    async def _send_command(
        self,
        command: str,
        wait_for_response: bool = False,
        disconnect_on_error: bool = True,
    ) -> Any | None:
        """Send a command to the device, optionally waiting for a response line."""
        if not self._connected and not self._closing:
            await self.connect()
        if not self._connected:
            return None

        async with self._lock:
            command_name = command.split(None, 1)[0].lower()
            future = None
            if wait_for_response:
                future = asyncio.Future()
                self._pending_responses.setdefault(command_name, deque()).append(future)

            try:
                _LOGGER.debug("TX: %s", command)
                self._writer.write(f"{command}\r\n".encode())
                await self._writer.drain()
                if future:
                    return await asyncio.wait_for(future, timeout=5.0)
                return "SENT"
            except (TimeoutError, OSError, asyncio.CancelledError) as err:
                _LOGGER.debug("Command '%s' failed: %s", command, err)
                if future:
                    deq = self._pending_responses.get(command_name)
                    if deq:
                        with contextlib.suppress(ValueError):
                            deq.remove(future)
                if disconnect_on_error or isinstance(err, OSError):
                    self._handle_disconnect()
                if isinstance(err, asyncio.CancelledError) and self._closing:
                    raise
                return None

    async def _read_loop(self) -> None:
        """Read lines from the device and dispatch to state parser."""
        try:
            while self._connected:
                line_bytes = await self._reader.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").strip()
                if line:
                    self._parse_line(line)
        except asyncio.CancelledError:
            pass
        except OSError as err:
            _LOGGER.debug("Read loop error: %s", err)
        finally:
            self._handle_disconnect()

    async def _poll_loop(self) -> None:
        """Periodically poll for device state."""
        try:
            consecutive_timeouts = 0
            while self._connected:
                result = await self._send_command(
                    "GetPowerState", wait_for_response=True, disconnect_on_error=False
                )
                if result is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        _LOGGER.warning("Poll timeout; disconnecting")
                        self._handle_disconnect()
                        break
                else:
                    consecutive_timeouts = 0

                if self.power_state != "standby":
                    for cmd in (
                        "GetMACAddress",
                        "GetTransportState",
                        "GetVolume",
                        "GetCurrentSongInfo",
                        "GetElapsedTime",
                        "GetTotalTime",
                    ):
                        await self._send_command(
                            cmd, wait_for_response=True, disconnect_on_error=False
                        )

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass
        except (TimeoutError, OSError) as err:
            _LOGGER.debug("Poll loop error: %s", err)

    def _parse_line(self, line: str) -> None:
        """Parse a response line and update state."""
        _LOGGER.debug("RX: %s", line)

        if ":" not in line:
            if self._list_future and not self._list_future.done():
                self._current_list.append(line)
            return

        command_key, value = line.split(":", 1)
        command_key = command_key.strip().lower()
        value = value.strip()

        if command_key.endswith("listresultsize"):
            self._current_list = []
            return
        if command_key.endswith("listresultend"):
            if self._list_future and not self._list_future.done():
                self._list_future.set_result(self._current_list)
                self._list_future = None
            return

        self._resolve_future(command_key, value)
        self._update_state(command_key, value)
        self._on_state_change()

    def _resolve_future(self, command_key: str, value: str) -> None:
        """Resolve a pending future if one is waiting for this command."""
        if command_key not in self._pending_responses:
            return
        deq = self._pending_responses[command_key]
        if not deq:
            return
        future = deq[0]
        should_resolve = True
        if command_key == "getcurrentsonginfo":
            is_terminal = value.lower() in {"genericerror", "error", "invalidcommand", "ok"}
            should_resolve = is_terminal
        if should_resolve:
            deq.popleft()
            if not future.done():
                future.set_result(value)

    def _update_state(self, command_key: str, value: str) -> None:
        """Update internal state from a parsed response line."""
        if command_key == "getpowerstate":
            self.power_state = value.lower()
        elif command_key == "gettransportstate":
            self.transport_state = value.lower()
        elif command_key == "getvolume":
            with contextlib.suppress(ValueError):
                self.volume = int(value)
        elif command_key == "getelapsedtime":
            self.position = self._parse_time(value)
            self.position_updated_at = time.time()
        elif command_key == "gettotaltime":
            self.duration = self._parse_time(value)
        elif command_key == "getmacaddress":
            self.mac_address = value
        elif command_key == "getversion":
            self.version = value
        elif command_key == "getcurrentsonginfo":
            self._parse_song_info(value)

    def _parse_time(self, time_str: str) -> int:
        """Parse H:MM:SS or MM:SS into seconds."""
        parts = time_str.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            if len(parts) == 1:
                return int(parts[0])
        except ValueError:
            pass
        return 0

    def _parse_song_info(self, value: str) -> None:
        """Parse a GetCurrentSongInfo response line."""
        if ":" not in value:
            return
        info_key, info_val = value.split(":", 1)
        key = info_key.strip().lower()
        val = info_val.strip()
        if key == "title":
            self.title = val
        elif key == "artist":
            self.artist = val
        elif key == "album":
            self.album = val
        elif key == "genre":
            self.genre = val
        elif key in {"resource[0] url", "playlisturl"}:
            self.url = val

    # --- Control commands ---

    async def play(self) -> None:
        """Resume playback."""
        await self._send_command("Play")
        self.transport_state = "play"

    async def pause(self) -> None:
        """Pause playback."""
        await self._send_command("Pause")
        self.transport_state = "pause"

    async def stop(self) -> None:
        """Stop playback."""
        await self._send_command("Stop")
        self.transport_state = "stop"

    async def play_url(
        self,
        url: str,
        title: str = "",
        artist: str = "",
        fmt: str = "WAV",
    ) -> None:
        """Push a stream URL to the device and start playing.

        Uses the documented working-song flow rather than the undocumented
        PlayStation command. Setting `format` explicitly avoids the
        SoundBridge's content-type probing, which is unreliable for
        chunked-encoded WAV with no Content-Length. remoteStream=1 marks the
        URL as an endless stream so the device does not loop on EOF.

        All commands are pipelined into a single TCP send so the device can
        process the whole sequence without per-command round-trips.
        """
        commands = [
            "ClearWorkingSong",
            f"SetWorkingSongInfo url {url}",
            f"SetWorkingSongInfo format {fmt}",
            "SetWorkingSongInfo remoteStream 1",
        ]
        if title:
            commands.append(f"SetWorkingSongInfo title {title}")
        if artist:
            commands.append(f"SetWorkingSongInfo artist {artist}")
        commands.append("QueueAndPlayOne working")
        await self._send_pipeline(commands)
        self.transport_state = "play"

    async def _send_pipeline(self, commands: list[str]) -> None:
        """Send several commands as a single TCP write (no per-command waits)."""
        if not self._connected and not self._closing:
            await self.connect()
        if not self._connected:
            return
        payload = "".join(f"{c}\r\n" for c in commands).encode()
        async with self._lock:
            try:
                for c in commands:
                    _LOGGER.debug("TX: %s", c)
                self._writer.write(payload)
                await self._writer.drain()
            except (TimeoutError, OSError, asyncio.CancelledError) as err:
                _LOGGER.debug("Pipeline send failed: %s", err)
                self._handle_disconnect()
                if isinstance(err, asyncio.CancelledError) and self._closing:
                    raise

    async def set_volume(self, volume: int) -> None:
        """Set volume level (0–100)."""
        await self._send_command(f"SetVolume {volume}")
        self.volume = volume

    async def set_mute(self, muted: bool) -> None:
        """Mute or unmute."""
        await self._send_command(f"Mute {'on' if muted else 'off'}")
        self.muted = muted

    async def turn_on(self) -> None:
        """Wake the device from standby."""
        await self._send_command("PlayPreset 0")
        self.power_state = "on"

    async def wait_for_power_on(self, timeout: float = 5.0) -> bool:
        """Poll until the device confirms it is on.

        :param timeout: Maximum seconds to wait.
        :returns: True if on within timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            result = await self._send_command(
                "GetPowerState", wait_for_response=True, disconnect_on_error=False
            )
            if result and result.lower() != "standby":
                return True
            await asyncio.sleep(0.1)
        return False

    async def turn_off(self) -> None:
        """Put the device into standby."""
        await self._send_command("SetPowerState standby")
        self.power_state = "standby"

    async def set_working_song_info(self, field: str, value: str) -> None:
        """Set a working song info field shown on the device display."""
        await self._send_command(f"SetWorkingSongInfo {field} {value}")
