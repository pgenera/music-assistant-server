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
    """Async client for the Roku SoundBridge RCP protocol (TCP port 5555)."""

    def __init__(self, host: str, port: int, on_state_change: Callable[[], None]) -> None:
        """
        Initialize the RCP client.

        :param host: IP address or hostname of the SoundBridge.
        :param port: TCP port for the RCP control connection (default 5555).
        :param on_state_change: Callback invoked whenever parsed device state changes.
        """
        self.host = host
        self.port = port
        self._on_state_change = on_state_change

        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._connected: bool = False
        self._closing: bool = False
        self._connect_lock = asyncio.Lock()
        self._lock = asyncio.Lock()
        self._pending_responses: dict[str, deque[asyncio.Future[Any]]] = {}

        self._read_task: asyncio.Task[None] | None = None
        self._poll_task: asyncio.Task[None] | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()

        # Device state
        self.power_state: str = "on"
        self.transport_state: str = "stop"
        self.volume: int = 0
        self.muted: bool = False
        self.title: str = ""
        self.artist: str = ""
        self.album: str = ""
        self.genre: str = ""
        self.url: str = ""
        self.duration: int = 0
        self.position: int = 0
        self.position_updated_at: float = 0.0
        self.mac_address: str = ""
        self.version: str = ""

        self._list_future: asyncio.Future[list[str]] | None = None
        self._current_list: list[str] = []

    @property
    def is_connected(self) -> bool:
        """Return True if connected to the SoundBridge."""
        return self._connected

    async def connect(self) -> bool:
        """Connect to the SoundBridge and start background read/poll tasks."""
        async with self._connect_lock:
            if self._connected:
                return True
            try:
                _LOGGER.debug("Connecting to %s:%d", self.host, self.port)
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(self.host, self.port), timeout=5.0
                )
                self._reader = reader
                self._writer = writer
                self._connected = True
                self._closing = False

                line = b""
                for _ in range(3):
                    line = await asyncio.wait_for(reader.readline(), timeout=5.0)
                    if line.strip():
                        break
                if not line.startswith(b"roku: ready"):
                    _LOGGER.debug("Invalid RCP banner from %s:%d: %r", self.host, self.port, line)
                    self._handle_disconnect()
                    return False

                if not self._read_task or self._read_task.done():
                    self._read_task = asyncio.create_task(self._read_loop())
                if not self._poll_task or self._poll_task.done():
                    self._poll_task = asyncio.create_task(self._poll_loop())
            except (TimeoutError, OSError) as err:
                _LOGGER.debug("Failed to connect to %s:%d: %s", self.host, self.port, err)
                self._handle_disconnect()
                return False
            else:
                return True

    async def disconnect(self) -> None:
        """Disconnect from the SoundBridge and cancel background tasks."""
        self._closing = True
        self._connected = False
        self._clear_pending_responses()

        for task in (
            self._read_task,
            self._poll_task,
            self._reconnect_task,
            *self._background_tasks,
        ):
            if task and not task.done():
                task.cancel()

        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
        self._reader = None

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
        length_ms: int | None = None,
    ) -> None:
        """
        Push a stream URL to the device and start playing.

        Uses the documented working-song flow rather than the undocumented
        PlayStation command. Setting ``format`` explicitly avoids the
        SoundBridge's content-type probing, which is unreliable for
        chunked-encoded WAV with no Content-Length. ``remoteStream=1`` marks
        the URL as an endless stream so the device does not loop on EOF.

        All commands are pipelined into a single TCP send so the device can
        process the whole sequence without per-command round-trips.

        :param url: Stream URL to play.
        :param title: Track title to display (when not relying on in-stream ICY).
        :param artist: Artist name to display (when not relying on in-stream ICY).
        :param fmt: Audio format token (e.g. ``MP3``, ``WAV``, ``AAC``, ``AIFF``).
        :param length_ms: Track length in milliseconds, if known.
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
        if length_ms is not None and length_ms > 0:
            commands.append(f"SetWorkingSongInfo trackLength {length_ms}")
        commands.append("QueueAndPlayOne working")
        await self._send_pipeline(commands)
        self.transport_state = "play"

    async def set_volume(self, volume: int) -> None:
        """
        Set volume level (0-100).

        :param volume: Target volume percentage from 0 to 100.
        """
        await self._send_command(f"SetVolume {volume}")
        self.volume = volume

    async def set_mute(self, *, muted: bool) -> None:
        """
        Mute or unmute audio output.

        :param muted: True to mute, False to unmute.
        """
        await self._send_command(f"Mute {'on' if muted else 'off'}")
        self.muted = muted

    async def turn_on(self) -> None:
        """Wake the device from standby."""
        await self._send_command("SetPowerState on yes")
        self.power_state = "on"

    async def wait_for_power_on(self, timeout: float = 5.0) -> bool:
        """
        Poll until the device confirms it is awake from standby.

        :param timeout: Maximum seconds to wait.
        :return: True if powered on within timeout.
        """
        deadline = asyncio.get_running_loop().time() + timeout
        while asyncio.get_running_loop().time() < deadline:
            result = await self._send_command(
                "GetPowerState", wait_for_response=True, disconnect_on_error=False
            )
            if isinstance(result, str) and result.lower() != "standby":
                return True
            await asyncio.sleep(0.1)
        return False

    async def turn_off(self) -> None:
        """Put the device into standby."""
        await self._send_command("SetPowerState standby")
        self.power_state = "standby"

    async def set_working_song_info(self, field: str, value: str) -> None:
        """
        Set a working song info field shown on the device VFD display.

        :param field: RCP working song field name (e.g. ``title``, ``artist``, ``trackLength``).
        :param value: Field value string.
        """
        await self._send_command(f"SetWorkingSongInfo {field} {value}")

    def _spawn_background_task(self, coro: Any) -> None:
        """Create and track a short-lived background task."""
        task: asyncio.Task[None] = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    def _handle_disconnect(self) -> None:
        """Handle an unexpected disconnection."""
        if self._closing:
            return
        was_connected = self._connected
        self._connected = False
        self._clear_pending_responses()
        self._spawn_background_task(self._cleanup_connection(was_connected=was_connected))
        self._ensure_reconnect()

    async def _cleanup_connection(self, *, was_connected: bool) -> None:
        """Clean up tasks and stream writer after disconnect."""
        current = asyncio.current_task()
        for task in (self._read_task, self._poll_task):
            if task and not task.done() and task is not current:
                task.cancel()
        if self._writer:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
            self._writer = None
        self._reader = None
        if was_connected:
            _LOGGER.info("Disconnected from %s:%d", self.host, self.port)
            self._on_state_change()

    def _ensure_reconnect(self) -> None:
        """Schedule reconnect loop if not already running."""
        if self._closing or (self._reconnect_task and not self._reconnect_task.done()):
            return
        self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self) -> None:
        """Reconnect with exponential backoff."""
        delay = 5
        while not self._connected and not self._closing:
            await asyncio.sleep(delay)
            if await self.connect():
                _LOGGER.info("Reconnected to %s:%d", self.host, self.port)
                self._on_state_change()
                break
            delay = min(delay * 2, 60)

    def _clear_pending_responses(self) -> None:
        """Resolve all pending response futures with None."""
        for deq in self._pending_responses.values():
            while deq:
                future = deq.popleft()
                if not future.done():
                    future.set_result(None)
        self._pending_responses.clear()
        if self._list_future and not self._list_future.done():
            self._list_future.set_result([])
        self._list_future = None

    def _discard_future(self, command_name: str, future: asyncio.Future[Any] | None) -> None:
        """Remove a timed-out or failed future from the pending queue."""
        if future is None:
            return
        deq = self._pending_responses.get(command_name)
        if deq:
            with contextlib.suppress(ValueError):
                deq.remove(future)

    async def _send_command(
        self,
        command: str,
        *,
        wait_for_response: bool = False,
        disconnect_on_error: bool = True,
    ) -> Any | None:
        """Send a single RCP command, optionally awaiting its response line."""
        if not self._connected and not self._closing:
            await self.connect()
        if not self._connected:
            return None

        async with self._lock:
            writer = self._writer
            if writer is None:
                return None
            command_name = command.split(None, 1)[0].lower()
            future: asyncio.Future[Any] | None = None
            if wait_for_response:
                future = asyncio.get_running_loop().create_future()
                self._pending_responses.setdefault(command_name, deque()).append(future)

            try:
                _LOGGER.debug("TX: %s", command)
                writer.write(f"{command}\r\n".encode())
                await writer.drain()
                return await asyncio.wait_for(future, timeout=5.0) if future else "SENT"
            except (TimeoutError, OSError, asyncio.CancelledError) as err:
                _LOGGER.debug("Command '%s' failed: %s", command, err)
                self._discard_future(command_name, future)
                if disconnect_on_error or isinstance(err, OSError):
                    self._handle_disconnect()
                if isinstance(err, asyncio.CancelledError) and self._closing:
                    raise
                return None

    async def _send_pipeline(self, commands: list[str]) -> None:
        """Send several RCP commands in a single TCP write."""
        if not self._connected and not self._closing:
            await self.connect()
        if not self._connected:
            return
        payload = "".join(f"{cmd}\r\n" for cmd in commands).encode()
        async with self._lock:
            writer = self._writer
            if writer is None:
                return
            try:
                for cmd in commands:
                    _LOGGER.debug("TX: %s", cmd)
                writer.write(payload)
                await writer.drain()
            except (TimeoutError, OSError, asyncio.CancelledError) as err:
                _LOGGER.debug("Pipeline send failed: %s", err)
                self._handle_disconnect()
                if isinstance(err, asyncio.CancelledError) and self._closing:
                    raise

    async def _read_loop(self) -> None:
        """Read lines from the RCP socket and dispatch to the state parser."""
        reader = self._reader
        if reader is None:
            return
        try:
            while self._connected:
                line_bytes = await reader.readline()
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
        """Periodically poll the SoundBridge for power, transport, volume, and song state."""
        try:
            consecutive_timeouts = 0
            while self._connected:
                result = await self._send_command(
                    "GetPowerState", wait_for_response=True, disconnect_on_error=False
                )
                if result is None:
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 3:
                        _LOGGER.warning(
                            "Poll timeout on %s:%d; disconnecting", self.host, self.port
                        )
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
                            cmd, wait_for_response=False, disconnect_on_error=False
                        )
                        await asyncio.sleep(0.05)

                await asyncio.sleep(5)
        except asyncio.CancelledError:
            pass

    def _parse_line(self, line: str) -> None:
        """Parse a single line received from the SoundBridge."""
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
        """Resolve a pending command future if one is waiting."""
        deq = self._pending_responses.get(command_key)
        if not deq:
            return
        future = deq[0]
        should_resolve = True
        if command_key == "getcurrentsonginfo":
            should_resolve = value.lower() in {"genericerror", "error", "invalidcommand", "ok"}
        if should_resolve:
            deq.popleft()
            if not future.done():
                future.set_result(value)

    def _update_state(self, command_key: str, value: str) -> None:
        """Update internal device attributes from a parsed response line."""
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

    @staticmethod
    def _parse_time(time_str: str) -> int:
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
        """Parse a GetCurrentSongInfo key:value line."""
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
