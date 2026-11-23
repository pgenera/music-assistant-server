"""Roku SoundBridge PlayerProtocol implementation for Music Assistant."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import (
    EventType,
    IdentifierType,
    PlaybackState,
    PlayerFeature,
    PlayerType,
)
from music_assistant_models.player import DeviceInfo, PlayerMedia

from music_assistant.constants import (
    CONF_ENABLE_ICY_METADATA,
    CONF_ENTRY_ENABLE_ICY_METADATA,
    CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3,
    CONF_OUTPUT_CODEC,
)
from music_assistant.models.player_protocol import PlayerProtocol

from .rcp_client import RcpClient

if TYPE_CHECKING:
    from collections.abc import Callable

    from music_assistant_models.event import MassEvent

    from music_assistant.models.player import Player

    from .provider import SoundBridgeProvider

# The SoundBridge does not support FLAC and rejects WAV via ICY/Shoutcast,
# so new SoundBridge protocols default to MP3 with basic ICY metadata:
#  - Reliable startup (avoids probe-and-reconnect issues seen with WAV)
#  - In-stream metadata blocks parsed natively by the SoundBridge
#  - ~400ms faster click-to-audio than WAV
# Users can still switch to WAV in the player settings UI if desired.
_CONF_ENTRY_ICY_METADATA_BASIC = ConfigEntry.from_dict(
    {**CONF_ENTRY_ENABLE_ICY_METADATA.to_dict(), "default_value": "basic"}
)

_TRANSPORT_TO_PLAYBACK = {
    "play": PlaybackState.PLAYING,
    "pause": PlaybackState.PAUSED,
    "stop": PlaybackState.IDLE,
    "buffering": PlaybackState.PLAYING,
}

# Seconds to hold the PLAYING state after issuing a play command, suppressing
# transient "stop" reports from the device during its initial buffering window.
_PLAY_GRACE_SECONDS = 15


class SoundBridgePlayer(PlayerProtocol):
    """A Roku SoundBridge player controlled via the RCP protocol."""

    _attr_type = PlayerType.PLAYER

    def __init__(
        self,
        provider: SoundBridgeProvider,
        protocol_id: str,
        host: str,
        port: int,
        *,
        name: str,
        resolved_ip: str | None = None,
    ) -> None:
        """
        Initialize the SoundBridge protocol player and RCP client.

        :param provider: Owning SoundBridgeProvider instance.
        :param protocol_id: Unique protocol identifier.
        :param host: Configured or mDNS-discovered host/IP.
        :param port: TCP port for RCP (default 5555).
        :param name: Initial display name for the player.
        :param resolved_ip: Resolved IPv4 address when ``host`` is a hostname.
        """
        super().__init__(provider, protocol_id)
        self.host = host
        self.port = port
        self.resolved_ip = resolved_ip
        self._client = RcpClient(host, port, self._on_client_state_change)
        self._last_play_url_time: float = 0.0
        self._last_pushed_title: str = ""
        self._last_pushed_artist: str = ""
        self._last_pushed_duration_ms: int = 0
        self._unsub_queue_event: Callable[[], None] | None = None

        self._attr_name = name
        self._attr_available = False
        self._attr_supported_features = {
            PlayerFeature.PLAY_MEDIA,
            PlayerFeature.POWER,
            PlayerFeature.VOLUME_SET,
            PlayerFeature.VOLUME_MUTE,
            PlayerFeature.PAUSE,
        }
        self._attr_device_info = DeviceInfo(
            model="SoundBridge",
            manufacturer="Roku",
        )
        self._attr_device_info.add_identifier(IdentifierType.IP_ADDRESS, resolved_ip or host)

    @property
    def needs_poll(self) -> bool:
        """Return True because polling synchronizes state and track metadata."""
        return True

    @property
    def poll_interval(self) -> int:
        """Return polling interval in seconds (faster while playing)."""
        return 10 if self._attr_playback_state == PlaybackState.PLAYING else 30

    async def get_config_entries(self) -> list[ConfigEntry]:
        """Return player-level config entries overriding codec and ICY defaults."""
        return [CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3, _CONF_ENTRY_ICY_METADATA_BASIC]

    async def on_config_updated(self) -> None:
        """
        Persist MP3/ICY protocol defaults and connect to the SoundBridge.

        ``get_config_entries`` sets the UI defaults to ``mp3`` / ``basic``, while
        the streams controller reads ``CONF_ENABLE_ICY_METADATA`` via
        ``get_protocol_value``, which only returns explicitly stored values.
        Writing defaults on first attach ensures ICY headers are active immediately
        without overwriting any explicit user choice.
        """
        for key, default in (
            (CONF_OUTPUT_CODEC, "mp3"),
            (CONF_ENABLE_ICY_METADATA, "basic"),
        ):
            if self.mass.config.get_protocol_value(self.protocol_id, key) is None:
                self.mass.config.set_protocol_value(self.protocol_id, key, default)

        await self._client.connect()
        self._sync_client_state()
        self.update_state()

    async def on_attach(self, player: Player) -> None:
        """
        Subscribe to queue updates when attached to a Player.

        :param player: The Player object this protocol is attached to.
        """
        del player
        if self._unsub_queue_event is None:
            self._unsub_queue_event = self.mass.subscribe(
                self._on_queue_event,
                event_filter=EventType.QUEUE_UPDATED,
            )

    async def on_detach(self) -> None:
        """Unsubscribe from events and close the RCP connection on detach."""
        await super().on_detach()
        if self._unsub_queue_event is not None:
            self._unsub_queue_event()
            self._unsub_queue_event = None
        self._attr_available = False
        await self._client.disconnect()

    async def on_remove(self) -> None:
        """Remove manually configured host entry when the player is deleted."""
        cast("SoundBridgeProvider", self.provider).forget_host(self.host, self.port)

    async def poll(self) -> None:
        """Refresh state from the RCP client and push active queue metadata."""
        self._sync_client_state()
        await self._push_current_track_metadata()
        self.update_state()

    async def power(self, powered: bool) -> None:
        """
        Wake the SoundBridge from standby or put it into standby.

        :param powered: True to wake from standby, False for standby.
        """
        if powered:
            await self._client.turn_on()
            await self._client.wait_for_power_on()
        else:
            await self._client.turn_off()
        self._sync_client_state()
        self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """
        Set output volume level (0-100).

        :param volume_level: Volume percentage from 0 to 100.
        """
        await self._client.set_volume(volume_level)
        self._attr_volume_level = volume_level
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """
        Mute or unmute audio output.

        :param muted: True to mute, False to unmute.
        """
        await self._client.set_mute(muted=muted)
        self._attr_volume_muted = muted
        self.update_state()

    async def play(self) -> None:
        """Resume playback."""
        await self._client.play()
        self._attr_playback_state = PlaybackState.PLAYING
        self.update_state()

    async def pause(self) -> None:
        """Pause playback."""
        await self._client.pause()
        self._attr_playback_state = PlaybackState.PAUSED
        self.update_state()

    async def stop(self) -> None:
        """Stop playback."""
        await self._client.stop()
        self._attr_playback_state = PlaybackState.IDLE
        self._attr_current_media = None
        self.update_state()

    async def play_media(self, media: PlayerMedia) -> None:
        """
        Resolve the Music Assistant stream URL and start playback over RCP.

        :param media: The PlayerMedia item/stream to play.
        """
        if self._client.power_state == "standby":
            await self._client.turn_on()
            await self._client.wait_for_power_on()

        url = await self.mass.streams.resolve_stream_url(self.protocol_id, media)
        ext = url.split("?", 1)[0].rsplit(".", 1)[-1].lower()
        fmt = {"wav": "WAV", "mp3": "MP3", "aac": "AAC", "aif": "AIFF", "aiff": "AIFF"}.get(
            ext, "WAV"
        )

        # In MP3 mode MA injects ICY metadata blocks in-stream which the SoundBridge
        # parses directly for title/artist; we still push single-track trackLength
        # because ICY headers do not carry duration.
        skip_rcp_meta = fmt == "MP3"
        queue_title, queue_artist, queue_duration = self._current_queue_track_metadata()
        title, artist = ("", "") if skip_rcp_meta else (queue_title, queue_artist)
        length_ms = int(queue_duration * 1000) if queue_duration else None

        self._last_play_url_time = time.time()
        self._last_pushed_title = title
        self._last_pushed_artist = artist
        self._last_pushed_duration_ms = length_ms or 0
        await self._client.play_url(
            url,
            title=title,
            artist=artist,
            fmt=fmt,
            length_ms=length_ms,
        )
        self._attr_playback_state = PlaybackState.PLAYING
        self._attr_current_media = media
        self.update_state()

    async def update_endpoint(
        self,
        host: str,
        port: int,
        resolved_ip: str | None = None,
    ) -> None:
        """
        Update connection endpoint if mDNS or manual discovery learned a new address.

        :param host: Updated host or IP address.
        :param port: Updated RCP TCP port.
        :param resolved_ip: Resolved IPv4 address if known.
        """
        if resolved_ip:
            self.resolved_ip = resolved_ip
            self._attr_device_info.add_identifier(IdentifierType.IP_ADDRESS, resolved_ip)
        if (host, port) == (self.host, self.port):
            return
        self.host = host
        self.port = port
        self._attr_device_info.add_identifier(IdentifierType.IP_ADDRESS, resolved_ip or host)
        if self.player is not None:
            await self._client.disconnect()
            self._client = RcpClient(host, port, self._on_client_state_change)
            await self._client.connect()
            self._sync_client_state()
            self.update_state()
        else:
            self._client.host = host
            self._client.port = port

    def _on_client_state_change(self) -> None:
        """Synchronize and publish state when the RCP client receives updates."""
        self._sync_client_state()
        self.update_state()

    def _sync_client_state(self) -> None:
        """Copy RCP client state into PlayerProtocol attributes."""
        client = self._client
        self._attr_available = client.is_connected

        if client.power_state == "standby":
            self._attr_powered = False
            self._attr_playback_state = PlaybackState.IDLE
        else:
            self._attr_powered = True
            transport = client.transport_state
            if transport == "stop" and (
                time.time() - self._last_play_url_time < _PLAY_GRACE_SECONDS
            ):
                transport = "buffering"
            self._attr_playback_state = _TRANSPORT_TO_PLAYBACK.get(transport, PlaybackState.IDLE)

        self._attr_volume_level = client.volume
        self._attr_volume_muted = client.muted

        if client.mac_address:
            normalized_mac = client.mac_address.upper().replace("-", ":")
            self._attr_device_info.add_identifier(IdentifierType.MAC_ADDRESS, normalized_mac)

        if self._attr_playback_state == PlaybackState.PLAYING:
            self._attr_elapsed_time = float(client.position)
            self._attr_elapsed_time_last_updated = client.position_updated_at or time.time()
            queue_title, queue_artist, queue_duration = self._current_queue_track_metadata()
            self._attr_current_media = PlayerMedia(
                uri=client.url or "",
                title=queue_title or client.title or None,
                artist=queue_artist or client.artist or None,
                album=client.album or None,
                duration=queue_duration or client.duration or None,
            )
        else:
            self._attr_elapsed_time = None
            self._attr_elapsed_time_last_updated = None
            self._attr_current_media = None

    async def _on_queue_event(self, event: MassEvent) -> None:
        """Push updated track metadata to the SoundBridge display when its queue advances."""
        if self._attr_playback_state != PlaybackState.PLAYING:
            return
        active_queue = (
            self.mass.players.get_active_queue(self.player)
            if self.player is not None
            else self.mass.player_queues.get_active_queue(self.protocol_id)
        )
        if active_queue is None or event.object_id != active_queue.queue_id:
            return
        await self._push_current_track_metadata()
        self._sync_client_state()
        self.update_state()

    async def _push_current_track_metadata(self) -> None:
        """Push the active queue item's title/artist/duration to the VFD via RCP."""
        if self._attr_playback_state != PlaybackState.PLAYING:
            return
        title, artist, duration_seconds = self._current_queue_track_metadata()

        if title and title != self._last_pushed_title:
            await self._client.set_working_song_info("title", title)
            self._last_pushed_title = title
        if artist and artist != self._last_pushed_artist:
            await self._client.set_working_song_info("artist", artist)
            self._last_pushed_artist = artist
        if duration_seconds:
            duration_ms = int(duration_seconds * 1000)
            if duration_ms != self._last_pushed_duration_ms:
                await self._client.set_working_song_info("trackLength", str(duration_ms))
                self._last_pushed_duration_ms = duration_ms

    def _current_queue_track_metadata(self) -> tuple[str, str, int | None]:
        """Return (title, artist, duration_seconds) for the active queue's current item."""
        queue = (
            self.mass.players.get_active_queue(self.player)
            if self.player is not None
            else self.mass.player_queues.get_active_queue(self.protocol_id)
        )
        if not queue or not queue.current_item:
            return "", "", None
        item = queue.current_item
        media_item = item.media_item
        title = (media_item.name if media_item else None) or item.name or ""
        artists = getattr(media_item, "artists", None) if media_item else None
        artist = " / ".join(a.name for a in artists) if artists else ""
        # Only expose duration when there is a single track in the queue: in flow mode
        # trackLength is locked at QueueAndPlayOne time for the continuous stream.
        duration = item.duration if (item.duration and queue.items == 1) else None
        return title, artist, duration
