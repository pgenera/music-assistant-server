"""SoundBridge player implementation for Music Assistant."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry, ConfigValueType
from music_assistant_models.enums import (
    EventType,
    IdentifierType,
    PlaybackState,
    PlayerFeature,
    PlayerType,
)
from music_assistant_models.event import MassEvent
from music_assistant_models.player import DeviceInfo, PlayerMedia

from music_assistant.constants import (
    CONF_ENTRY_ENABLE_ICY_METADATA,
    CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3,
)
from music_assistant.models.player import Player

from .rcp_client import RcpClient

if TYPE_CHECKING:
    from .provider import SoundBridgeProvider


# The SoundBridge does not support FLAC and rejects WAV via ICY/Shoutcast,
# so we default new SoundBridge players to MP3 with basic ICY metadata.
# MP3+ICY gives us:
#  - Reliable startup (no probe-and-reconnect issues seen with WAV)
#  - In-stream metadata (no separate RCP SetWorkingSongInfo needed per track)
#  - ~400ms faster click-to-audio than WAV
# The user can still switch to WAV via the standard config UI if desired.
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
# transient "stop" reports from the device during its buffering phase.
# Without this, MA sees IDLE and immediately re-calls play_media, creating a
# rapid reconnect loop that floods MA with competing flow streams.
_PLAY_GRACE_SECONDS = 15


class SoundBridgePlayer(Player):
    """A Roku SoundBridge player controlled via the RCP protocol."""

    _attr_type = PlayerType.PLAYER

    def __init__(
        self,
        provider: SoundBridgeProvider,
        player_id: str,
        host: str,
        port: int,
        name: str,
    ) -> None:
        """Initialize the player and create an RCP client."""
        self._client = RcpClient(host, port, self._on_client_state_change)
        self._last_play_url_time: float = 0.0
        self._last_pushed_title: str = ""
        self._last_pushed_artist: str = ""
        self._last_pushed_duration_ms: int = 0
        self._unsub_queue_event = None
        # Set name before super().__init__ so PlayerState is built with the right name.
        # All other _attr_* assignments must come AFTER super().__init__ because
        # super().__init__ resets mutable _attr_* defaults (e.g. supported_features = set()).
        self._attr_name = name
        super().__init__(provider, player_id)
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
        self._attr_device_info.identifiers[IdentifierType.IP_ADDRESS] = host

    def _on_client_state_change(self) -> None:
        """Handle a state change notification from the RCP client."""
        self._sync_client_state()
        self.update_state()

    def _sync_client_state(self) -> None:
        """Copy RCP client state into player attributes."""
        client = self._client
        self._attr_available = client.is_connected

        if client.power_state == "standby":
            self._attr_powered = False
            self._attr_playback_state = PlaybackState.IDLE
        else:
            self._attr_powered = True
            transport = client.transport_state
            # During the buffering window after a play command the device reports
            # "stop" before it starts playing. Suppress that so MA does not see
            # an unexpected IDLE and immediately re-issue play_media.
            if transport == "stop" and (
                time.time() - self._last_play_url_time < _PLAY_GRACE_SECONDS
            ):
                transport = "buffering"
            self._attr_playback_state = _TRANSPORT_TO_PLAYBACK.get(transport, PlaybackState.IDLE)

        self._attr_volume_level = client.volume
        self._attr_volume_muted = client.muted

        if client.mac_address:
            self._attr_device_info.identifiers[IdentifierType.MAC_ADDRESS] = (
                client.mac_address.upper().replace("-", ":")
            )

        if self._attr_playback_state == PlaybackState.PLAYING:
            self._attr_elapsed_time = float(client.position)
            self._attr_elapsed_time_last_updated = client.position_updated_at or time.time()
            # Prefer MA's queue current_item over RCP-polled title/artist —
            # MA knows the new track title at the exact moment the queue
            # advances, whereas RCP needs the ICY block to land on the device
            # (~1-2s) plus the next poll cycle (~5s) before GetCurrentSongInfo
            # surfaces it. Fall back to RCP only when there is no active queue
            # (e.g. user kicks off playback from the device's IR remote).
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

    async def get_config_entries(
        self,
        action: str | None = None,
        values: dict[str, ConfigValueType] | None = None,
    ) -> list[ConfigEntry]:
        """Override codec and ICY defaults for SoundBridge devices."""
        return [CONF_ENTRY_OUTPUT_CODEC_DEFAULT_MP3, _CONF_ENTRY_ICY_METADATA_BASIC]

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs polling."""
        return True

    @property
    def poll_interval(self) -> int:
        """Return polling interval in seconds."""
        return 10 if self._attr_playback_state == PlaybackState.PLAYING else 30

    async def on_config_updated(self) -> None:
        """Persist codec/ICY defaults, subscribe to queue events, and connect.

        get_config_entries() overrides the display default to mp3/basic, but
        MA's streams controller reads with get_raw_player_config_value() which
        falls back to the BASE entry default (disabled) when nothing is
        persisted. So we explicitly write our defaults the first time we run
        for this player; user-set values won't be overwritten because we only
        write when the raw value is missing.
        """
        for key, default in (
            ("output_codec", "mp3"),
            ("enable_icy_metadata", "basic"),
        ):
            if self.mass.config.get_raw_player_config_value(self.player_id, key) is None:
                self.mass.config.set_raw_player_config_value(self.player_id, key, default)

        # Subscribe to queue events so we can push title/artist to the device
        # the moment the queue advances, instead of waiting for the next poll
        # cycle. Tied to this player_id so we only see events for our queue.
        if self._unsub_queue_event is None:
            self._unsub_queue_event = self.mass.subscribe(
                self._on_queue_event,
                event_filter=EventType.QUEUE_UPDATED,
                id_filter=self.player_id,
            )

        await self._client.connect()

    async def on_unload(self) -> None:
        """Disconnect the device when player is unloaded."""
        if self._unsub_queue_event is not None:
            self._unsub_queue_event()
            self._unsub_queue_event = None
        await self._client.disconnect()

    async def _on_queue_event(self, event: MassEvent) -> None:
        """React to queue updates by pushing the new title/artist to the device.

        MA fires QUEUE_UPDATED when the current item changes (track advance,
        manual skip, etc.). Pushing via RCP here gives the device display
        the new metadata immediately — far faster than waiting for the next
        ICY block to land or the next poll cycle to run.
        """
        if self._attr_playback_state != PlaybackState.PLAYING:
            return
        await self._push_current_track_metadata()

    async def poll(self) -> None:
        """Refresh state from the client."""
        self._sync_client_state()
        await self._push_current_track_metadata()
        self.update_state()

    async def _push_current_track_metadata(self) -> None:
        """Push the active queue item's title/artist to the device display.

        Pushes via RCP SetWorkingSongInfo regardless of codec. For MP3
        playback the device will also receive ICY in-stream metadata
        eventually, but an explicit push at the moment of track change
        is much faster than waiting for the buffer to drain to the new
        ICY block.
        """
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

    async def power(self, powered: bool) -> None:
        """Handle power on/off — wakes from standby or puts device into standby."""
        if powered:
            await self._client.turn_on()
            await self._client.wait_for_power_on()
        else:
            await self._client.turn_off()
        self._sync_client_state()
        self.update_state()

    async def volume_set(self, volume_level: int) -> None:
        """Set volume (0-100)."""
        await self._client.set_volume(volume_level)
        self._attr_volume_level = volume_level
        self.update_state()

    async def volume_mute(self, muted: bool) -> None:
        """Mute or unmute."""
        await self._client.set_mute(muted)
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
        """Resolve the MA stream URL and push it directly to the device."""
        if self._client.power_state == "standby":
            await self._client.turn_on()
            await self._client.wait_for_power_on()

        url = await self.mass.streams.resolve_stream_url(self.player_id, media)
        ext = url.rsplit(".", 1)[-1].lower()
        fmt = {"wav": "WAV", "mp3": "MP3", "aac": "AAC", "aif": "AIFF", "aiff": "AIFF"}.get(
            ext, "WAV"
        )

        # When MA injects ICY metadata in-stream (MP3 mode), skip the title/
        # artist RCP push — the device parses ICY blocks and updates the
        # display itself. We always push trackLength though: ICY doesn't
        # carry duration, so without it the device has no total-time to show.
        skip_rcp_meta = fmt == "MP3"
        queue_title, queue_artist, queue_duration = self._current_queue_track_metadata()
        title, artist = ("", "") if skip_rcp_meta else (queue_title, queue_artist)
        length_ms = int(queue_duration * 1000) if queue_duration else None

        self._last_play_url_time = time.time()
        self._last_pushed_title = title
        self._last_pushed_artist = artist
        self._last_pushed_duration_ms = length_ms or 0
        await self._client.play_url(
            url, title=title, artist=artist, fmt=fmt, length_ms=length_ms
        )
        self._attr_playback_state = PlaybackState.PLAYING
        self._attr_current_media = media
        self.update_state()

    def _current_queue_track_metadata(self) -> tuple[str, str, int | None]:
        """Return (title, artist, duration_seconds) of the active queue's current item.

        Title and artist are empty strings when not available. Duration is
        None when:
          - the queue item doesn't carry a known length (e.g. a live stream
            queued from outside MA), OR
          - the queue has more than one item (in flow mode the SoundBridge
            sees the whole queue as one continuous song, so the per-track
            trackLength we'd push is wrong after the first track — better
            to show no total than a stale one).
        """
        queue = self.mass.player_queues.get_active_queue(self.player_id)
        if not queue or not queue.current_item:
            return "", "", None
        item = queue.current_item
        media_item = item.media_item
        title = (media_item.name if media_item else None) or item.name or ""
        artists = getattr(media_item, "artists", None) if media_item else None
        artist = " / ".join(a.name for a in artists) if artists else ""
        # Only expose a duration when there's exactly one track in the queue.
        # The flow stream's trackLength gets locked at QueueAndPlayOne time
        # and the device ignores updates to it; better to skip than mislead.
        duration = item.duration if (item.duration and queue.items == 1) else None
        return title, artist, duration
