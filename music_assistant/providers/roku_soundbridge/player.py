"""SoundBridge player implementation for Music Assistant."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from music_assistant_models.enums import (
    ContentType,
    IdentifierType,
    PlaybackState,
    PlayerFeature,
    PlayerType,
)
from music_assistant_models.player import DeviceInfo, PlayerMedia

from music_assistant.models.player import Player

from .rcp_client import RcpClient

if TYPE_CHECKING:
    from .provider import SoundBridgeProvider


_TRANSPORT_TO_PLAYBACK = {
    "play": PlaybackState.PLAYING,
    "pause": PlaybackState.PAUSED,
    "stop": PlaybackState.IDLE,
    "buffering": PlaybackState.PLAYING,
}


class SoundBridgePlayer(Player):
    """A Roku SoundBridge player controlled via the RCP protocol."""

    _attr_type = PlayerType.PLAYER

    def __init__(self, provider: SoundBridgeProvider, player_id: str, host: str, port: int, name: str) -> None:
        """Initialize the player and create an RCP client."""
        self._client = RcpClient(host, port, self._on_client_state_change)
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
        """Called by RcpClient whenever device state changes."""
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
            self._attr_playback_state = _TRANSPORT_TO_PLAYBACK.get(
                client.transport_state, PlaybackState.IDLE
            )

        self._attr_volume_level = client.volume
        self._attr_volume_muted = client.muted

        if client.mac_address:
            self._attr_device_info.identifiers[IdentifierType.MAC_ADDRESS] = (
                client.mac_address.upper().replace("-", ":")
            )

        if self._attr_playback_state == PlaybackState.PLAYING:
            self._attr_elapsed_time = float(client.position)
            self._attr_elapsed_time_last_updated = client.position_updated_at or time.time()
            self._attr_current_media = PlayerMedia(
                uri=client.url or "",
                title=client.title or None,
                artist=client.artist or None,
                album=client.album or None,
                duration=client.duration or None,
            )
        else:
            self._attr_elapsed_time = None
            self._attr_elapsed_time_last_updated = None
            self._attr_current_media = None

    @property
    def needs_poll(self) -> bool:
        """Return if the player needs polling."""
        return True

    @property
    def poll_interval(self) -> int:
        """Return polling interval in seconds."""
        return 10 if self._attr_playback_state == PlaybackState.PLAYING else 30

    async def on_config_updated(self) -> None:
        """Connect when config is first loaded."""
        await self._client.connect()

    async def on_unload(self) -> None:
        """Disconnect when player is unloaded."""
        await self._client.disconnect()

    async def poll(self) -> None:
        """Refresh state from the client."""
        self._sync_client_state()
        self.update_state()

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
        """Set volume (0–100)."""
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
        """Resolve the MA stream URL and push it to the device via PlayStation."""
        if self._client.power_state == "standby":
            await self._client.turn_on()
            await self._client.wait_for_power_on()

        # Prefer WAV (lossless, uncompressed). Exception: if the source is already
        # MP3, pass it through to avoid a lossy transcode cycle.
        output_fmt = "wav"
        if media.source_id and media.queue_item_id:
            queue_item = self.mass.player_queues.get_item(media.source_id, media.queue_item_id)
            if (
                queue_item
                and queue_item.streamdetails
                and queue_item.streamdetails.audio_format.content_type == ContentType.MP3
            ):
                output_fmt = "mp3"

        url = await self.mass.streams.resolve_stream_url(self.player_id, media)
        url = url.rsplit(".", 1)[0] + "." + output_fmt

        await self._client.play_url(url)
        self._attr_playback_state = PlaybackState.PLAYING
        self._attr_current_media = media
        self.update_state()
