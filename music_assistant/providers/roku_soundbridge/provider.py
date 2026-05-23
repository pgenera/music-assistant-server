"""SoundBridge player provider for Music Assistant."""

from __future__ import annotations

from typing import cast

from music_assistant.models.player_provider import PlayerProvider

from .constants import CONF_HOST, CONF_NAME, CONF_PORT, DEFAULT_NAME, DEFAULT_PORT
from .player import SoundBridgePlayer


class SoundBridgeProvider(PlayerProvider):
    """Player provider for Roku SoundBridge devices."""

    async def discover_players(self) -> None:
        """Register the configured SoundBridge as a player."""
        host = cast("str", self.config.get_value(CONF_HOST))
        port = int(cast("int | str", self.config.get_value(CONF_PORT, DEFAULT_PORT)))
        name = cast("str", self.config.get_value(CONF_NAME, DEFAULT_NAME))

        player_id = f"soundbridge_{host.replace('.', '_').replace(':', '_')}"
        player = SoundBridgePlayer(
            provider=self,
            player_id=player_id,
            host=host,
            port=port,
            name=name,
        )
        await self.mass.players.register(player)

    async def unload(self, is_removed: bool = False) -> None:
        """Disconnect players when the provider is unloaded."""
        for player in self.players:
            await self.mass.players.unregister(player.player_id)
