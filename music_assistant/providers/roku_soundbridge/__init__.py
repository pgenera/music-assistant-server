"""Roku SoundBridge player provider for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from music_assistant_models.config_entries import ConfigEntry
from music_assistant_models.enums import ConfigEntryType, ProviderFeature

from .constants import CONF_HOST, CONF_NAME, CONF_PORT, DEFAULT_NAME, DEFAULT_PORT
from .provider import SoundBridgeProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ConfigValueType, ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType

SUPPORTED_FEATURES: set[ProviderFeature] = set()


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """Initialize a SoundBridge provider instance."""
    return SoundBridgeProvider(mass, manifest, config, SUPPORTED_FEATURES)


async def get_config_entries(
    mass: MusicAssistant,
    instance_id: str | None = None,
    action: str | None = None,
    values: dict[str, ConfigValueType] | None = None,
) -> tuple[ConfigEntry, ...]:
    """Return config entries for a SoundBridge provider instance."""
    return (
        ConfigEntry(
            key=CONF_HOST,
            type=ConfigEntryType.STRING,
            label="Host",
            required=True,
            description="IP address or hostname of the SoundBridge.",
        ),
        ConfigEntry(
            key=CONF_PORT,
            type=ConfigEntryType.INTEGER,
            label="RCP Port",
            required=False,
            default_value=str(DEFAULT_PORT),
            description="TCP port for the RCP control connection (default 5555).",
        ),
        ConfigEntry(
            key=CONF_NAME,
            type=ConfigEntryType.STRING,
            label="Name",
            required=False,
            default_value=DEFAULT_NAME,
            description="Display name for this player.",
        ),
    )
