"""Roku SoundBridge player provider for Music Assistant."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import SoundBridgeProvider

if TYPE_CHECKING:
    from music_assistant_models.config_entries import ProviderConfig
    from music_assistant_models.provider import ProviderManifest

    from music_assistant.mass import MusicAssistant
    from music_assistant.models import ProviderInstanceType


async def setup(
    mass: MusicAssistant, manifest: ProviderManifest, config: ProviderConfig
) -> ProviderInstanceType:
    """
    Initialize a Roku SoundBridge provider instance.

    :param mass: The MusicAssistant server instance.
    :param manifest: The provider manifest.
    :param config: The provider configuration.
    """
    return SoundBridgeProvider(mass, manifest, config)
