"""Tests for the DLNA player."""

from unittest.mock import MagicMock

import pytest

from music_assistant.providers.dlna.player import DLNAPlayer
from music_assistant_models.enums import PlayerType


@pytest.mark.asyncio
async def test_dlna_player_get_config_entries_roku_wav() -> None:
    """Test that Roku SoundBridge defaults to WAV due to FLAC unavailability."""
    provider = MagicMock()
    
    device = MagicMock()
    device.manufacturer = "Roku"
    device.model_name = "SoundBridge Radio"
    # Provide a list of sink_protocol_info that DOES NOT contain flac but DOES contain wav
    device.sink_protocol_info = [
        "http-get:*:audio/mpeg:*",
        "http-get:*:audio/wav:*",
        "http-get:*:audio/wma:*"
    ]
    
    player = DLNAPlayer(provider, "uuid:1234", "http://127.0.0.1/desc.xml", device=device)
    
    entries = await player.get_config_entries()
    
    # We expect one of the entries to be the output codec, with default_value="wav"
    codec_entries = [e for e in entries if e.key == "output_codec"]
    assert len(codec_entries) == 1
    assert codec_entries[0].default_value == "wav"


@pytest.mark.asyncio
async def test_dlna_player_get_config_entries_flac_supported() -> None:
    """Test that a device supporting FLAC does not get WAV overridden."""
    provider = MagicMock()
    
    device = MagicMock()
    device.manufacturer = "Roku"  # Even if Roku, if it somehow supports flac, don't override
    device.model_name = "SoundBridge"
    device.sink_protocol_info = [
        "http-get:*:audio/flac:*",
        "http-get:*:audio/wav:*"
    ]
    
    player = DLNAPlayer(provider, "uuid:1234", "http://127.0.0.1/desc.xml", device=device)
    
    entries = await player.get_config_entries()
    
    # No extra output codec entry should be added enforcing WAV
    # The default PLAYER_CONFIG_ENTRIES might not even have output_codec, 
    # but we just check we didn't add the override.
    # Actually DLNAPlayer provider doesn't add output_codec by default, it relies on universal player default
    # So length should just be the default PLAYER_CONFIG_ENTRIES length.
    # Let's just assert "wav" is not the default.
    codec_entries = [e for e in entries if e.key == "output_codec"]
    assert len(codec_entries) == 0


@pytest.mark.asyncio
async def test_dlna_player_get_config_entries_mp3_fallback() -> None:
    """Test that a device supporting neither FLAC nor WAV falls back to MP3."""
    provider = MagicMock()
    
    device = MagicMock()
    device.manufacturer = "SomeBrand"
    device.model_name = "SomeDevice"
    device.sink_protocol_info = [
        "http-get:*:audio/mpeg:*"
    ]
    
    player = DLNAPlayer(provider, "uuid:1234", "http://127.0.0.1/desc.xml", device=device)
    
    entries = await player.get_config_entries()
    
    codec_entries = [e for e in entries if e.key == "output_codec"]
    assert len(codec_entries) == 1
    assert codec_entries[0].default_value == "mp3"
