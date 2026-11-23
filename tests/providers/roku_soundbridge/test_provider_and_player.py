"""Unit tests for SoundBridgeProvider dual discovery and SoundBridgePlayer protocol behavior."""

from __future__ import annotations

import ipaddress
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from music_assistant_models.enums import (
    EventType,
    IdentifierType,
    MediaType,
    PlaybackState,
)
from music_assistant_models.event import MassEvent
from music_assistant_models.player import PlayerMedia
from zeroconf import ServiceStateChange

from music_assistant.constants import CONF_ENABLE_ICY_METADATA, CONF_OUTPUT_CODEC
from music_assistant.providers.roku_soundbridge.player import SoundBridgePlayer
from music_assistant.providers.roku_soundbridge.provider import (
    CONF_MANUAL_IPS,
    SoundBridgeProvider,
    extract_mdns_friendly_name,
    make_protocol_id,
    parse_host_entry,
)


def _make_mock_provider(
    manual_hosts: list[str] | None = None,
) -> tuple[SoundBridgeProvider, MagicMock]:
    """Construct a SoundBridgeProvider and its underlying MagicMock mass controller."""
    provider = SoundBridgeProvider.__new__(SoundBridgeProvider)
    values = {CONF_MANUAL_IPS: SimpleNamespace(value=list(manual_hosts or []))}
    config = MagicMock()
    config.instance_id = "roku_soundbridge_1"
    config.values = values
    config.get_value.side_effect = lambda key, default=None: (
        values[key].value if key in values else default
    )
    provider.config = config
    provider.logger = MagicMock()
    manifest = MagicMock()
    manifest.domain = "roku_soundbridge"
    provider.manifest = manifest

    registered: dict[str, SoundBridgePlayer] = {}

    async def _register(protocol: SoundBridgePlayer) -> None:
        registered[protocol.protocol_id] = protocol

    async def _remove_protocol(protocol_id: str) -> None:
        registered.pop(protocol_id, None)

    mass = MagicMock()
    mass.config.get_base_protocol_config.side_effect = lambda pid, _prov_id: SimpleNamespace(
        protocol_id=pid,
        enabled=True,
        default_name=None,
        setup_data={},
        values={},
    )
    stored_protocol_values: dict[tuple[str, str], Any] = {}
    mass.config.get_protocol_value.side_effect = lambda pid, key, default=None: (
        stored_protocol_values.get((pid, key), default)
    )

    def _set_protocol_value(pid: str, key: str, val: Any) -> None:
        stored_protocol_values[(pid, key)] = val

    mass.config.set_protocol_value.side_effect = _set_protocol_value
    mass.config.get_protocol_record.side_effect = lambda pid: (
        {"protocol_id": pid} if pid in registered else None
    )
    mass.players.wants_protocol.return_value = True
    mass.players.register = AsyncMock(side_effect=_register)
    mass.players.remove_protocol = AsyncMock(side_effect=_remove_protocol)
    mass.players.get_protocol.side_effect = registered.get
    mass.players.trigger_player_update = MagicMock()

    provider.mass = mass
    return provider, mass


def _make_zeroconf_info(ip: str, port: int = 5555) -> MagicMock:
    """Create a mock AsyncServiceInfo returning the given IPv4 address and port."""
    info = MagicMock()
    info.port = port
    addr = ipaddress.IPv4Address(ip)
    info.ip_addresses_by_version.side_effect = lambda version: (
        [addr] if "V4" in str(version) else []
    )
    return info


def test_host_entry_and_protocol_id_helpers() -> None:
    """Verify host[:port] parsing, backward-compatible protocol IDs, and mDNS name parsing."""
    assert parse_host_entry("  ") is None
    assert parse_host_entry("roku.fivesevenfive.org") == ("roku.fivesevenfive.org", 5555)
    assert parse_host_entry("192.168.1.50:5556") == ("192.168.1.50", 5556)

    assert make_protocol_id("roku.fivesevenfive.org", 5555) == "soundbridge_roku_fivesevenfive_org"
    assert make_protocol_id("192.168.1.50", 5556) == "soundbridge_192_168_1_50_5556"

    assert (
        extract_mdns_friendly_name("Kitchen SoundBridge._roku-rcp._tcp.local.", "192.168.1.50")
        == "Kitchen SoundBridge"
    )
    assert (
        extract_mdns_friendly_name("SoundBridge._roku-rcp._tcp.local.", "192.168.1.50")
        == "SoundBridge (192.168.1.50)"
    )


async def test_multiple_soundbridges_via_mdns_and_manual_with_dedup() -> None:
    """Multiple SoundBridges register via mDNS and manual config without duplicating shared IPs."""
    provider, _mass = _make_mock_provider(manual_hosts=["192.168.1.50", "10.20.0.15:5556"])
    await provider.handle_async_init()
    await provider.loaded_in_mass()
    await provider.discover_players()

    assert set(provider._players) == {
        "soundbridge_192_168_1_50",
        "soundbridge_10_20_0_15_5556",
    }

    # mDNS announces 192.168.1.50 (already manually configured) AND a third unit 192.168.1.51
    await provider.on_mdns_service_state_change(
        "Living Room._roku-rcp._tcp.local.",
        ServiceStateChange.Added,
        _make_zeroconf_info("192.168.1.50", 5555),
    )
    await provider.on_mdns_service_state_change(
        "Office._roku-rcp._tcp.local.",
        ServiceStateChange.Added,
        _make_zeroconf_info("192.168.1.51", 5555),
    )

    assert set(provider._players) == {
        "soundbridge_192_168_1_50",
        "soundbridge_10_20_0_15_5556",
        "soundbridge_192_168_1_51",
    }
    assert provider._players["soundbridge_192_168_1_51"].name == "Office"


async def test_removing_player_prunes_manual_host_entry() -> None:
    """Removing a manually configured SoundBridge prunes only that host from provider config."""
    provider, mass = _make_mock_provider(manual_hosts=["192.168.1.50", "192.168.1.52:5555"])
    await provider.handle_async_init()
    await provider.loaded_in_mass()
    await provider.discover_players()

    player = provider._players["soundbridge_192_168_1_50"]
    await player.on_remove()

    mass.config.set_raw_provider_config_value.assert_called_once_with(
        "roku_soundbridge_1",
        CONF_MANUAL_IPS,
        ["192.168.1.52:5555"],
        encrypted=False,
        immediate=False,
    )


async def test_player_defaults_mp3_icy_and_play_media_modes() -> None:
    """SoundBridgePlayer defaults to MP3+basic ICY, skips RCP title on MP3, and pushes WAV metadata."""
    provider, mass = _make_mock_provider()
    await provider.handle_async_init()

    player = SoundBridgePlayer(
        provider=provider,
        protocol_id="soundbridge_192_168_1_50",
        host="192.168.1.50",
        port=5555,
        name="Living Room",
    )
    mock_client = MagicMock()
    mock_client.is_connected = True
    mock_client.power_state = "on"
    mock_client.transport_state = "stop"
    mock_client.volume = 40
    mock_client.muted = False
    mock_client.mac_address = "00-0d-4b-aa-bb-cc"
    mock_client.position = 0
    mock_client.position_updated_at = 0.0
    mock_client.url = ""
    mock_client.title = ""
    mock_client.artist = ""
    mock_client.album = ""
    mock_client.duration = 0
    mock_client.connect = AsyncMock(return_value=True)
    mock_client.play_url = AsyncMock()
    mock_client.set_working_song_info = AsyncMock()
    player._client = mock_client

    entries = await player.get_config_entries()
    by_key = {entry.key: entry for entry in entries}
    assert by_key[CONF_OUTPUT_CODEC].default_value == "mp3"
    assert by_key[CONF_ENABLE_ICY_METADATA].default_value == "basic"

    await player.on_config_updated()
    assert mass.config.get_protocol_value("soundbridge_192_168_1_50", CONF_OUTPUT_CODEC) == "mp3"
    assert (
        mass.config.get_protocol_value("soundbridge_192_168_1_50", CONF_ENABLE_ICY_METADATA)
        == "basic"
    )
    assert (IdentifierType.MAC_ADDRESS, "00:0D:4B:AA:BB:CC") in player.identifiers

    parent_player = MagicMock()
    parent_player.player_id = "ma_player_1"
    player._player = parent_player

    queue = SimpleNamespace(
        queue_id="ma_player_1",
        items=1,
        current_item=SimpleNamespace(
            name="Fallback Name",
            duration=210,
            media_item=SimpleNamespace(
                name="So What",
                artists=[SimpleNamespace(name="Miles Davis")],
            ),
        ),
    )
    mass.players.get_active_queue.return_value = queue

    mass.streams.resolve_stream_url = AsyncMock(
        return_value="http://192.168.1.10:8097/flow/session/item.mp3"
    )
    media = PlayerMedia(uri="library://track/1", media_type=MediaType.TRACK)
    await player.play_media(media)
    mock_client.play_url.assert_awaited_once_with(
        "http://192.168.1.10:8097/flow/session/item.mp3",
        title="",
        artist="",
        fmt="MP3",
        length_ms=210000,
    )
    assert player.playback_state == PlaybackState.PLAYING

    # Queue advances to track 2 of 2: QUEUE_UPDATED immediately pushes new title/artist via RCP
    queue.items = 2
    queue.current_item = SimpleNamespace(
        name="Freddie Freeloader",
        duration=589,
        media_item=SimpleNamespace(
            name="Freddie Freeloader",
            artists=[SimpleNamespace(name="Miles Davis")],
        ),
    )
    await player._on_queue_event(
        MassEvent(EventType.QUEUE_UPDATED, object_id="ma_player_1", data=queue)
    )
    mock_client.set_working_song_info.assert_any_await("title", "Freddie Freeloader")
    mock_client.set_working_song_info.assert_any_await("artist", "Miles Davis")

    # WAV stream with >1 queue items: pushes title & artist via RCP, omits trackLength
    mock_client.play_url.reset_mock()
    mass.streams.resolve_stream_url = AsyncMock(
        return_value="http://192.168.1.10:8097/flow/session/item.wav"
    )
    await player.play_media(media)
    mock_client.play_url.assert_awaited_once_with(
        "http://192.168.1.10:8097/flow/session/item.wav",
        title="Freddie Freeloader",
        artist="Miles Davis",
        fmt="WAV",
        length_ms=None,
    )
