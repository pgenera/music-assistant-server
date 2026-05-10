"""Tests for parsing audio file tags (ID3, MP4/AAC, Vorbis, APEv2, etc.)."""

import pathlib
import shutil
from unittest.mock import AsyncMock, MagicMock

import mutagen
import pytest
from music_assistant_models.errors import InvalidDataError

from music_assistant.constants import UNKNOWN_ARTIST
from music_assistant.helpers import tags
from music_assistant.helpers.tags import (
    _parse_apev2_tags,
    _parse_id3_tags,
    _parse_mp4_tags,
    _parse_vorbis_tags,
    parse_tags_mutagen,
    split_artists,
    write_replaygain_track_gain,
)

RESOURCES_DIR = pathlib.Path(__file__).parent.parent.resolve().joinpath("fixtures")

FILE_MP3 = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.mp3"))
FILE_MP3_ID3V24_MULTIVALUE = str(RESOURCES_DIR.joinpath("MultiArtist-ID3v24-NullSeparated.mp3"))
FILE_M4A = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.m4a"))
FILE_FLAC = str(RESOURCES_DIR.joinpath("MultipleArtists.flac"))
FILE_FLAC_SEMICOLON = str(RESOURCES_DIR.joinpath("ArtistWithSemicolon.flac"))
FILE_WV = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.wv"))


async def test_parse_metadata_from_id3tags() -> None:
    """Test parsing of parsing metadata from ID3 tags."""
    filename = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album == "MyAlbum"
    assert _tags.title == "MyTitle"
    assert _tags.duration == 1.032
    assert _tags.album_artists == ("MyArtist",)
    assert _tags.artists == ("MyArtist", "MyArtist2")
    assert _tags.genres == ("Genre1", "Genre2")
    assert _tags.musicbrainz_albumartistids == ("abcdefg",)
    assert _tags.musicbrainz_artistids == ("abcdefg",)
    assert _tags.musicbrainz_releasegroupid == "abcdefg"
    assert _tags.musicbrainz_recordingid == "abcdefg"
    # test parsing disc/track number
    _tags.tags["disc"] = ""
    assert _tags.disc is None
    _tags.tags["disc"] = "1"
    assert _tags.disc == 1
    _tags.tags["disc"] = "1/1"
    assert _tags.disc == 1
    # test parsing album year
    _tags.tags["date"] = "blah"
    assert _tags.year is None
    _tags.tags.pop("date", None)
    assert _tags.year is None
    _tags.tags["date"] = "2022"
    assert _tags.year == 2022
    _tags.tags["date"] = "2022-05-05"
    assert _tags.year == 2022
    _tags.tags["date"] = ""
    assert _tags.year is None


async def test_parse_id3v24_null_separated_artists() -> None:
    """Test parsing ID3v2.4 tags with null-separated multi-value TPE1/TPE2."""
    _tags = await tags.async_parse_tags(FILE_MP3_ID3V24_MULTIVALUE)
    # Null-separated artists in TPE1 should be parsed as multiple artists
    assert _tags.artists == ("Artist One", "Artist Two", "Artist Three")
    # Null-separated album artists in TPE2 should be parsed as multiple album artists
    assert _tags.album_artists == ("Album Artist A", "Album Artist B")
    # MB IDs should match
    assert _tags.musicbrainz_artistids == ("mb-artist-1", "mb-artist-2", "mb-artist-3")
    assert _tags.musicbrainz_albumartistids == ("mb-albumartist-1", "mb-albumartist-2")


async def test_parse_metadata_from_mp4tags() -> None:
    """Test parsing of metadata from MP4/AAC tags."""
    filename = FILE_M4A
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album == "MyAlbum"
    assert _tags.title == "MyTitle"
    assert _tags.album_artists == ("MyArtist",)
    assert _tags.artists == ("MyArtist", "MyArtist2")
    assert _tags.genres == ("Genre1", "Genre2")
    assert _tags.musicbrainz_albumartistids == ("abcdefg",)
    assert _tags.musicbrainz_artistids == ("abcdefg",)
    assert _tags.musicbrainz_releasegroupid == "abcdefg"
    assert _tags.musicbrainz_recordingid == "abcdefg"
    # test track/disc from MP4 tuples
    assert _tags.track == 5
    assert _tags.disc == 1
    # test total track/disc
    assert _tags.tags.get("tracktotal") == "12"
    assert _tags.tags.get("disctotal") == "2"
    # test year
    assert _tags.year == 2022
    # test sort tags (artistsort/albumartistsort returned as lists to match ID3 behavior)
    assert _tags.tags.get("titlesort") == "MyTitle Sort"
    assert _tags.tags.get("artistsort") == ["MyArtist Sort"]  # type: ignore[comparison-overlap]
    assert _tags.tags.get("albumsort") == "MyAlbum Sort"
    assert _tags.tags.get("albumartistsort") == ["MyAlbumArtist Sort"]  # type: ignore[comparison-overlap]


def test_parse_metadata_from_apev2tags() -> None:
    """Test parsing of metadata from APEv2 tags (WavPack).

    Uses parse_tags_mutagen directly since the minimal WavPack fixture
    does not contain valid audio data for ffprobe to parse.
    """
    result = parse_tags_mutagen(FILE_WV)
    assert result.get("album") == "MyAlbum"
    assert result.get("title") == "MyTitle"
    assert result.get("albumartist") == "MyArtist"
    assert result.get("artist") == "MyArtist"
    assert result.get("artists") == ["MyArtist", "MyArtist2"]
    assert result.get("genre") == ["Genre1", "Genre2"]
    assert result.get("musicbrainzalbumartistid") == ["abcdefg"]
    assert result.get("musicbrainzartistid") == ["abcdefg"]
    assert result.get("musicbrainzreleasegroupid") == "abcdefg"
    assert result.get("musicbrainzrecordingid") == "abcdefg"
    # test track/disc (APEv2 uses "5/12" format like ID3)
    assert result.get("track") == "5/12"
    assert result.get("disc") == "1/2"
    # test year
    assert result.get("date") == "2022"
    # test sort tags (artistsort/albumartistsort returned as lists to match ID3 behavior)
    assert result.get("titlesort") == "MyTitle Sort"
    assert result.get("artistsort") == ["MyArtist Sort"]
    assert result.get("albumsort") == "MyAlbum Sort"
    assert result.get("albumartistsort") == ["MyAlbumArtist Sort"]


async def test_parse_metadata_from_flac_with_multiple_artist_fields() -> None:
    """Test parsing of FLAC file with multiple ARTIST fields (per Vorbis spec)."""
    _tags = await tags.async_parse_tags(FILE_FLAC)
    assert _tags.album == "Test Album"
    assert _tags.title == "Test Track"
    # Multiple ARTIST fields should be treated as authoritative list
    assert _tags.artists == ("Artist One", "Artist Two", "Artist Three")
    # Multiple ALBUMARTIST fields should be treated as authoritative list
    assert _tags.album_artists == ("Album Artist 1", "Album Artist 2")
    assert _tags.genres == ("Rock", "Pop")
    assert _tags.year == 2024
    # MusicBrainz IDs
    assert _tags.musicbrainz_artistids == ("mb-artist-id-1", "mb-artist-id-2", "mb-artist-id-3")
    assert _tags.musicbrainz_albumartistids == ("mb-albumartist-id-1", "mb-albumartist-id-2")
    assert _tags.musicbrainz_recordingid == "mb-track-id"
    # Track/disc from Vorbis comments
    assert _tags.track == 5
    assert _tags.disc == 1


async def test_parse_metadata_from_filename() -> None:
    """Test parsing of parsing metadata from filename."""
    filename = str(RESOURCES_DIR.joinpath("MyArtist - MyTitle without Tags.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album is None
    assert _tags.title == "MyTitle without Tags"
    assert _tags.duration == 1.032
    assert _tags.album_artists == ()
    assert _tags.artists == ("MyArtist",)
    assert _tags.genres == ()
    assert _tags.musicbrainz_albumartistids == ()
    assert _tags.musicbrainz_artistids == ()
    assert _tags.musicbrainz_releasegroupid is None
    assert _tags.musicbrainz_recordingid is None


async def test_parse_metadata_from_invalid_filename() -> None:
    """Test parsing of parsing metadata from (invalid) filename."""
    filename = str(RESOURCES_DIR.joinpath("test.mp3"))
    _tags = await tags.async_parse_tags(filename)
    assert _tags.album is None
    assert _tags.title == "test"
    assert _tags.duration == 1.032
    assert _tags.album_artists == ()
    assert _tags.artists == (UNKNOWN_ARTIST,)
    assert _tags.genres == ()
    assert _tags.musicbrainz_albumartistids == ()
    assert _tags.musicbrainz_artistids == ()
    assert _tags.musicbrainz_releasegroupid is None
    assert _tags.musicbrainz_recordingid is None


def test_split_artists_semicolon() -> None:
    """Test that split_artists splits on the semicolon delimiter."""
    assert split_artists("Artist A;Artist B") == ("Artist A", "Artist B")
    assert split_artists("Artist A; Artist B; Artist C") == ("Artist A", "Artist B", "Artist C")
    # No semicolon, no split
    assert split_artists("Single Artist") == ("Single Artist",)


def test_split_artists_featuring() -> None:
    """Test that split_artists always splits on featuring patterns."""
    assert split_artists("John Lennon feat. Yoko Ono") == ("John Lennon", "Yoko Ono")
    assert split_artists("Artist A featuring Artist B") == ("Artist A", "Artist B")
    assert split_artists("Artist A ft. Artist B") == ("Artist A", "Artist B")
    assert split_artists("Artist A vs. Artist B") == ("Artist A", "Artist B")
    # Combined semicolon and featuring
    assert split_artists("Drake;Eminem feat. Rihanna") == ("Drake", "Eminem", "Rihanna")


def test_split_artists_no_unsafe_splitters() -> None:
    """Test that split_artists does NOT split on '&', ',', '+', or 'with'.

    These are too ambiguous (e.g. "Hall & Oates", "Simon & Garfunkel",
    "Jerk With a Bomb") and would over-split real artist names. When the
    user wants disambiguation they should provide MusicBrainz Artist IDs
    or use the format-native multi-value tag (multiple ARTIST fields,
    null-separated TPE1, ARTISTS plural).
    """
    assert split_artists("Hall & Oates") == ("Hall & Oates",)
    assert split_artists("Simon & Garfunkel") == ("Simon & Garfunkel",)
    assert split_artists("Jerk With a Bomb") == ("Jerk With a Bomb",)
    assert split_artists("Shabson, Krgovich & Harris") == ("Shabson, Krgovich & Harris",)


def _create_mock_vorbis_tags(tag_dict: dict[str, list[str]]) -> MagicMock:
    """Create a mock VCommentDict with the given tags.

    :param tag_dict: Dictionary mapping tag names to lists of values.
    """
    mock = MagicMock()
    mock.get = lambda key: tag_dict.get(key.upper())
    return mock


def test_parse_vorbis_tags_multiple_artist_fields() -> None:
    """Test that multiple ARTIST fields are treated as authoritative artist list."""
    # Per Vorbis spec: multiple ARTIST fields should list all artists
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["My Song"],
            "ALBUM": ["My Album"],
            "ARTIST": ["Artist 1", "Artist 2", "Artist 3"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ARTIST fields should be stored as "artists" (plural)
    assert result.get("artists") == ["Artist 1", "Artist 2", "Artist 3"]
    # Single "artist" key should NOT be set when multiple artists are present
    assert "artist" not in result
    assert result.get("title") == "My Song"
    assert result.get("album") == "My Album"


def test_parse_vorbis_tags_single_artist_field() -> None:
    """Test that a single ARTIST field is stored as singular artist."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["My Song"],
            "ARTIST": ["Single Artist"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Single ARTIST should use singular key for normal parsing logic
    assert result.get("artist") == "Single Artist"
    assert "artists" not in result


def test_parse_vorbis_tags_multiple_albumartist_fields() -> None:
    """Test that multiple ALBUMARTIST fields are treated as authoritative list."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ALBUMARTIST": ["Album Artist 1", "Album Artist 2"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ALBUMARTIST fields populate the "albumartists" multi-value bucket
    assert result.get("albumartists") == ["Album Artist 1", "Album Artist 2"]
    assert "albumartist" not in result


def test_parse_vorbis_tags_single_albumartist_field() -> None:
    """Test that a single ALBUMARTIST field is stored as singular."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ALBUMARTIST": ["Single Album Artist"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("albumartist") == "Single Album Artist"
    assert "albumartists" not in result


def test_parse_vorbis_tags_multiple_artist_fields_take_precedence() -> None:
    """Multiple ARTIST fields (Vorbis-spec way) take precedence over the plural ARTISTS tag.

    Tests that when both multiple ARTIST and plural ARTISTS are present, the
    format-native multi-value path wins.
    """
    mock_tags = _create_mock_vorbis_tags(
        {
            "ARTIST": ["Artist A", "Artist B"],
            "ARTISTS": ["Should Not Win 1", "Should Not Win 2", "Should Not Win 3"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("artists") == ["Artist A", "Artist B"]


def test_parse_vorbis_tags_plural_artists_tag_used_as_fallback() -> None:
    """Test that the plural ARTISTS tag is used when only a single ARTIST field exists."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ARTIST": ["Single Artist"],
            "ARTISTS": ["Plural Artist 1", "Plural Artist 2"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("artists") == ["Plural Artist 1", "Plural Artist 2"]


def test_parse_vorbis_tags_musicbrainz_ids() -> None:
    """Test that MusicBrainz IDs are parsed correctly from Vorbis tags."""
    mock_tags = _create_mock_vorbis_tags(
        {
            "ARTIST": ["Artist 1", "Artist 2"],
            "MUSICBRAINZ_ARTISTID": ["mb-id-1", "mb-id-2"],
            "MUSICBRAINZ_ALBUMID": ["mb-album-id"],
            "MUSICBRAINZ_TRACKID": ["mb-track-id"],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    assert result.get("musicbrainzartistid") == ["mb-id-1", "mb-id-2"]
    assert result.get("musicbrainzalbumid") == "mb-album-id"
    assert result.get("musicbrainzrecordingid") == "mb-track-id"


def test_parse_vorbis_multi_value_releasetype() -> None:
    """Repeated RELEASETYPE Vorbis fields are joined into a single value."""
    mock_tags = _create_mock_vorbis_tags({"RELEASETYPE": ["album", "live"]})
    result = _parse_vorbis_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def _create_mock_apev2_tags(tag_dict: dict[str, str]) -> MagicMock:
    r"""Create a mock APEv2 tags object.

    :param tag_dict: Dictionary mapping tag names to values (use \x00 for multi-value).
    """
    mock = MagicMock()
    mock.__contains__ = lambda _, key: key in tag_dict
    mock.__getitem__ = lambda _, key: tag_dict[key]
    mock.keys = lambda: tag_dict.keys()
    return mock


def test_parse_apev2_tags_multi_value_artists() -> None:
    """Test that APEv2 multi-value fields (null-separated) are parsed correctly."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Title": "My Song",
            "Album": "My Album",
            "Artist": "Single Artist",
            "Artists": "Artist 1\x00Artist 2\x00Artist 3",  # Null-separated
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("title") == "My Song"
    assert result.get("album") == "My Album"
    assert result.get("artist") == "Single Artist"
    assert result.get("artists") == ["Artist 1", "Artist 2", "Artist 3"]


def test_parse_apev2_tags_musicbrainz_ids() -> None:
    """Test that MusicBrainz IDs are parsed correctly from APEv2 tags."""
    mock_tags = _create_mock_apev2_tags(
        {
            "MUSICBRAINZ_ARTISTID": "mb-id-1\x00mb-id-2",  # Multi-value
            "MUSICBRAINZ_ALBUMID": "mb-album-id",
            "MUSICBRAINZ_TRACKID": "mb-track-id",  # Recording ID in APEv2
            "MUSICBRAINZ_RELEASEGROUPID": "mb-rg-id",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("musicbrainzartistid") == ["mb-id-1", "mb-id-2"]
    assert result.get("musicbrainzalbumid") == "mb-album-id"
    assert result.get("musicbrainzrecordingid") == "mb-track-id"
    assert result.get("musicbrainzreleasegroupid") == "mb-rg-id"


def test_parse_apev2_multi_value_musicbrainz_albumtype() -> None:
    """Null-separated MUSICBRAINZ_ALBUMTYPE values are joined into a single value."""
    mock_tags = _create_mock_apev2_tags({"MUSICBRAINZ_ALBUMTYPE": "album\x00live"})
    result = _parse_apev2_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_parse_apev2_tags_genre_multi_value() -> None:
    """Test that APEv2 genre with multiple values is parsed correctly."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Genre": "Rock\x00Pop\x00Jazz",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    assert result.get("genre") == ["Rock", "Pop", "Jazz"]


def test_parse_apev2_tags_null_separated_artists() -> None:
    """Test that APEv2 null-separated Artist field is parsed as multiple artists."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Artist": "ave;new\x00佐倉紗織",
            "Album Artist": "Album Artist A\x00Album Artist B",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    # Multiple null-separated values should be stored as "artists" (plural)
    assert result.get("artists") == ["ave;new", "佐倉紗織"]
    assert result.get("albumartists") == ["Album Artist A", "Album Artist B"]
    # Singular keys should not be set
    assert "artist" not in result
    assert "albumartist" not in result


def test_parse_apev2_tags_single_artist() -> None:
    """Test that APEv2 single Artist field is parsed as singular."""
    mock_tags = _create_mock_apev2_tags(
        {
            "Artist": "Single Artist",
            "Album Artist": "Single Album Artist",
        }
    )

    result = _parse_apev2_tags(mock_tags)

    # Single value should be stored as "artist" (singular)
    assert result.get("artist") == "Single Artist"
    assert result.get("albumartist") == "Single Album Artist"
    # Plural keys should not be set
    assert "artists" not in result
    assert "albumartists" not in result


def test_parse_mp4_multi_value_musicbrainz_albumtype() -> None:
    """Multi-value MP4 freeform album type entries are joined into a single value."""
    mock_tags = MagicMock()
    mock_tags.__contains__ = lambda _, key: key == "----:com.apple.iTunes:MusicBrainz Album Type"
    mock_tags.__getitem__ = lambda _, _k: [b"album", b"live"]
    result = _parse_mp4_tags(mock_tags)
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_parse_id3_multi_value_musicbrainz_albumtype() -> None:
    """Multi-value TXXX:MusicBrainz Album Type frame entries are joined into a single value."""
    frame = MagicMock()
    frame.text = ["album", "live"]
    result = _parse_id3_tags({"TXXX:MusicBrainz Album Type": frame})
    assert result.get("musicbrainzalbumtype") == "album;live"


def test_vorbis_multiple_artist_fields_semicolon_in_name() -> None:
    """Test that multiple ARTIST fields in Vorbis with semicolons are handled correctly.

    Regression test for the "ave;new" edge case per the Vorbis spec:
    - Japanese artist "ave;new" has a semicolon in their name
    - Vorbis allows multiple ARTIST (singular) fields for multi-artist tracks
    - The semicolon within "ave;new" must NOT cause additional splitting

    Correct Vorbis tagging (per https://xiph.org/vorbis/doc/v-comment.html):
        ARTIST=ave;new
        ARTIST=佐倉紗織
        MUSICBRAINZ_ARTISTID=2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
        MUSICBRAINZ_ARTISTID=822c07bd-1f8a-4fef-acdb-8acfe82fbef5

    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    # Simulate Vorbis tags with multiple ARTIST fields (correct per Vorbis spec)
    mock_tags = _create_mock_vorbis_tags(
        {
            "TITLE": ["Call My Dears"],
            "ALBUM": ["Lovable"],
            "ARTIST": ["ave;new", "佐倉紗織"],  # Multiple ARTIST fields
            "ARTISTSORT": ["ave;new feat.Sakura, Saori"],
            "MUSICBRAINZ_ARTISTID": [
                "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
                "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
            ],
        }
    )

    result = _parse_vorbis_tags(mock_tags)

    # Multiple ARTIST fields should be stored as "artists" (plural key)
    assert result.get("artists") == ["ave;new", "佐倉紗織"]
    # MusicBrainz Artist IDs should be preserved
    assert result.get("musicbrainzartistid") == [
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    ]

    # Now test that AudioTags.artists property correctly handles the multiple fields
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags=result,
        has_cover_image=False,
        filename="01 - ave;new feat.佐倉紗織 - Call My Dears.flac",
    )

    # The artists property must return exactly 2 artists
    assert audio_tags.artists == ("ave;new", "佐倉紗織")
    # The semicolon in "ave;new" must NOT cause it to be split
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists
    # MusicBrainz Artist IDs should match the artist count
    assert audio_tags.musicbrainz_artistids == (
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    )
    assert len(audio_tags.artists) == len(audio_tags.musicbrainz_artistids)


async def test_flac_multiple_artist_fields_semicolon_e2e() -> None:
    """End-to-end test: FLAC with multiple ARTIST fields, one containing semicolon.

    Tests real file parsing to ensure the full pipeline correctly handles
    artist names with semicolons when using multiple ARTIST fields per Vorbis spec.

    See: https://xiph.org/vorbis/doc/v-comment.html
    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    audio_tags = await tags.async_parse_tags(FILE_FLAC_SEMICOLON)

    # Verify the artists are correctly parsed without splitting on semicolons
    assert audio_tags.artists == ("ave;new", "佐倉紗織")
    assert "ave" not in audio_tags.artists
    assert "new" not in audio_tags.artists

    # Verify MB Artist IDs match
    assert audio_tags.musicbrainz_artistids == (
        "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        "822c07bd-1f8a-4fef-acdb-8acfe82fbef5",
    )
    assert len(audio_tags.artists) == len(audio_tags.musicbrainz_artistids)

    # Verify other tags
    assert audio_tags.title == "Call My Dears"
    assert audio_tags.album == "Lovable"


def test_artist_tag_semicolon_split() -> None:
    """Test that the single ARTIST field is split on semicolons in the fallback path.

    The "ave;new" semicolon-in-name case is handled by resolving the MusicBrainz
    Artist ID (see test_resolve_artists_from_mbids_handles_semicolon_in_name) which
    happens at the provider layer before this property is consulted. This test only
    verifies the property's tag-parsing fallback behaviour when no MBID resolution
    has happened.
    """
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="mp3",
        bit_rate=None,
        duration=180.0,
        tags={
            "artist": "Artist A;Artist B",
        },
        has_cover_image=False,
        filename="test.mp3",
    )

    assert audio_tags.artists == ("Artist A", "Artist B")


def test_artist_tag_single_mbid_preserves_semicolon_name() -> None:
    """A lone MBID disambiguates a semicolon-in-name single artist on the fallback path.

    Even without the MB lookup running, the property must trust that one MBID
    means one artist and return the raw tag string intact. Otherwise "ave;new"
    would be split into ("ave", "new") whenever the MB mirror is unreachable
    or the MB provider isn't loaded.
    """
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags={
            "artist": "ave;new",
            "musicbrainzartistid": "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        },
        has_cover_image=False,
        filename="test.flac",
    )

    assert audio_tags.artists == ("ave;new",)


def test_albumartist_tag_semicolon_split() -> None:
    """Test that the single ALBUMARTIST field is split on semicolons in the fallback."""
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="mp3",
        bit_rate=None,
        duration=180.0,
        tags={
            "albumartist": "Artist A;Artist B",
        },
        has_cover_image=False,
        filename="test.mp3",
    )

    assert audio_tags.album_artists == ("Artist A", "Artist B")


def test_albumartist_tag_single_mbid_preserves_semicolon_name() -> None:
    """Mirror of test_artist_tag_single_mbid_preserves_semicolon_name for album artists."""
    audio_tags = tags.AudioTags(
        raw={},
        sample_rate=44100,
        channels=2,
        bits_per_sample=16,
        format="flac",
        bit_rate=None,
        duration=180.0,
        tags={
            "albumartist": "ave;new",
            "musicbrainzalbumartistid": "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba",
        },
        has_cover_image=False,
        filename="test.flac",
    )

    assert audio_tags.album_artists == ("ave;new",)


def _read_replaygain_track_gain(path: str) -> str | None:
    """Read REPLAYGAIN_TRACK_GAIN from a file using mutagen (format-agnostic)."""
    audio = mutagen.File(path)  # type: ignore[attr-defined]
    if audio is None or audio.tags is None:
        return None
    tag_key_mp4 = "----:com.apple.iTunes:REPLAYGAIN_TRACK_GAIN"
    if tag_key_mp4 in audio.tags:
        val = audio.tags[tag_key_mp4][0]
        return val.decode("utf-8") if isinstance(val, bytes) else str(val)
    if "TXXX:REPLAYGAIN_TRACK_GAIN" in audio.tags:
        return str(audio.tags["TXXX:REPLAYGAIN_TRACK_GAIN"].text[0])
    if "REPLAYGAIN_TRACK_GAIN" in audio.tags:
        return str(audio.tags["REPLAYGAIN_TRACK_GAIN"][0])
    return None


@pytest.mark.parametrize(
    "source",
    [FILE_MP3, FILE_M4A, FILE_FLAC, FILE_WV],
)
async def test_write_replaygain_track_gain_roundtrip(tmp_path: pathlib.Path, source: str) -> None:
    """Write a REPLAYGAIN_TRACK_GAIN tag and verify the value is read back."""
    dest = tmp_path / pathlib.Path(source).name
    shutil.copy(source, dest)

    assert await write_replaygain_track_gain(str(dest), -5.3) is True
    assert _read_replaygain_track_gain(str(dest)) == "-5.30 dB"

    # verify overwrite replaces the previous value
    assert await write_replaygain_track_gain(str(dest), -2.1) is True
    assert _read_replaygain_track_gain(str(dest)) == "-2.10 dB"


async def test_write_replaygain_track_gain_missing_file(tmp_path: pathlib.Path) -> None:
    """Return False if the file does not exist or cannot be opened."""
    assert await write_replaygain_track_gain(str(tmp_path / "nope.mp3"), -5.0) is False


async def test_write_replaygain_track_gain_read_only(tmp_path: pathlib.Path) -> None:
    """Return False if the file cannot be written to."""
    dest = tmp_path / "readonly.mp3"
    shutil.copy(FILE_MP3, dest)
    dest.chmod(0o444)
    try:
        assert await write_replaygain_track_gain(str(dest), -5.0) is False
    finally:
        # restore permissions so tmp_path cleanup can remove the file
        dest.chmod(0o644)


async def test_resolve_artists_from_mbids_handles_semicolon_in_name() -> None:
    """End-to-end test for the MBID resolution path with a semicolon-in-name artist.

    "ave;new" is a real Japanese artist whose name contains a semicolon. The only
    way to correctly resolve this artist when the file uses a single ARTIST/ARTISTS
    field is to look up the canonical name via the MusicBrainz Artist ID.

    See: https://musicbrainz.org/artist/2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba
    """
    mbid = "2ade7b3c-a6f1-4d00-b7f7-fc60abf25dba"
    mock_artist = MagicMock()
    mock_artist.name = "ave;new"
    mock_artist.sort_name = "ave;new"

    mock_provider = MagicMock()
    mock_provider.get_artist_details = AsyncMock(return_value=mock_artist)

    resolved = await tags.resolve_artists_from_mbids((mbid,), mock_provider)

    assert resolved == [("ave;new", mbid, "ave;new")]
    mock_provider.get_artist_details.assert_awaited_once_with(mbid)


async def test_resolve_artists_from_mbids_partial_failure_preserves_positions() -> None:
    """A failed MBID lookup yields None at its position, not list-shrinking.

    The position-aligned return shape lets callers index-fall-back to the raw
    tag string for the failed MBID, instead of silently dropping that artist
    and shifting subsequent positions.
    """
    mbids = ("aaa", "bbb", "ccc")
    artist_a = MagicMock(name="A", sort_name="A")
    artist_a.name = "Artist A"
    artist_a.sort_name = "Artist A sort"
    artist_c = MagicMock()
    artist_c.name = "Artist C"
    artist_c.sort_name = "Artist C sort"

    async def fake_get(mbid: str) -> MagicMock:
        if mbid == "bbb":
            raise InvalidDataError("boom")
        return artist_a if mbid == "aaa" else artist_c

    mock_provider = MagicMock()
    mock_provider.get_artist_details = AsyncMock(side_effect=fake_get)

    resolved = await tags.resolve_artists_from_mbids(mbids, mock_provider)

    assert resolved == [
        ("Artist A", "aaa", "Artist A sort"),
        None,
        ("Artist C", "ccc", "Artist C sort"),
    ]


async def test_resolve_artists_from_mbids_all_failed_returns_all_none() -> None:
    """All-fail case: every position is None; caller can fall back wholesale to tags."""
    mbids = ("aaa", "bbb")
    mock_provider = MagicMock()
    mock_provider.get_artist_details = AsyncMock(side_effect=InvalidDataError("boom"))

    resolved = await tags.resolve_artists_from_mbids(mbids, mock_provider)

    assert resolved == [None, None]
