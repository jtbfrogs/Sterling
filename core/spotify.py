"""
Sterling Spotify Integration
=============================
Controls Spotify playback via the Spotify Web API using spotipy.

Requirements:
  - Spotify Premium account (free accounts cannot control playback via API)
  - A Spotify developer app — create one at developer.spotify.com (free, 2 mins):
      1. Log in → Create App → name it anything
      2. Add redirect URI: http://localhost:8888/callback
      3. Copy Client ID and Client Secret into config.yaml

First run:
  Sterling will open a browser tab for Spotify login and permission grant.
  After you approve, the token is cached at spotify.cache_path in config.yaml.
  Every run after that is fully automatic — no browser needed.

Scopes used:
  user-read-playback-state    — read current track / device info
  user-modify-playback-state  — play, pause, skip, volume
  user-read-currently-playing — now playing
"""

import spotipy
from spotipy.oauth2 import SpotifyOAuth
from typing import Optional

from utils.logger import setup_logger

logger = setup_logger("sterling.spotify")

SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
])


class Spotify:
    """
    Sterling Spotify controller.

    Basic usage:
        sp = Spotify(client_id, client_secret, redirect_uri)
        sp.play()                    # resume
        sp.play("Radiohead")         # search and play
        sp.pause()
        sp.skip()
        sp.set_volume(50)
        track = sp.now_playing()     # returns "Song — Artist" or None
    """

    def __init__(
        self,
        client_id:     str,
        client_secret: str,
        redirect_uri:  str = "http://localhost:8888/callback",
        cache_path:    str = ".spotify_cache",
    ):
        self._client = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=SCOPES,
            cache_path=cache_path,
            open_browser=True,
        ))
        logger.info("Spotify client initialised.")

    # ─────────────────────────────────────────────────────────────────────────
    # Playback control
    # ─────────────────────────────────────────────────────────────────────────

    def play(self, query: Optional[str] = None) -> bool:
        """
        Resume playback, or search for a query and play the top result.

        Args:
            query: Artist, song, or album name. None = resume current playback.

        Returns:
            True on success, False on failure.
        """
        try:
            device_id = self._active_device()
            if query:
                uri = self._search(query)
                if not uri:
                    logger.warning(f"Spotify: no results for '{query}'")
                    return False
                # Artist/playlist URIs use context_uri, track URIs use uris list
                if ":track:" in uri:
                    self._client.start_playback(device_id=device_id, uris=[uri])
                else:
                    self._client.start_playback(device_id=device_id, context_uri=uri)
                logger.info(f"Spotify: playing '{query}'")
            else:
                self._client.start_playback(device_id=device_id)
                logger.info("Spotify: resumed playback")
            return True
        except Exception as e:
            logger.error(f"Spotify play failed: {e}")
            return False

    def pause(self) -> bool:
        """Pause playback."""
        try:
            self._client.pause_playback()
            logger.info("Spotify: paused")
            return True
        except Exception as e:
            logger.error(f"Spotify pause failed: {e}")
            return False

    def skip(self) -> bool:
        """Skip to the next track."""
        try:
            self._client.next_track()
            logger.info("Spotify: skipped to next")
            return True
        except Exception as e:
            logger.error(f"Spotify skip failed: {e}")
            return False

    def previous(self) -> bool:
        """Go back to the previous track."""
        try:
            self._client.previous_track()
            logger.info("Spotify: went to previous track")
            return True
        except Exception as e:
            logger.error(f"Spotify previous failed: {e}")
            return False

    def set_volume(self, percent: int) -> bool:
        """Set volume 0–100."""
        percent = max(0, min(100, percent))
        try:
            self._client.volume(percent)
            logger.info(f"Spotify: volume set to {percent}%")
            return True
        except Exception as e:
            logger.error(f"Spotify set_volume failed: {e}")
            return False

    def volume_up(self, step: int = 15) -> bool:
        """Increase volume by step percent."""
        current = self._current_volume()
        if current is None:
            return False
        return self.set_volume(min(100, current + step))

    def volume_down(self, step: int = 15) -> bool:
        """Decrease volume by step percent."""
        current = self._current_volume()
        if current is None:
            return False
        return self.set_volume(max(0, current - step))

    def now_playing(self) -> Optional[str]:
        """
        Returns the current track as "Song — Artist", or None if nothing is playing.
        """
        try:
            result = self._client.current_playback()
            if result and result.get("is_playing"):
                track  = result["item"]["name"]
                artist = result["item"]["artists"][0]["name"]
                return f"{track} by {artist}"
            return None
        except Exception as e:
            logger.error(f"Spotify now_playing failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _search(self, query: str) -> Optional[str]:
        """
        Search Spotify for a query. Returns the URI of the best match.
        Tries artist → track → playlist in order.
        """
        try:
            # Try artist first
            results = self._client.search(q=query, type="artist", limit=1)
            artists = results["artists"]["items"]
            if artists:
                return artists[0]["uri"]

            # Fall back to track
            results = self._client.search(q=query, type="track", limit=1)
            tracks = results["tracks"]["items"]
            if tracks:
                return tracks[0]["uri"]

            # Fall back to playlist
            results = self._client.search(q=query, type="playlist", limit=1)
            playlists = results["playlists"]["items"]
            if playlists:
                return playlists[0]["uri"]

        except Exception as e:
            logger.error(f"Spotify search failed: {e}")

        return None

    def _active_device(self) -> Optional[str]:
        """Return the ID of the currently active Spotify device, or None."""
        try:
            devices = self._client.devices()
            for d in devices.get("devices", []):
                if d["is_active"]:
                    return d["id"]
            # No active device — return first available
            available = devices.get("devices", [])
            if available:
                return available[0]["id"]
        except Exception as e:
            logger.debug(f"Could not get Spotify devices: {e}")
        return None

    def _current_volume(self) -> Optional[int]:
        """Return current playback volume 0–100, or None."""
        try:
            result = self._client.current_playback()
            if result and result.get("device"):
                return result["device"].get("volume_percent")
        except Exception:
            pass
        return None
