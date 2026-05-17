"""
Sterling Weather Utility — wttr.in
===================================
Fetches current conditions and a two-day forecast from wttr.in.
No API key, no account, completely free.
"""

import requests
from utils.logger import setup_logger

logger = setup_logger("sterling.weather")

WTTR_URL = "https://wttr.in/{location}?format=j1"


def _max_chance(hourly: list, key: str) -> int:
    """Return the highest percentage across all hourly slots for a given condition."""
    try:
        return max(int(h.get(key, 0)) for h in hourly)
    except Exception:
        return 0


def _midday_desc(hourly: list) -> str:
    """Return the weather description closest to midday (index 4 = noon slot)."""
    try:
        return hourly[4]["weatherDesc"][0].get("value", "")
    except Exception:
        return ""


def get_weather_context(location: str) -> str:
    """
    Fetch current weather and two-day forecast for a location.
    Returns a structured plain-English string for injecting into the LLM message.

    Args:
        location: e.g. "Superior, Colorado"

    Returns:
        A detailed weather summary string, or empty string on failure.
    """
    try:
        url = WTTR_URL.format(location=requests.utils.quote(location))
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        # ── Current conditions ────────────────────────────────────────────────
        current = data["current_condition"][0]
        temp_f      = current.get("temp_F", "?")
        feels_f     = current.get("FeelsLikeF", "?")
        description = current["weatherDesc"][0].get("value", "unknown")
        humidity    = current.get("humidity", "?")
        wind_mph    = current.get("windspeedMiles", "?")
        wind_dir    = current.get("winddir16Point", "")

        # ── Today's forecast ──────────────────────────────────────────────────
        today        = data["weather"][0]
        high_f       = today.get("maxtempF", "?")
        low_f        = today.get("mintempF", "?")
        rain_pct     = _max_chance(today["hourly"], "chanceofrain")
        snow_pct     = _max_chance(today["hourly"], "chanceofsnow")
        today_desc   = _midday_desc(today["hourly"])

        # ── Tomorrow's forecast ───────────────────────────────────────────────
        tomorrow       = data["weather"][1]
        tom_high_f     = tomorrow.get("maxtempF", "?")
        tom_low_f      = tomorrow.get("mintempF", "?")
        tom_rain_pct   = _max_chance(tomorrow["hourly"], "chanceofrain")
        tom_snow_pct   = _max_chance(tomorrow["hourly"], "chanceofsnow")
        tom_desc       = _midday_desc(tomorrow["hourly"])

        # ── Build precipitation notes ─────────────────────────────────────────
        def precip_note(rain: int, snow: int) -> str:
            notes = []
            if snow >= 20:
                notes.append(f"{snow}% chance of snow")
            if rain >= 20:
                notes.append(f"{rain}% chance of rain")
            return ", ".join(notes) if notes else "no significant precipitation expected"

        today_precip    = precip_note(rain_pct, snow_pct)
        tomorrow_precip = precip_note(tom_rain_pct, tom_snow_pct)

        summary = (
            f"Weather data for {location} — "
            f"Right now: {description}, {temp_f}°F (feels like {feels_f}°F), "
            f"humidity {humidity}%, wind {wind_mph} mph {wind_dir}. "
            f"Today: high {high_f}°F, low {low_f}°F, {today_desc.lower() or description.lower()}, {today_precip}. "
            f"Tomorrow: high {tom_high_f}°F, low {tom_low_f}°F, {tom_desc.lower()}, {tomorrow_precip}."
        )

        logger.debug(f"Weather fetched: {summary}")
        return summary

    except requests.exceptions.Timeout:
        logger.warning("Weather fetch timed out.")
        return ""
    except Exception as e:
        logger.warning(f"Weather fetch failed: {e}")
        return ""
