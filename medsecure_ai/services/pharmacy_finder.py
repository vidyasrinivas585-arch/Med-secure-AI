"""
services/pharmacy_finder.py
Google Places API integration for nearby licensed pharmacy search.
Supports geolocation-based nearby search and manual city/PIN/area lookup.
"""

import os
import logging
import math
from urllib.parse import urlencode

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
DEFAULT_RADIUS_KM = float(os.getenv("PHARMACY_SEARCH_RADIUS_KM", "5"))
PLACES_NEARBY_URL = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
PLACES_TEXT_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
GEOCODE_URL = "https://maps.googleapis.com/maps/api/geocode/json"
PLACE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance between two coordinates in kilometres."""
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return round(r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 2)


def _format_pharmacy(place: dict, origin_lat: float, origin_lng: float) -> dict:
    """Normalise a Google Places result into our pharmacy schema."""
    loc = place.get("geometry", {}).get("location", {})
    plat = loc.get("lat", 0)
    plng = loc.get("lng", 0)
    distance = _haversine_km(origin_lat, origin_lng, plat, plng)

    opening = place.get("opening_hours", {})
    is_open = opening.get("open_now")

    return {
        "name": place.get("name", "Unknown Pharmacy"),
        "address": place.get("vicinity") or place.get("formatted_address", "Address unavailable"),
        "rating": place.get("rating"),
        "user_ratings_total": place.get("user_ratings_total", 0),
        "distance_km": distance,
        "lat": plat,
        "lng": plng,
        "place_id": place.get("place_id", ""),
        "is_open": is_open,
        "opening_status": (
            "Open now" if is_open is True
            else "Closed now" if is_open is False
            else "Hours unknown"
        ),
        "phone": None,  # populated via details API when needed
        "maps_url": f"https://www.google.com/maps/search/?api=1&query={plat},{plng}&query_place_id={place.get('place_id', '')}",
    }


def _fetch_phone_number(place_id: str) -> str | None:
    """Fetch formatted phone number for a place (optional enrichment)."""
    if not place_id or not GOOGLE_MAPS_API_KEY:
        return None
    try:
        params = {
            "place_id": place_id,
            "fields": "formatted_phone_number",
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(PLACE_DETAILS_URL, params=params, timeout=8)
        data = resp.json()
        if data.get("status") == "OK":
            return data.get("result", {}).get("formatted_phone_number")
    except Exception as e:
        logger.warning(f"Phone lookup failed for {place_id}: {e}")
    return None


def geocode_location(query: str) -> tuple[float, float] | None:
    """
    Convert a city, PIN code, or area name to lat/lng coordinates.

    Args:
        query: Free-text location (e.g. 'Bengaluru 560001', 'MG Road Bangalore').

    Returns:
        (lat, lng) tuple or None on failure.
    """
    if not GOOGLE_MAPS_API_KEY:
        logger.error("GOOGLE_MAPS_API_KEY not configured.")
        return None
    try:
        params = {"address": query, "key": GOOGLE_MAPS_API_KEY}
        resp = requests.get(GEOCODE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") != "OK" or not data.get("results"):
            logger.warning(f"Geocode failed for '{query}': {data.get('status')}")
            return None
        loc = data["results"][0]["geometry"]["location"]
        return loc["lat"], loc["lng"]
    except Exception as e:
        logger.error(f"Geocode error: {e}")
        return None


def find_nearby_pharmacies(
    lat: float,
    lng: float,
    radius_km: float | None = None,
    enrich_phones: bool = False,
) -> dict:
    """
    Search for licensed pharmacies near the given coordinates.

    Args:
        lat, lng: User coordinates.
        radius_km: Search radius in km (default from env, typically 5).
        enrich_phones: Whether to call Place Details for phone numbers.

    Returns:
        dict with success flag, pharmacies list, and optional error message.
    """
    if not GOOGLE_MAPS_API_KEY:
        return {
            "success": False,
            "pharmacies": [],
            "error": "Google Maps API key is not configured. Please set GOOGLE_MAPS_API_KEY in .env",
        }

    radius_m = int((radius_km or DEFAULT_RADIUS_KM) * 1000)

    try:
        params = {
            "location": f"{lat},{lng}",
            "radius": radius_m,
            "type": "pharmacy",
            "keyword": "licensed pharmacy chemist medical store",
            "key": GOOGLE_MAPS_API_KEY,
        }
        resp = requests.get(PLACES_NEARBY_URL, params=params, timeout=12)
        data = resp.json()

        status = data.get("status", "UNKNOWN")
        if status not in ("OK", "ZERO_RESULTS"):
            return {
                "success": False,
                "pharmacies": [],
                "error": f"Google Places API error: {status}",
            }

        raw_places = data.get("results", [])
        pharmacies = [_format_pharmacy(p, lat, lng) for p in raw_places]

        # Sort by distance ascending
        pharmacies.sort(key=lambda p: p["distance_km"])

        # Optionally enrich top results with phone numbers (limit API calls)
        if enrich_phones:
            for pharmacy in pharmacies[:8]:
                phone = _fetch_phone_number(pharmacy["place_id"])
                pharmacy["phone"] = phone

        return {
            "success": True,
            "pharmacies": pharmacies,
            "count": len(pharmacies),
            "origin": {"lat": lat, "lng": lng},
            "radius_km": radius_km or DEFAULT_RADIUS_KM,
            "error": None,
        }

    except requests.RequestException as e:
        logger.error(f"Pharmacy search network error: {e}")
        return {
            "success": False,
            "pharmacies": [],
            "error": "Unable to reach Google Places API. Check your internet connection.",
        }
    except Exception as e:
        logger.error(f"Pharmacy search failed: {e}")
        return {
            "success": False,
            "pharmacies": [],
            "error": str(e),
        }


def search_pharmacies_by_location(
    location_query: str,
    radius_km: float | None = None,
    enrich_phones: bool = True,
) -> dict:
    """
    Geocode a manual location query then find nearby pharmacies.

    Args:
        location_query: City, PIN code, or area name.
        radius_km: Search radius in km.

    Returns:
        Same schema as find_nearby_pharmacies, plus geocoded address.
    """
    coords = geocode_location(location_query)
    if not coords:
        return {
            "success": False,
            "pharmacies": [],
            "error": f"Could not find location: '{location_query}'. Try a city name or PIN code.",
        }

    lat, lng = coords
    result = find_nearby_pharmacies(lat, lng, radius_km, enrich_phones)
    result["searched_location"] = location_query
    result["geocoded"] = {"lat": lat, "lng": lng}
    return result
