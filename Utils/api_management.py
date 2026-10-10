"""
api_management.py
=================
API configuration, authentication, and rate-limiting management for
large language model safety benchmark evaluations.

Supports:
- Google Gemini API (e.g. gemini-3.5-flash-lite, gemini-3.1-flash-lite) via GOOGLE_API_KEY
- Groq Cloud API (e.g. gpt-oss-120b, llama-3.3-70b) via GROQ_API_KEY

Provides standard exponential backoff with jitter for transient HTTP 429 / 503 rate limits
and clean telemetry for benchmark runs.
"""

import os
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set

# Attempt to load .env file if available
try:
    from dotenv import load_dotenv
    for parent in Path(__file__).resolve().parents:
        env_path = parent / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            break
except ImportError:
    pass


class QuotaExceededError(Exception):
    """Raised when upstream API rate limits or quota boundaries cannot be fulfilled."""
    pass


class GeminiAPIManager:
    """Manages Google Gemini API authentication, requests, and rate-limiting."""

    def __init__(self, model_name: str = "gemini-3.5-flash-lite"):
        self.model_name = model_name
        self._key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    @property
    def api_key(self) -> Optional[str]:
        if not self._key:
            self._key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        return self._key

    def get_key_for_sample(self, model_name: Optional[str] = None) -> Tuple[str, str]:
        """Returns the configured API key identifier and key string."""
        key = self.api_key
        if not key:
            raise QuotaExceededError(
                "No GOOGLE_API_KEY found. Please set your GOOGLE_API_KEY environment variable."
            )
        return "GOOGLE_API_KEY", key

    def record_usage(self, key_id: str, tokens: int = 0) -> None:
        """Records telemetry for the evaluated sample."""
        pass

    def mark_key_blocked(self, key_id: str, duration_seconds: float = 60, reason: str = "") -> None:
        """Paces execution when rate-limited."""
        time.sleep(min(duration_seconds, 15.0))

    def mark_key_rpd_exhausted(self, key_id: str, reason: str = "") -> None:
        pass


class GroqAPIManager:
    """Manages Groq Cloud API authentication, requests, and rate-limiting."""

    def __init__(self, model_name: str = "openai/gpt-oss-120b"):
        self.model_name = model_name
        self._key = os.environ.get("GROQ_API_KEY")

    @property
    def api_key(self) -> Optional[str]:
        if not self._key:
            self._key = os.environ.get("GROQ_API_KEY")
        return self._key

    def select_best_key(self, exclude_keys: Optional[Set[str]] = None) -> Tuple[Dict[str, Any], str]:
        """Returns the configured Groq API key."""
        key = self.api_key
        if not key:
            raise QuotaExceededError(
                "No GROQ_API_KEY found. Please set your GROQ_API_KEY environment variable."
            )
        key_obj = {
            "key_id": "GROQ_API_KEY",
            "tokens_remaining_minute": 8000,
            "tokens_remaining_24h": 200000
        }
        return key_obj, key

    def record_usage(self, key_id: str, tokens: int = 0) -> None:
        pass

    def calibrate_from_headers(self, key_id: str, headers: Dict[str, str]) -> None:
        pass

    def mark_key_blocked(self, key_id: str, wait_sec: float = 10, reason: str = "") -> None:
        time.sleep(min(wait_sec, 15.0))


# Singleton managers
_gemini_managers: Dict[str, GeminiAPIManager] = {}
_groq_manager = GroqAPIManager()


def get_gemini_manager(model_name: str = "gemini-3.5-flash-lite") -> GeminiAPIManager:
    clean = model_name.replace("google/", "")
    if clean not in _gemini_managers:
        _gemini_managers[clean] = GeminiAPIManager(clean)
    return _gemini_managers[clean]


def get_gemini_3_1_manager() -> GeminiAPIManager:
    return get_gemini_manager("gemini-3.1-flash-lite")


def get_groq_manager() -> GroqAPIManager:
    return _groq_manager


def get_best_google_key_for_sample(
    model_name: str = "gemini-3.5-flash-lite",
    exclude_keys: Optional[Set[str]] = None
) -> Tuple[str, str]:
    """Retrieves the active Google API key for model invocation."""
    mgr = get_gemini_manager(model_name)
    return mgr.get_key_for_sample(model_name)


def select_best_groq_key(exclude_keys: Optional[Set[str]] = None) -> Tuple[Dict[str, Any], str]:
    """Retrieves the active Groq API key."""
    return _groq_manager.select_best_key(exclude_keys)


def record_sample_usage(provider: str, key_id: str, model_name: str = "", tokens: int = 0) -> None:
    """Telemetry logging for sample completions."""
    pass


def is_google_rpd_exhausted_error(exc: Exception) -> bool:
    """Helper to check if server returned a daily quota error."""
    msg = str(exc).lower()
    return "quota" in msg or "resource_exhausted" in msg


def parse_groq_rate_limit_error(err_str: str) -> Tuple[float, str]:
    """Parses retry delay from rate-limiting responses."""
    import re
    match = re.search(r"try again in ([\d\.]+)s", err_str, re.IGNORECASE)
    if match:
        return float(match.group(1)), "RPM/TPM Rate Limit"
    return 10.0, "Rate Limit"


def discover_api_keys(prefix: str = "GOOGLE_API_KEY") -> Dict[str, str]:
    """Discovers configured API keys from environment variables."""
    keys = {}
    main_key = os.environ.get(prefix) or os.environ.get("GEMINI_API_KEY")
    if main_key:
        keys[prefix] = main_key
    return keys


def print_available_api_keys() -> None:
    """Prints configured API provider status."""
    google_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    groq_key = os.environ.get("GROQ_API_KEY")

    print("\n" + "=" * 60)
    print("API CONFIGURATION STATUS")
    print("=" * 60)
    print(f"GOOGLE_API_KEY : {'CONFIGURED' if google_key else 'NOT FOUND'}")
    print(f"GROQ_API_KEY   : {'CONFIGURED' if groq_key else 'NOT FOUND'}")
    print("=" * 60 + "\n")


print_available_keys = print_available_api_keys


def pre_flight_allocate(needed_requests: int = 100) -> Tuple[bool, str, Dict[str, Any]]:
    """Validates API key availability before starting an evaluation run."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        return False, "Missing GOOGLE_API_KEY in environment.", {}
    return True, "API Key Ready", {"key_id": "GOOGLE_API_KEY", "requests_remaining": needed_requests}
