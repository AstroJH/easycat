"""Filter database facade: local store + SVO2 online access.

Responsibilities are separated: :class:`FilterStore` persists filters,
:mod:`easycat.astrofilter.svo2` fetches from SVO2, and :class:`FilterDB`
glues them together (memory -> local ECSV -> SVO2, with automatic caching).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..download.client import HttpClient
from . import svo2
from .core import AstroFilter
from .store import FilterStore

__all__ = ["FilterDB", "AstroFilterDB", "default_cache_dir"]


def default_cache_dir() -> Path:
    """Default filter cache directory (``$EASYCAT_DATA/filters`` or
    ``~/.easycat/filters``)."""
    base = os.environ.get("EASYCAT_DATA")
    if base:
        return Path(base) / "filters"

    # default
    return Path.home() / ".easycat" / "filters"


class FilterDB:
    """Retrieve, cache and search astronomical filters.

    Parameters
    ----------
    cache_dir : str or Path or None
        Directory for the ECSV filter cache (default:
        ``~/.easycat/filters`` or ``$EASYCAT_DATA/filters``).
    client : HttpClient or None
        Shared HTTP client (rate limiting / retries applied globally).
    """

    def __init__(self, cache_dir: Optional[Path] = None,
                 client: Optional[HttpClient] = None):
        self.store = FilterStore(
            Path(cache_dir)
            if cache_dir
            else default_cache_dir()
        )

        self.client = client or svo2._client()

    # ------------------------------------------------------------------ #
    def get(self, filter_id: str) -> Optional[AstroFilter]:
        """Get a filter: memory -> local ECSV -> SVO2 (then cached)."""
        fid = svo2.normalize_filter_id(filter_id)
        cached = self.store.load(fid)
        if cached is not None:
            return cached
        
        try:
            filt = svo2.fetch_filter(fid, client=self.client)
        except Exception as exc:
            logger = __import__("logging").getLogger("easycat.astrofilter")
            logger.warning("failed to fetch %s from SVO2: %s", fid, exc)
            return None
        filt.zp_vega  # compute once (online) so it is persisted below
        self.store.save(filt)
        return filt

    # alias kept from the original API
    get_filter = get

    def fetch(self, filter_id: str, *, refresh: bool = False) -> AstroFilter:
        """Fetch a filter from SVO2 (``refresh=True`` bypasses the cache)."""
        fid = svo2.normalize_filter_id(filter_id)
        if not refresh:
            cached = self.store.load(fid)
            if cached is not None:
                return cached
        filt = svo2.fetch_filter(fid, client=self.client)
        filt.zp_vega  # compute once (online) so it is persisted below
        self.store.save(filt)
        return filt

    def register(self, filter_id: str, filt: AstroFilter,
                 override: bool = False) -> None:
        """Register an in-memory / custom filter and persist it."""
        if self.store.contains(filter_id) and not override:
            raise KeyError(f"filter already cached: {filter_id}")
        filt.filter_id = svo2.normalize_filter_id(filter_id)
        self.store.save(filt)

    def search(self, query: str = "") -> List[Dict[str, Any]]:
        """Search local cache; also query SVO2 (best effort)."""
        results = self.store.search(query)
        if query:
            results.extend(svo2.search(query, client=self.client))
        # de-duplicate by filterID, keep local first
        seen = set()
        out = []
        for r in results:
            key = r.get("filterID") or r.get("filter_id")
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
        return out

    def list(self) -> List[str]:
        """IDs of all locally cached filters."""
        return self.store.list_ids()

    def clear_cache(self) -> None:
        """Clear the in-memory layer (files on disk are kept)."""
        self.store._memory.clear()

AstroFilterDB = FilterDB
