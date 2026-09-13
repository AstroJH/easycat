"""MUSE (ESO) product downloads.

ESO products are identified by a stable ``dp_id`` such as
``ADP.2024-04-30T18:20:44.624``.  ESO exposes a direct dataportal URL for
each product; no product-specific URL construction is needed::

    https://dataportal.eso.org/dataPortal/file/<dp_id>

The dataportal Range service is unreliable beyond roughly 0.8 GB, so this
archive deliberately defaults to ``resume="never"``.  Atomic ``.part``
handling, retries, size verification, progress and optional checksums still
come from :meth:`HttpClient.download_file`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import FetchContext, ItemResult, SurveyArchive


DATAPORTAL_URL = "https://dataportal.eso.org/dataPortal/file"

# dp_id values are archive identifiers, not arbitrary paths.  Keep the
# character set deliberately narrow so a caller cannot turn an id into a path
# traversal or URL-injection vector.
_SAFE_DP_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")

MUSE_MODES = ("cube", "whitelight", "exmap", "any")


def valid_dp_id(dp_id: str) -> bool:
    """Return whether ``dp_id`` is a plausible, path-safe ESO identifier."""
    return bool(dp_id) and _SAFE_DP_ID.fullmatch(str(dp_id)) is not None


def dataportal_url(dp_id: str) -> str:
    """Return the direct ESO dataportal download URL for ``dp_id``."""
    if not valid_dp_id(dp_id):
        raise ValueError(f"invalid ESO dp_id: {dp_id!r}")
    return f"{DATAPORTAL_URL}/{dp_id}"


class MUSEArchive(SurveyArchive):
    """Download ESO Phase 3 MUSE products by ``dp_id``.

    Parameters
    ----------
    mode : {"cube", "whitelight", "exmap", "any"}
        Provenance label for the product being downloaded.  ESO identifies
        every product by its own ``dp_id``; this option does not alter the
        URL, but is recorded in ``ItemResult.meta["product"]``.
    dataportal_base : str
        Base URL of the ESO dataportal file service.  Primarily useful for
        mirrors and offline tests.
    download_kwargs : dict or None
        Defaults forwarded to :meth:`HttpClient.download_file`.  The archive
        sets ``resume="never"`` and ``validate="fits"``; caller-provided
        values override these defaults.

    Notes
    -----
    This class only downloads a known product identifier.  Target-name or
    coordinate discovery through ESO ObsCore is deliberately outside the
    Archive; resolve those queries to a ``dp_id`` first, then call
    :meth:`fetch_one` or use this archive with :class:`DownloadRunner`.
    """

    name = "muse"
    default_batch_size = 1

    def __init__(
        self,
        *,
        mode: str = "cube",
        dataportal_base: str = DATAPORTAL_URL,
        download_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if mode not in MUSE_MODES:
            raise ValueError(
                f"mode must be one of {MUSE_MODES}, got {mode!r}"
            )

        defaults: Dict[str, Any] = {
            "resume": "never",
            "validate": "fits",
        }
        defaults.update(download_kwargs or {})
        super().__init__(
            download_kwargs=defaults,
            mode=mode,
            dataportal_base=dataportal_base.rstrip("/"),
        )
        self.mode = mode
        self.dataportal_base = dataportal_base.rstrip("/")

    # ------------------------------------------------------------------ #
    # URL / path helpers
    # ------------------------------------------------------------------ #
    def url_for(self, dp_id: str) -> str:
        """Return the direct ESO dataportal URL for ``dp_id``."""
        if not valid_dp_id(dp_id):
            raise ValueError(f"invalid ESO dp_id: {dp_id!r}")
        return f"{self.dataportal_base}/{dp_id}"

    def output_path(self, ctx: FetchContext, obj_id: str, *,
                    row: Optional[pd.Series] = None) -> Path:
        """Default output path: ``<store_dir>/<dp_id>.fits``."""
        return ctx.dest(obj_id, ctx.store_dir / f"{obj_id}.fits", row=row)

    def item_metadata(self, row: Optional[pd.Series] = None) -> Dict[str, Any]:
        meta = super().item_metadata(row)
        meta["product"] = f"muse-{self.mode}"
        meta["source"] = "ESO Phase 3"
        return meta

    # ------------------------------------------------------------------ #
    # SurveyArchive
    # ------------------------------------------------------------------ #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        results: List[ItemResult] = []
        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            try:
                url = self.url_for(obj_id)
                dest = self.output_path(ctx, obj_id, row=row)
                self._download_file(ctx, url, dest)
                result = ItemResult(obj_id=obj_id, success=True, data=dest)
                results.append(
                    self.enrich_result(result, row=row, url=url, dest=dest)
                )
            except Exception as exc:
                status = getattr(exc, "status", None)
                if status is None:
                    status = getattr(
                        getattr(exc, "response", None), "status_code", None,
                    )
                url = None
                try:
                    url = self.url_for(obj_id)
                except Exception:
                    pass
                if status == 404:
                    # A valid dp_id can disappear or not be public yet; treat
                    # that like the other archives' "no data" result.
                    result = ItemResult(obj_id=obj_id, success=True, data=None)
                else:
                    result = ItemResult(
                        obj_id=obj_id, success=False, error=repr(exc),
                    )
                results.append(self.enrich_result(result, row=row, url=url))
        return results


__all__ = [
    "MUSEArchive",
    "DATAPORTAL_URL",
    "MUSE_MODES",
    "dataportal_url",
    "valid_dp_id",
]
