"""ZTF light-curve download via the IRSA ZTF-LC-API.

Each source is a single HTTP request to the IRSA light-curve service
(``nph_light_curves``).  The :class:`DownloadRunner` parallelises these
requests and shares one HTTP connection pool, which is enough for
samples of up to a few thousand sources.

For very large samples (> 10^4 sources) consider the IRSA ZTF HATS /
Parquet bulk products instead (see docs/generated/download.md).
"""
from __future__ import annotations

import io
import logging
from typing import List, Optional

import pandas as pd

from ..base import FetchContext, ItemResult, SurveyArchive

logger = logging.getLogger("easycat.download")

ZTF_LC_URL = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"


class ZTFArchive(SurveyArchive):
    """ZTF light-curve downloader (per-position cone query).

    Parameters
    ----------
    bands : str
        Comma-separated filter list, e.g. ``"g,r,i"``.
    radius_arcsec : float
        Cone-search radius in arcseconds.
    store_format : str
        ``"csv"`` (default) or ``"fits"``.
    max_objects : int or None
        If set, keep only the ``max_objects`` closest matching ZTF objects
        per source (the API returns all objects inside the cone).
    """

    name = "ztf"
    default_batch_size = 1

    def __init__(
        self,
        *,
        bands: str = "g,r,i",
        radius_arcsec: float = 3.0,
        store_format: str = "csv",
        max_objects: Optional[int] = None,
    ):
        super().__init__(
            bands=bands,
            radius_arcsec=radius_arcsec,
            store_format=store_format,
            max_objects=max_objects,
        )
        self.bands = bands
        self.radius_arcsec = float(radius_arcsec)
        self.store_format = store_format
        self.max_objects = max_objects

    # ------------------------------------------------------------------ #
    # SurveyArchive
    # ------------------------------------------------------------------ #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        results: List[ItemResult] = []
        for _, row in rows.iterrows():
            results.append(self._fetch_one(row, ctx))
        return results

    def _fetch_one(self, row: pd.Series, ctx: FetchContext) -> ItemResult:
        obj_id = ctx.row_id(row)
        ra, dec = ctx.row_coord(row)
        radius_deg = self.radius_arcsec / 3600.0

        url = (
            f"{ZTF_LC_URL}?POS=CIRCLE+{ra:.7f}+{dec:.7f}+{radius_deg:.7f}"
            f"&BANDNAME={self.bands}&FORMAT=CSV"
        )

        try:
            resp = ctx.client.get(url)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            return ItemResult(obj_id=obj_id, success=False, error=repr(exc))

        if df is None or len(df) == 0:
            # Query succeeded but no ZTF light curve within the cone.
            return ItemResult(obj_id=obj_id, success=True, data=None)

        if self.max_objects is not None:
            # Keep the closest objects (light curves are ordered by the
            # server with the closest match first).
            df = df.head(self.max_objects)

        try:
            out_path = ctx.store_dir / f"{obj_id}.{self.store_format}"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if self.store_format == "csv":
                df.to_csv(out_path, index=False)
            elif self.store_format == "fits":
                from astropy.table import Table

                Table.from_pandas(df).write(out_path, overwrite=True)
            else:
                raise ValueError(f"Unknown store_format: {self.store_format}")
        except Exception as exc:
            return ItemResult(obj_id=obj_id, success=False, error=repr(exc))

        return ItemResult(obj_id=obj_id, success=True, data=df)
