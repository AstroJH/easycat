"""Pan-STARRS photometry downloader (MAST API).

Per-position cone queries against the MAST Pan-STARRS catalog API.
The :class:`DownloadRunner` parallelises the queries and reuses one HTTP
connection pool.
"""
from __future__ import annotations

import io
import logging
from typing import List

import pandas as pd

from ..base import FetchContext, ItemResult, SurveyArchive

logger = logging.getLogger("easycat.download")

BASEURL = "https://catalogs.mast.stsci.edu/api/v0.1/panstarrs"

FILTER_ID_MAP = {"g": 1, "r": 2, "i": 3, "z": 4, "y": 5, "all": None}


class PanSTARRSArchive(SurveyArchive):
    """Pan-STARRS photometry downloader.

    Parameters
    ----------
    band : str
        ``"g"``, ``"r"``, ``"i"``, ``"z"``, ``"y"`` or ``"all"``.
    radius_arcsec : float
        Cone-search radius.
    release : str
        ``"dr1"`` or ``"dr2"``.
    table : str
        ``"mean"``, ``"stack"``, ``"detection"`` or ``"forced_mean"``.
    store_format : str
        ``"csv"`` (default) or ``"fits"``.
    """

    name = "panstarrs"
    default_batch_size = 1

    def __init__(
        self,
        *,
        band: str = "all",
        radius_arcsec: float = 2.0,
        release: str = "dr2",
        table: str = "detection",
        store_format: str = "csv",
    ):
        if band not in FILTER_ID_MAP:
            raise ValueError(f"unknown band: {band!r}")
        super().__init__(
            band=band, radius_arcsec=radius_arcsec, release=release,
            table=table, store_format=store_format,
        )
        self.band = band
        self.radius_arcsec = float(radius_arcsec)
        self.release = release
        self.table = table
        self.store_format = store_format

    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        results: List[ItemResult] = []
        for _, row in rows.iterrows():
            results.append(self._fetch_one(row, ctx))
        return results

    def _fetch_one(self, row: pd.Series, ctx: FetchContext) -> ItemResult:
        obj_id = ctx.row_id(row)
        ra, dec = ctx.row_coord(row)
        radius_deg = self.radius_arcsec / 3600.0
        filter_id = FILTER_ID_MAP[self.band]

        url = (
            f"{BASEURL}/{self.release}/{self.table}.csv"
            f"?ra={ra:.7f}&dec={dec:.7f}&radius={radius_deg:.7f}"
        )
        if filter_id is not None:
            url += f"&filterID={filter_id}"

        try:
            resp = ctx.client.get(url)
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as exc:
            return ItemResult(obj_id=obj_id, success=False, error=repr(exc))

        if df is None or len(df) == 0:
            return ItemResult(obj_id=obj_id, success=True, data=None)

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
