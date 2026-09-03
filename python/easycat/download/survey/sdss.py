"""SDSS photometry / spectra downloader.

Uses the SkyServer Cross-ID web service (via :mod:`astroquery.sdss`) to
cross-match a batch of coordinates in a *single* request, then downloads
the individual ``lite`` spectra from the Science Archive Server (SAS) by
``survey / plate / mjd / fiber``.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
from astroquery.sdss import SDSS

from ..base import FetchContext, ItemResult, SurveyArchive

logger = logging.getLogger("easycat.download")

SAS = "https://data.sdss.org/sas/dr18"

# survey (as returned by Cross-ID) -> SAS prefix for the lite spectra.
SURVEY_PATHS: Dict[str, List[str]] = {
    "sdss": [
        f"{SAS}/spectro/sdss/redux/26/spectra/lite",
        f"{SAS}/spectro/sdss/redux/26/spectra/full",
    ],

    "boss": [
        f"{SAS}/spectro/boss/redux/v6_0_4/spectra/lite",
        f"{SAS}/spectro/boss/redux/v6_0_4/spectra/full",
    ],

    "eboss": [
        f"{SAS}/prior-surveys/sdss4-dr17-eboss/spectro/redux/v5_13_2/spectra/lite",
        f"{SAS}/prior-surveys/sdss4-dr17-eboss/spectro/redux/v5_13_2/spectra/full",
        f"{SAS}/spectro/boss/redux/v6_0_4/spectra/lite",
    ],
}

PHOTO_COLUMNS = ["OBJID", "RA", "DEC", "U", "G", "R", "I", "Z", "TYPE"]

# The SkyServer Cross-ID tool is not meant to be hammered concurrently;
# serialise the cross-match step (spectrum downloads still parallelise).
_CROSSID_LOCK = threading.Lock()


class SDSSArchive(SurveyArchive):
    """SDSS downloader.

    Parameters
    ----------
    mode : str
        ``"spectra"``, ``"photometry"`` or ``"both"``.
    radius_arcsec : float
        Cross-match radius (the SkyServer Cross-ID limit is 3 arcmin).
    """

    name = "sdss"
    default_batch_size = 32

    def __init__(self, *, mode: str = "both", radius_arcsec: float = 3.0):
        if mode not in ("spectra", "photometry", "both"):
            raise ValueError(f"unknown mode: {mode!r}")
        super().__init__(mode=mode, radius_arcsec=radius_arcsec)
        self.mode = mode
        self.radius_arcsec = float(radius_arcsec)

    # ------------------------------------------------------------------ #
    # SurveyArchive
    # ------------------------------------------------------------------ #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        coords = SkyCoord(
            [float(r[ctx.ra_column]) for _, r in rows.iterrows()],
            [float(r[ctx.dec_column]) for _, r in rows.iterrows()],
            unit="deg",
        )

        last_exc: Optional[Exception] = None
        xid = None

        for attempt in range(5):
            try:
                with _CROSSID_LOCK:
                    xid = SDSS.query_crossid(
                        coords,
                        radius=self.radius_arcsec * u.arcsec,
                        photoobj_fields=["ra", "dec", "u", "g", "r", "i", "z", "type"],
                        specobj_fields=["ra", "dec", "plate", "mjd", "fiberid", "survey", "class"],
                        cache=False,
                    )
                break
            except Exception as exc:
                # transient SSL/server errors -> retry
                last_exc = exc
                time.sleep(2.0 * (attempt + 1)) # linear backoff
    
        if xid is None:
            return [
                ItemResult(obj_id=ctx.row_id(row), success=False, error=repr(last_exc))
                for _, row in rows.iterrows()
            ]

        results: List[ItemResult] = []
        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            ra, dec = ctx.row_coord(row)
            matches = self._match(xid, ra, dec)

            if matches is None or len(matches) == 0:
                results.append(ItemResult(obj_id=obj_id, success=True, data=None))
                continue
            best = matches[0]

            try:
                out = self._store(ctx, obj_id, best)
                results.append(ItemResult(obj_id=obj_id, success=True, data=out))
            except Exception as exc:
                results.append(ItemResult(obj_id=obj_id, success=False, error=repr(exc)))
                
        return results

    def _match(self, xid: Table, ra: float, dec: float) -> Optional[Table]:
        """Return cross-ID rows within the matching radius, closest first."""
        if xid is None or len(xid) == 0:
            return None
        
        mra = np.asarray(xid["ra"], float)
        mdec = np.asarray(xid["dec"], float)
        target = SkyCoord(ra, dec, unit="deg")
        coords = SkyCoord(mra, mdec, unit="deg")
        sep = target.separation(coords).to_value(u.arcsec)
        mask = sep <= self.radius_arcsec

        if not np.any(mask):
            return None
        
        order = np.argsort(sep[mask])
        return xid[mask][order]

    def _store(self, ctx: FetchContext, obj_id: str, row) -> Path:
        out = ctx.store_dir / f"{obj_id}.fits"
        out.parent.mkdir(parents=True, exist_ok=True)

        plate = int(row["plate"])
        mjd = int(row["mjd"])
        fiber = int(row["fiberid"])
        survey = str(row["survey"]).lower()

        if self.mode == "spectra":
            if not self._download_spectrum(ctx, out, plate, mjd, fiber, survey):

                raise RuntimeError(
                    f"spectrum download failed for plate={plate} mjd={mjd} "
                    f"fiber={fiber} survey={survey}"
                )
            
            return out

        if self.mode == "both":
            if self._download_spectrum(ctx, out, plate, mjd, fiber, survey):
                return out
            
            logger.warning(
                "%s: no spectrum, storing photometry only", obj_id
            )

        self._write_photometry(ctx, obj_id, row, out)
        return out

    def _download_spectrum(
        self, ctx: FetchContext, dest: Path, plate: int, mjd: int, fiber: int, survey: str
    ) -> bool:
        fname = f"spec-{plate}-{mjd}-{fiber:04d}.fits"
        candidates = SURVEY_PATHS.get(survey, [])
        for prefix in candidates:
            url = f"{prefix}/{plate}/{fname}"
            try:
                ctx.client.download_file(url, dest, overwrite=True)
                return True
            except Exception:
                continue
        return False

    def _write_photometry(self, ctx: FetchContext, obj_id: str, row, out: Path) -> None:
        data = {
            "OBJID": [int(row["objID"])] if "objID" in row.colnames else [np.nan],
            "RA": [float(row["ra"])],
            "DEC": [float(row["dec"])],
            "TYPE": [str(row["type"])] if "type" in row.colnames else [""],
        }

        for band, col in (("U", "u"), ("G", "g"), ("R", "r"), ("I", "i"), ("Z", "z")):
            if col in row.colnames:
                data[band] = [float(row[col]) if row[col] == row[col] else np.nan]
                
        t = Table(data)
        t.meta = {}
        t.write(out, overwrite=True)
