"""SDSS downloader: spectra, photometry, MaNGA IFU data and images.

Modes
-----
``"spectra"`` / ``"photometry"`` / ``"both"``
    SkyServer Cross-ID (batch) -> SAS ``lite`` spectra by
    ``survey/plate/mjd/fiber`` (photometry fallback).
``"manga"``
    MaNGA IFU data.  Targets are matched against the DRP summary catalogue
    (``drpall``, read lazily column-by-column) to obtain
    ``plate``/``ifudsgn``, then the requested product is fetched from SAS::

        DRP:  <sas>/manga/spectro/redux/<ver>/<plate>/stack/
                  manga-<plate>-<ifu>-<LOGRSS|LINRSS|LOGCUBE|LINCUBE>.fits.gz
        DAP:  <sas>/manga/spectro/analysis/<ver>/<dapver>/<DAPTYPE>/
                  <plate>/<ifu>/manga-<plate>-<ifu>-<MAPS|LOGCUBE>-<DAPTYPE>.fits.gz

``"image"``
    SkyServer image cutouts (JPEG, ``ImgCutout/getjpeg``).  Note that the
    service occasionally times out for large cutouts in some fields --
    reducing ``image_width``/``image_height`` (or increasing ``image_scale``)
    usually helps.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astroquery.sdss import SDSS

from ..base import FetchContext, ItemResult, SurveyArchive
from ..client import RangeFile

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

# ---- MaNGA (SDSS-IV IFU survey) ------------------------------------------ #
# MaNGA DRP v3_1_1 is served from the DR17 tree (DR18 has no manga/ path).
MANGA_SAS = "https://data.sdss.org/sas/dr17"
MANGA_DRPALL = f"{MANGA_SAS}/manga/spectro/redux/v3_1_1/drpall-v3_1_1.fits"

# DRP 3D products (per plate/ifudesign, under <plate>/stack/)
MANGA_DRP_PRODUCTS = ("LOGRSS", "LINRSS", "LOGCUBE", "LINCUBE")
# MaNGA DAP products (per plate/ifudesign, under <DAPTYPE>/<plate>/<ifu>/)
MANGA_DAP_PRODUCTS = ("MAPS", "LOGCUBE")
MANGA_DAPTYPES = ("SPX-MILESHC-MASTARSSP", "HYB10-MILESHC-MASTARSSP",
                  "HYB10-MILESHC-MASTARHC2", "VOR10-MILESHC-MASTARSSP")

# ---- SkyServer image cutouts -------------------------------------------- #
SKYSERVER = "https://skyserver.sdss.org/dr18"
IMGCUTOUT_URL = f"{SKYSERVER}/SkyServerWS/ImgCutout/getjpeg"

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

    def __init__(
        self,
        *,
        mode: str = "both",
        radius_arcsec: float = 3.0,
        # -- MaNGA (mode="manga") --
        manga_product: str = "LOGRSS",
        manga_dap: Optional[str] = None,
        manga_drpall: Optional[str] = None,
        manga_sas: str = MANGA_SAS,
        manga_version: str = "v3_1_1",
        manga_dap_version: str = "3.1.0",
        # -- images (mode="image") --
        image_width: int = 512,
        image_height: int = 512,
        image_scale: float = 0.3,
        image_opt: str = "",
        cache_dir: Optional[Path] = None,
        download_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if mode not in ("spectra", "photometry", "both", "manga", "image"):
            raise ValueError(
                "mode must be one of spectra/photometry/both/manga/image, "
                f"got {mode!r}"
            )
        super().__init__(
            download_kwargs=download_kwargs,
            mode=mode,
            radius_arcsec=radius_arcsec,
        )
        self.mode = mode
        self.radius_arcsec = float(radius_arcsec)

        # MaNGA options
        self.manga_product = str(manga_product).upper()
        self.manga_dap = str(manga_dap).upper() if manga_dap else None
        self.manga_sas = manga_sas.rstrip("/")
        self.manga_version = manga_version
        self.manga_dap_version = manga_dap_version
        # a local drpall file, or an explicit URL/path override
        self.manga_drpall = manga_drpall
        if mode == "manga":
            if self.manga_dap is None:
                if self.manga_product not in MANGA_DRP_PRODUCTS:
                    raise ValueError(
                        f"unknown MaNGA DRP product {manga_product!r} "
                        f"(available: {list(MANGA_DRP_PRODUCTS)})"
                    )
            else:
                if self.manga_dap not in MANGA_DAPTYPES:
                    raise ValueError(
                        f"unknown MaNGA DAPTYPE {manga_dap!r} "
                        f"(available: {list(MANGA_DAPTYPES)})"
                    )
                if self.manga_product not in MANGA_DAP_PRODUCTS:
                    raise ValueError(
                        f"unknown MaNGA DAP product {manga_product!r} "
                        f"(available: {list(MANGA_DAP_PRODUCTS)})"
                    )

        # image options
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.image_scale = float(image_scale)
        self.image_opt = image_opt

        self.cache_dir = Path(cache_dir) if cache_dir else None

        # cross-ID is batched; MaNGA/images are one request per source
        self.default_batch_size = 1 if mode in ("manga", "image") else 32

        self._manga_lock = threading.Lock()
        self._manga_index = None

    # ------------------------------------------------------------------ #
    # SurveyArchive
    # ------------------------------------------------------------------ #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        if self.mode == "manga":
            return self._fetch_manga_batch(rows, ctx)
        if self.mode == "image":
            return self._fetch_image_batch(rows, ctx)
        return self._fetch_crossid_batch(rows, ctx)

    # ------------------------------------------------------------------ #
    # spectra / photometry (SkyServer Cross-ID)
    # ------------------------------------------------------------------ #
    def output_path(self, ctx: FetchContext, obj_id: str, *,
                    row: Optional[pd.Series] = None) -> Path:
        """Per-source output file, depending on the mode."""
        if self.mode == "manga":
            return ctx.dest(
                obj_id, ctx.store_dir / f"{obj_id}.fits.gz", row=row,
            )
        if self.mode == "image":
            return ctx.dest(obj_id, ctx.store_dir / f"{obj_id}.jpg", row=row)
        return ctx.dest(  # spectra / photometry
            obj_id, ctx.store_dir / f"{obj_id}.fits", row=row,
        )

    def prepare_item_row(self, target: Any, row: pd.Series) -> pd.Series:
        """Allow MaNGA ``plateifu`` identifiers in ``fetch_one``."""
        if self.mode == "manga" and "plateifu" not in row.index:
            value = str(target)
            parts = value.split("-")
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                row["plateifu"] = value
        return row

    def item_metadata(self, row: Optional[pd.Series] = None) -> Dict[str, Any]:
        meta = super().item_metadata(row)
        if self.mode == "manga" and self.manga_dap:
            meta["product"] = f"{self.manga_product}-{self.manga_dap}"
        return meta

    def _fetch_crossid_batch(self, rows: pd.DataFrame,
                             ctx: FetchContext) -> List[ItemResult]:
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
                out = self._store(ctx, obj_id, best, source_row=row)
                result = ItemResult(obj_id=obj_id, success=True, data=out)
                results.append(self.enrich_result(result, row=row, dest=out))
            except Exception as exc:
                result = ItemResult(obj_id=obj_id, success=False, error=repr(exc))
                results.append(self.enrich_result(result, row=row))
                
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

    def _store(self, ctx: FetchContext, obj_id: str, row,
               *, source_row: Optional[pd.Series] = None) -> Path:
        out = ctx.dest(
            obj_id, ctx.store_dir / f"{obj_id}.fits", row=source_row,
        )
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
                self._download_file(ctx, url, dest)
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

    # ------------------------------------------------------------------ #
    # MaNGA IFU data (mode="manga")
    # ------------------------------------------------------------------ #
    def _cache_root(self, ctx: FetchContext) -> Path:
        """Where the drpall cache lives (path only -- nothing is created).

        The directory is materialised by ``HttpClient.download_file()`` when
        the index is actually needed, so merely asking for the path (e.g. to
        report it) leaves no empty directory behind.
        """
        return self.cache_dir or (ctx.store_dir / "_sdss_cache")

    MANGA_INDEX_COLUMNS = ("plate", "ifudsgn", "plateifu", "objra", "objdec")

    def manga_index(self, ctx: FetchContext) -> Dict[str, np.ndarray]:
        """Lazy MaNGA DRP summary (``drpall``) index.

        ``drpall`` is a wide table (~75 MB); individual columns cannot be
        read cheaply by byte range, so it is downloaded **once** into the
        cache directory and read locally afterwards.  Use ``manga_drpall``
        to point at an existing local file (or a different URL) instead.
        """
        if self._manga_index is not None:
            return self._manga_index
        with self._manga_lock:
            if self._manga_index is not None:
                return self._manga_index

            local = None
            if self.manga_drpall and Path(self.manga_drpall).exists():
                local = Path(self.manga_drpall)
            if local is None:
                cache = self._cache_root(ctx)
                local = cache / f"drpall-{self.manga_version}.fits"
                if not local.exists():
                    url = self.manga_drpall or (
                        f"{self.manga_sas}/manga/spectro/redux/"
                        f"{self.manga_version}/drpall-{self.manga_version}.fits"
                    )
                    logger.info("downloading MaNGA drpall (%.0f MB) -> %s",
                                75, local)
                    ctx.client.download_file(url, local)

            with fits.open(local, memmap=False) as hdul:
                data = hdul[1].data
                self._manga_index = {
                    c: np.asarray(data[c]) for c in self.MANGA_INDEX_COLUMNS
                }
        return self._manga_index

    def manga_url(self, plate: int, ifudesign: int, *,
                  product: Optional[str] = None,
                  dap: Optional[str] = None) -> str:
        """SAS URL of one MaNGA product (DRP or DAP)."""
        product = (product or self.manga_product).upper()
        dap = (dap if dap is not None else self.manga_dap)
        if dap:
            dap = dap.upper()
            return (f"{self.manga_sas}/manga/spectro/analysis/"
                    f"{self.manga_version}/{self.manga_dap_version}/{dap}/"
                    f"{plate}/{ifudesign}/"
                    f"manga-{plate}-{ifudesign}-{product}-{dap}.fits.gz")
        return (f"{self.manga_sas}/manga/spectro/redux/{self.manga_version}/"
                f"{plate}/stack/manga-{plate}-{ifudesign}-{product}.fits.gz")

    @staticmethod
    def _plate_ifu_from_row(row: pd.Series) -> Optional[Tuple[int, int]]:
        """Read ``(plate, ifudesign)`` from a catalog row when available.

        Avoids downloading the 75 MB ``drpall`` when the input catalog
        already carries MaNGA identifiers (``plate``+``ifudsgn`` or
        ``plateifu`` like ``"8138-12704"``).
        """
        for plate_col, ifu_col in (("plate", "ifudsgn"), ("plate", "ifudesign"),
                                   ("PLATE", "IFUDSGN")):
            if plate_col in row.index and ifu_col in row.index:
                try:
                    return int(row[plate_col]), int(row[ifu_col])
                except (TypeError, ValueError):
                    pass
        for col in ("plateifu", "PLATEIFU", "obj_id"):
            if col in row.index:
                s = str(row[col])
                if "-" in s:
                    a, b = s.split("-")[:2]
                    try:
                        return int(a), int(b)
                    except ValueError:
                        pass
        return None

    def _fetch_manga_batch(self, rows: pd.DataFrame,
                           ctx: FetchContext) -> List[ItemResult]:
        # the drpall index is only needed for rows without MaNGA ids
        idx = None
        if any(self._plate_ifu_from_row(row) is None
               for _, row in rows.iterrows()):
            idx = self.manga_index(ctx)
            coords = SkyCoord(idx["objra"], idx["objdec"], unit="deg")

        results: List[ItemResult] = []

        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            plate_ifu = self._plate_ifu_from_row(row)

            if plate_ifu is None:
                ra, dec = ctx.row_coord(row)
                target = SkyCoord(ra, dec, unit="deg")
                sep = target.separation(coords).to_value(u.arcsec)
                j = int(np.argmin(sep)) if len(sep) else -1
                if j < 0 or sep[j] > self.radius_arcsec:
                    results.append(ItemResult(obj_id=obj_id, success=True,
                                              data=None))
                    continue
                plate, ifu = int(idx["plate"][j]), int(idx["ifudsgn"][j])
            else:
                plate, ifu = plate_ifu
            url = self.manga_url(plate, ifu)
            dest = ctx.dest(
                obj_id, ctx.store_dir / f"{obj_id}.fits.gz", row=row,
            )
            try:
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
                if status == 404:      # target exists, product not available
                    result = ItemResult(obj_id=obj_id, success=True, data=None)
                    results.append(self.enrich_result(result, row=row, url=url))
                else:
                    result = ItemResult(
                        obj_id=obj_id, success=False, error=repr(exc),
                    )
                    results.append(self.enrich_result(result, row=row, url=url))
        return results

    # ------------------------------------------------------------------ #
    # image cutouts (mode="image")
    # ------------------------------------------------------------------ #
    def image_url(self, ra: float, dec: float) -> str:
        """SkyServer ``ImgCutout`` JPEG URL."""
        return (f"{IMGCUTOUT_URL}?ra={ra}&dec={dec}"
                f"&width={self.image_width}&height={self.image_height}"
                f"&scale={self.image_scale}&opt={self.image_opt}")

    def _fetch_image_batch(self, rows: pd.DataFrame,
                           ctx: FetchContext) -> List[ItemResult]:
        results: List[ItemResult] = []
        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            ra, dec = ctx.row_coord(row)
            dest = ctx.dest(obj_id, ctx.store_dir / f"{obj_id}.jpg", row=row)
            try:
                url = self.image_url(ra, dec)
                self._download_file(ctx, url, dest)
                result = ItemResult(obj_id=obj_id, success=True, data=dest)
                results.append(
                    self.enrich_result(result, row=row, url=url, dest=dest)
                )
            except Exception as exc:
                result = ItemResult(obj_id=obj_id, success=False, error=repr(exc))
                results.append(self.enrich_result(result, row=row))
        return results
