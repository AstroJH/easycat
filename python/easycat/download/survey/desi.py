"""DESI data downloader.

Two kinds of data are supported (selected via ``mode``):

* ``"photometry"`` - Legacy Surveys Tractor per-brick catalogs
  (``tractor-<brick>.fits``).  This is the photometry used for DESI target
  selection (g/r/z + WISE W1/W2). Each brick file is downloaded once and
  cached, then all sources inside the brick are matched locally.

* ``"spectra"`` — DESI DR1 coadded spectra.  DESI stores spectra per
  HEALPix pixel (nside=64, nested scheme); each ``coadd-*.fits`` file
  contains every target in a pixel and can be ~1 GB.  Instead of
  downloading whole files, :class:`DESIArchive` opens the file through
  :class:`~easycat.download.client.RangeFile` (HTTP Range requests) and
  transfers only the handful of rows belonging to the requested target.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal
from numpy.typing import NDArray

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from ...healpix import ra_dec_to_healpix
from ..base import FetchContext, ItemResult, SurveyArchive
from ..client import RangeFile

logger = logging.getLogger("easycat.download")

# --------------------------------------------------------------------------- #
# Legacy Surveys (photometry)
# --------------------------------------------------------------------------- #
LS_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
BRICKS_RELPATH = "survey-bricks.fits.gz"

# DESI targeting used Legacy Surveys (LS) DR9.
# NOTE: DR10 updated the southern (DECaLS) footprint.
LS_RELEASES = ("dr9", "dr10")

# --------------------------------------------------------------------------- #
# DESI spectra (DR1, "iron" production)
# --------------------------------------------------------------------------- #
DESI_BASE = "https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/healpix"

# Candidate (survey, program) pairs, in the order they are tried.
DESI_SURVEY_PROGRAMS = [
    ("main", "dark"),
    ("main", "bright"),
    ("sv3", "dark"),
    ("sv3", "bright"),
    ("sv1", "dark"),
    ("sv1", "bright"),
    ("sv2", "dark"),
    ("sv2", "bright"),
    ("special", "dark"),
    ("special", "bright"),
    ("cmx", "dark"),
    ("cmx", "bright"),
]

SPECTRUM_COLUMNS = [
    "FLUX_G", "FLUX_R", "FLUX_Z", "FLUX_W1", "FLUX_W2",
    "FLUX_IVAR_G", "FLUX_IVAR_R", "FLUX_IVAR_Z", "FLUX_IVAR_W1", "FLUX_IVAR_W2",
    "FIBERFLUX_G", "FIBERFLUX_R", "FIBERFLUX_Z",
    "FIBERTOTFLUX_G", "FIBERTOTFLUX_R", "FIBERTOTFLUX_Z",
    "NOBS_G", "NOBS_R", "NOBS_Z", "NOBS_W1", "NOBS_W2",
    "MW_TRANSMISSION_G", "MW_TRANSMISSION_R", "MW_TRANSMISSION_Z",
    "MW_TRANSMISSION_W1", "MW_TRANSMISSION_W2",
    "TYPE", "MASKBITS", "OBJID",
]

# Tractor catalog columns (lowercase in the files) -> saved output names.
PHOTOMETRY_COLUMNS = {
    "objid": "OBJID",
    "type": "TYPE",
    "ra": "RA",
    "dec": "DEC",
    "flux_g": "FLUX_G", "flux_r": "FLUX_R", "flux_z": "FLUX_Z",
    "flux_w1": "FLUX_W1", "flux_w2": "FLUX_W2",
    "flux_ivar_g": "FLUX_IVAR_G", "flux_ivar_r": "FLUX_IVAR_R",
    "flux_ivar_z": "FLUX_IVAR_Z", "flux_ivar_w1": "FLUX_IVAR_W1",
    "flux_ivar_w2": "FLUX_IVAR_W2",
    "fiberflux_g": "FIBERFLUX_G", "fiberflux_r": "FIBERFLUX_R",
    "fiberflux_z": "FIBERFLUX_Z",
    "fibertotflux_g": "FIBERTOTFLUX_G", "fibertotflux_r": "FIBERTOTFLUX_R",
    "fibertotflux_z": "FIBERTOTFLUX_Z",
    "nobs_g": "NOBS_G", "nobs_r": "NOBS_R", "nobs_z": "NOBS_Z",
    "nobs_w1": "NOBS_W1", "nobs_w2": "NOBS_W2",
    "mw_transmission_g": "MW_TRANSMISSION_G", "mw_transmission_r": "MW_TRANSMISSION_R",
    "mw_transmission_z": "MW_TRANSMISSION_Z", "mw_transmission_w1": "MW_TRANSMISSION_W1",
    "mw_transmission_w2": "MW_TRANSMISSION_W2",
    "psfsize_g": "PSFSIZE_G", "psfsize_r": "PSFSIZE_R", "psfsize_z": "PSFSIZE_Z",
    "psfdepth_g": "PSFDEPTH_G", "psfdepth_r": "PSFDEPTH_R", "psfdepth_z": "PSFDEPTH_Z",
    "fracflux_g": "FRACFLUX_G", "fracflux_r": "FRACFLUX_R", "fracflux_z": "FRACFLUX_Z",
    "fracmasked_g": "FRACMASKED_G", "fracmasked_r": "FRACMASKED_R",
    "fracmasked_z": "FRACMASKED_Z",
    "allmask_g": "ALLMASK_G", "allmask_r": "ALLMASK_R", "allmask_z": "ALLMASK_Z",
    "wisemask_w1": "WISEMASK_W1", "wisemask_w2": "WISEMASK_W2",
    "ebv": "EBV", "mjd_min": "MJD_MIN", "mjd_max": "MJD_MAX",
}


class DESIArchive(SurveyArchive):
    """DESI / Legacy Surveys downloader.

    Parameters
    ----------
    mode : str
        ``"photometry"`` (Legacy Surveys Tractor) or ``"spectra"``
        (DESI DR1 coadds via HTTP Range).
    radius_arcsec : float
        Matching radius.
    ls_release : str
        Legacy Surveys release for photometry (``"dr9"`` default, the
        release used by DESI target selection; ``"dr10"`` updates the
        southern footprint).
    cache_dir : str or Path or None
        Where to cache the bricks index / tractor / redrock files.
        Defaults to ``<store_dir>/_desi_cache``.
    """

    name = "desi"
    default_batch_size = 16

    def __init__(
        self,
        *,
        mode: Literal["photometry", "spectra"] = "photometry",
        radius_arcsec: float = 3.0,
        ls_release: str = "dr9",
        cache_dir: Optional[Path] = None,
    ):
        if mode not in ("photometry", "spectra"):
            raise ValueError(f"mode must be 'photometry' or 'spectra', got {mode!r}")
        
        super().__init__(
            mode=mode,
            radius_arcsec=radius_arcsec,
            ls_release=ls_release,
        )

        self.mode = mode
        self.radius_arcsec = float(radius_arcsec)
        self.ls_release = ls_release
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # In-memory caches shared across fetch_batch calls in one process.
        self._bricks: Optional[Table]        = None
        self._brick_ra: Optional[NDArray]    = None
        self._brick_dec: Optional[NDArray]   = None
        self._brick_names: Optional[NDArray] = None

    # ------------------------------------------------------------------ #
    # SurveyArchive
    # ------------------------------------------------------------------ #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        if self.mode == "photometry":
            return self._fetch_photometry_batch(rows, ctx)
        
        return self._fetch_spectra_batch(rows, ctx)

    # ------------------------------------------------------------------ #
    # photometry
    # ------------------------------------------------------------------ #
    def _fetch_photometry_batch(
        self, rows: pd.DataFrame, ctx: FetchContext
    ) -> List[ItemResult]:
        bricks = self._load_bricks(ctx)
        cache = self._cache_dir(ctx)

        # Group source rows by brick.
        by_brick: Dict[str, List[int]] = {}
        for idx, (_, row) in enumerate(rows.iterrows()):
            ra, dec = ctx.row_coord(row)
            brickname = self._find_brick(ra, dec, bricks)
            if brickname is None:
                continue
            by_brick.setdefault(brickname, []).append(idx)

        results: List[Optional[ItemResult]] = [None] * len(rows)
        for brickname, idxs in by_brick.items():
            try:
                dec_center = float(self._brick_dec[np.flatnonzero(
                    self._brick_names == brickname
                )[0]]) if self._brick_dec is not None else 0.0
                tractor = self._load_tractor(ctx, cache, brickname, dec_center)
                if tractor is None or len(tractor) == 0:
                    for i in idxs:
                        obj_id = ctx.row_id(rows.iloc[i])
                        results[i] = ItemResult(obj_id=obj_id, success=True, data=None)
                    continue

                coords = SkyCoord(
                    np.asarray(tractor["ra"], float),
                    np.asarray(tractor["dec"], float),
                    unit="deg",
                )

                for i in idxs:
                    row = rows.iloc[i]
                    obj_id = ctx.row_id(row)
                    ra, dec = ctx.row_coord(row)
                    target = SkyCoord(ra, dec, unit="deg")
                    sep = target.separation(coords).to_value(u.arcsec)
                    j = int(np.argmin(sep))
                    if sep[j] > self.radius_arcsec:
                        results[i] = ItemResult(obj_id=obj_id, success=True, data=None)
                        continue
                    try:
                        out = self._write_photometry(ctx, obj_id, tractor[j], brickname)
                        results[i] = ItemResult(obj_id=obj_id, success=True, data=out)
                    except Exception as exc:
                        results[i] = ItemResult(obj_id=obj_id, success=False, error=repr(exc))
            except Exception as exc:
                for i in idxs:
                    obj_id = ctx.row_id(rows.iloc[i])
                    results[i] = ItemResult(obj_id=obj_id, success=False, error=repr(exc))

        # Fill in rows whose brick could not be determined.
        for i, r in enumerate(results):
            if r is None:
                obj_id = ctx.row_id(rows.iloc[i])
                results[i] = ItemResult(
                    obj_id=obj_id, success=False, error="brick not found"
                )
        return results

    def _write_photometry(self, ctx, obj_id, row, brickname) -> Path:
        out = ctx.store_dir / f"{obj_id}.fits"
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = [c for c in PHOTOMETRY_COLUMNS if c in row.colnames]
        rename = {c: PHOTOMETRY_COLUMNS[c] for c in cols}
        data = {rename.get(c, c): [row[c]] for c in cols}
        t = Table(data)
        t["BRICKNAME"] = brickname
        t.meta = {}
        t.write(out, overwrite=True)
        return out

    def _load_bricks(self, ctx: FetchContext) -> Table:
        if self._bricks is not None:
            return self._bricks
        cache = self._cache_dir(ctx)
        local = cache / BRICKS_RELPATH
        url = f"{LS_BASE}/{self.ls_release}/{BRICKS_RELPATH}"
        if not local.exists():
            logger.info("Downloading LS bricks index %s (one-time)...", url)
            ctx.client.download_file(url, local)
        import gzip

        with gzip.open(local, "rb") as f:
            with fits.open(io.BytesIO(f.read())) as h:
                self._bricks = Table(h[1].data)
        self._brick_ra = np.asarray(self._bricks["RA"], float)
        self._brick_dec = np.asarray(self._bricks["DEC"], float)
        self._brick_names = np.asarray(self._bricks["BRICKNAME"])
        return self._bricks

    def _find_brick(self, ra: float, dec: float, bricks: Table) -> Optional[str]:
        # Use the brick bounding boxes; fall back to nearest centre.
        ra1 = np.asarray(bricks["RA1"], float)
        ra2 = np.asarray(bricks["RA2"], float)
        dec1 = np.asarray(bricks["DEC1"], float)
        dec2 = np.asarray(bricks["DEC2"], float)

        # Handle RA wrap (e.g. a box crossing 0/360).
        wrap = ra2 < ra1
        inside = (dec1 <= dec) & (dec <= dec2) & (
            ((ra1 <= ra) & (ra <= ra2)) | (wrap & ((ra >= ra1) | (ra <= ra2)))
        )
        if np.any(inside):
            idx = int(np.flatnonzero(inside)[0])
            return str(self._brick_names[idx])
        # Fallback: nearest centre (must be close).
        sep = np.sqrt(
            ((self._brick_ra - ra) * np.cos(np.deg2rad(dec))) ** 2
            + (self._brick_dec - dec) ** 2
        )
        i = int(np.argmin(sep))
        if sep[i] < 0.2:
            return str(self._brick_names[i])
        return None

    def _load_tractor(
        self, ctx, cache: Path, brickname: str, dec_center: float
    ) -> Optional[Table]:
        region = "north" if dec_center >= 32.375 else "south"
        pre3 = brickname[:3]

        local = cache / "tractor" / f"tractor-{brickname}.fits"
        if not local.exists():
            url = f"{LS_BASE}/{self.ls_release}/{region}/tractor/{pre3}/tractor-{brickname}.fits"
            try:
                ctx.client.download_file(url, local)
            except Exception:
                # Fall back to the other region (footprint edge cases).
                other = "south" if region == "north" else "north"
                url = f"{LS_BASE}/{self.ls_release}/{other}/tractor/{pre3}/tractor-{brickname}.fits"
                ctx.client.download_file(url, local)
        with fits.open(local, memmap=False) as h:
            return Table(h[1].data)

    # ------------------------------------------------------------------ #
    # spectra
    # ------------------------------------------------------------------ #
    def _fetch_spectra_batch(
        self, rows: pd.DataFrame, ctx: FetchContext
    ) -> List[ItemResult]:
        # Group by HEALPix pixel so one coadd file serves many sources.
        pixels: Dict[int, List[int]] = {}
        for i, (_, row) in enumerate(rows.iterrows()):
            ra, dec = ctx.row_coord(row)
            hp = int(ra_dec_to_healpix(ra, dec, nside=64)[0])
            pixels.setdefault(hp, []).append(i)

        results: List[Optional[ItemResult]] = [None] * len(rows)
        for hp, idxs in pixels.items():
            try:
                self._process_pixel(rows, idxs, hp, ctx, results)
            except Exception as exc:
                for i in idxs:
                    obj_id = ctx.row_id(rows.iloc[i])
                    results[i] = ItemResult(obj_id=obj_id, success=False, error=repr(exc))
        for i, r in enumerate(results):
            if r is None:
                obj_id = ctx.row_id(rows.iloc[i])
                results[i] = ItemResult(obj_id=obj_id, success=False, error="no data")
        return results

    def _process_pixel(self, rows, idxs, hp, ctx, results) -> None:
        cache = self._cache_dir(ctx)
        for survey, program in DESI_SURVEY_PROGRAMS:
            url = (
                f"{DESI_BASE}/{survey}/{program}/{hp // 100}/{hp}/"
                f"coadd-{survey}-{program}-{hp}.fits"
            )
            # Quick existence check before opening a RangeFile.
            try:
                resp = ctx.client.head(url)
            except Exception:
                continue
            if resp.status_code != 200:
                continue

            try:
                with RangeFile(url, client=ctx.client) as rf:
                    with fits.open(rf, memmap=False) as hdul:
                        if "FIBERMAP" not in hdul:
                            continue
                        fm = hdul["FIBERMAP"].data
                        tra = np.asarray(fm["TARGET_RA"], float)
                        tdec = np.asarray(fm["TARGET_DEC"], float)
                        tids = np.asarray(fm["TARGETID"])
                        coords = SkyCoord(tra, tdec, unit="deg")

                        for i in idxs:
                            row = rows.iloc[i]
                            obj_id = ctx.row_id(row)
                            ra, dec = ctx.row_coord(row)
                            target = SkyCoord(ra, dec, unit="deg")
                            sep = target.separation(coords).to_value(u.arcsec)
                            j = int(np.argmin(sep))
                            if sep[j] > self.radius_arcsec:
                                results[i] = ItemResult(
                                    obj_id=obj_id, success=True, data=None
                                )
                                continue
                            try:
                                out = self._extract_spectrum(
                                    ctx, cache, hdul, survey, program, hp,
                                    obj_id, int(tids[j]), j, ra, dec,
                                )
                                results[i] = ItemResult(
                                    obj_id=obj_id, success=True, data=out
                                )
                            except Exception as exc:
                                results[i] = ItemResult(
                                    obj_id=obj_id, success=False, error=repr(exc)
                                )
                return  # first (survey, program) with a match wins
            except Exception as exc:
                logger.warning("DESI pixel %d %s/%s failed: %s", hp, survey, program, exc)
                continue

    def _extract_spectrum(
        self, ctx, cache, hdul, survey, program, hp,
        obj_id, targetid, row, ra, dec,
    ) -> Path:
        arms = {}
        for arm in "BRZ":
            wave = np.asarray(hdul[f"{arm}_WAVELENGTH"].data).copy()
            flux = np.asarray(hdul[f"{arm}_FLUX"].section[row, :]).copy()
            ivar = np.asarray(hdul[f"{arm}_IVAR"].section[row, :]).copy()
            mask = np.asarray(hdul[f"{arm}_MASK"].section[row, :]).copy()
            arms[arm] = (wave, flux, ivar, mask)

        z, spectype = self._redrock_z(ctx, cache, survey, program, hp, targetid)

        out = ctx.store_dir / f"{obj_id}.fits"
        out.parent.mkdir(parents=True, exist_ok=True)

        primary = fits.PrimaryHDU()
        primary.header["TARGETID"] = targetid
        primary.header["SURVEY"] = survey
        primary.header["PROGRAM"] = program
        primary.header["HEALPIX"] = hp
        primary.header["Z"] = z if z is not None else np.nan
        primary.header["SPECTYPE"] = spectype or ""
        primary.header["RA"] = ra
        primary.header["DEC"] = dec

        hdus = [primary]
        for arm, (wave, flux, ivar, mask) in arms.items():
            hdus.append(fits.ImageHDU(wave, name=f"{arm}_WAVELENGTH"))
            hdus.append(fits.ImageHDU(flux, name=f"{arm}_FLUX"))
            hdus.append(fits.ImageHDU(ivar, name=f"{arm}_IVAR"))
            hdus.append(fits.ImageHDU(mask, name=f"{arm}_MASK"))
        fits.HDUList(hdus).writeto(out, overwrite=True)
        return out

    def _redrock_z(
        self, ctx, cache: Path, survey: str, program: str, hp: int, targetid: int
    ) -> Tuple[Optional[float], Optional[str]]:
        """Look up Z / SPECTYPE from the per-pixel redrock catalog."""
        
        local = cache / "redrock" / f"redrock-{survey}-{program}-{hp}.fits"
        if not local.exists():
            url = (
                f"{DESI_BASE}/{survey}/{program}/{hp // 100}/{hp}/"
                f"redrock-{survey}-{program}-{hp}.fits"
            )
            try:
                ctx.client.download_file(url, local)
            except Exception:
                return None, None
        try:
            with fits.open(local, memmap=False) as h:
                d = h[1].data
            tids = np.asarray(d["TARGETID"])
            hit = np.flatnonzero(tids == targetid)
            if len(hit) == 0:
                return None, None
            row = d[hit[0]]
            z = float(row["Z"]) if "Z" in d.columns.names else None
            st = str(row["SPECTYPE"]) if "SPECTYPE" in d.columns.names else None
            return z, st
        except Exception:
            return None, None

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _cache_dir(self, ctx: FetchContext) -> Path:
        if self.cache_dir is not None:
            path = Path(self.cache_dir)
        else:
            path = ctx.store_dir / "_desi_cache"
    
        path.mkdir(parents=True, exist_ok=True)
        return path
