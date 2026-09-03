"""WISE/NEOWISE light-curve download via the IRSA Gator bulk upload API.

The Gator ``spatial=Upload`` mode performs cone searches for many positions
in a *single* HTTP request: you upload an IPAC table of coordinates and the
server returns every catalog match, tagged with the input row counter
(``cntr_01``) and the match separation (``dist_x``).

``WISEArchive`` replaces the old per-source
NEOWISE + AllWISE double query with ~2 requests per *batch* of sources.
"""
from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import MaskedColumn, Table, vstack

from ..base import FetchContext, ItemResult, SurveyArchive

logger = logging.getLogger("easycat.download")

GATOR_URL = "https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query"

NEOWISE_CATALOG = "neowiser_p1bs_psd"
ALLWISE_CATALOG = "allwise_p3as_mep"

# Columns requested from each catalog (must match combine_wisedata()).
NEOWISE_SELCOLS = (
    "ra,dec,mjd,w1mpro,w1sigmpro,w1rchi2,w2mpro,w2sigmpro,w2rchi2,"
    "na,nb,qi_fact,cc_flags,qual_frame,saa_sep,moon_masked"
)
ALLWISE_SELCOLS = (
    "ra,dec,mjd,w1mpro_ep,w1sigmpro_ep,w1rchi2_ep,w2mpro_ep,"
    "w2sigmpro_ep,w2rchi2_ep,na,nb,qi_fact,cc_flags,saa_sep,moon_masked"
)

# Fields of the final per-source light-curve table.
LC_FIELDS = [
    "raj2000", "dej2000", "mjd",
    "w1mag", "w1sigmag", "w1rchi2",
    "w2mag", "w2sigmag", "w2rchi2",
    "na", "nb", "qi_fact", "cc_flags",
    "qual_frame", "saa_sep", "moon_masked",
]


NEOWISE_MAP = {
    "ra": "raj2000", "dec": "dej2000", "mjd": "mjd",
    "w1mpro": "w1mag", "w1sigmpro": "w1sigmag", "w1rchi2": "w1rchi2",
    "w2mpro": "w2mag", "w2sigmpro": "w2sigmag", "w2rchi2": "w2rchi2",
    "na": "na", "nb": "nb", "qi_fact": "qi_fact", "cc_flags": "cc_flags",
    "qual_frame": "qual_frame", "saa_sep": "saa_sep", "moon_masked": "moon_masked",
}

ALLWISE_MAP = {
    "ra": "raj2000", "dec": "dej2000", "mjd": "mjd",
    "w1mpro_ep": "w1mag", "w1sigmpro_ep": "w1sigmag", "w1rchi2_ep": "w1rchi2",
    "w2mpro_ep": "w2mag", "w2sigmpro_ep": "w2sigmag", "w2rchi2_ep": "w2rchi2",
    "na": "na", "nb": "nb", "qi_fact": "qi_fact", "cc_flags": "cc_flags",
    "saa_sep": "saa_sep", "moon_masked": "moon_masked",
}


def _project_columns(table: Optional[Table], mapping: dict) -> Optional[Table]:
    """Keep only the columns in ``mapping`` (renamed to the map values)."""
    if table is None or len(table) == 0:
        return None
    old = [c for c in mapping if c in table.colnames]
    if not old:
        return None
    new = [mapping[c] for c in old]
    sub = table[old].copy()
    sub.rename_columns(old, new)
    return sub


def combine_wisedata(t_neowise: Optional[Table], t_allwise: Optional[Table]) -> Table:
    """Merge NEOWISE and AllWISE photometry into one light-curve table.

    Parameters
    ----------
    t_neowise : astropy.table.Table or None
        NEOWISE epochs (columns ``ra, dec, mjd, w1mpro, ...``).
    t_allwise : astropy.table.Table or None
        AllWISE epochs (columns ``ra, dec, mjd, w1mpro_ep, ...``) or None.

    Returns
    -------
    astropy.table.Table
        Combined table with the standard easycat light-curve columns.
    """
    t_neo = _project_columns(t_neowise, NEOWISE_MAP)
    t_all = _project_columns(t_allwise, ALLWISE_MAP)

    if t_all is not None and "qual_frame" not in t_all.colnames:
        t_all["qual_frame"] = -1

    tables = [t for t in (t_neo, t_all) if t is not None and len(t) > 0]
    if not tables:
        return Table(names=LC_FIELDS)

    t_combined: Table = vstack(tables, metadata_conflicts="silent")
    t_combined = t_combined[LC_FIELDS]

    # Normal values for these fields should be >= 0; use -1 for masked /
    # abnormal values.
    for col in ("w1mag", "w2mag", "w1sigmag", "w2sigmag",
                "w1rchi2", "w2rchi2"):
        c = t_combined[col]
        if isinstance(c, MaskedColumn):
            t_combined[col] = c.filled(fill_value=-1)
        else:
            arr = np.asarray(c, dtype=float)
            if np.isnan(arr).any():
                t_combined[col] = np.where(np.isnan(arr), -1.0, arr)

    return t_combined


def _build_ipac_table(rows: pd.DataFrame, ra_col: str, dec_col: str) -> str:
    """Build the IPAC table body for a Gator Upload cone search."""
    lines = [
        "\\ EQUINOX = J2000.0",
        "|   ra     |   dec    |",
        "|   double |   double |",
    ]
    for _, row in rows.iterrows():
        ra = float(row[ra_col])
        dec = float(row[dec_col])
        lines.append(f" {ra:.7f}  {dec:.7f}")
    return "\n".join(lines)


def _rows_retrieved(text: str) -> int:
    m = re.search(r"RowsRetrieved\s*=\s*(\d+)", text)
    return int(m.group(1)) if m else -1


class WISEArchive(SurveyArchive):
    """Batch WISE/NEOWISE light-curve downloader.

    Parameters
    ----------
    radius_arcsec : float
        Cone-search radius in arcseconds (per source).
    store_format : str
        ``"fits"`` (default) writes ``<obj_id>.fits`` per source.
    """

    name = "wise"
    default_batch_size = 200

    def __init__(self, *, radius_arcsec: float = 3.0, store_format: str = "fits"):
        super().__init__(
            radius_arcsec=radius_arcsec,
            store_format=store_format,
        )
        self.radius_arcsec = float(radius_arcsec)
        self.store_format = store_format

    # ---------------------------------------- #
    # SurveyArchive
    # ---------------------------------------- #
    def fetch_batch(self, rows: pd.DataFrame, ctx: FetchContext) -> List[ItemResult]:
        if len(rows) == 0:
            return []

        table_text = _build_ipac_table(rows, ctx.ra_column, ctx.dec_column)

        t_neowise = self._query_catalog(
            ctx, NEOWISE_CATALOG, NEOWISE_SELCOLS, table_text,
            required=True,
        )
        t_allwise = self._query_catalog(
            ctx, ALLWISE_CATALOG, ALLWISE_SELCOLS, table_text,
            required=False,
        )

        # Map Gator's 1-based input row counter -> original catalog index.
        neowise_by_row: Dict[int, Optional[Table]] = self._slice_by_row(t_neowise)
        allwise_by_row: Dict[int, Optional[Table]] = self._slice_by_row(t_allwise)

        results: List[ItemResult] = []
        for idx, (_, row) in enumerate(rows.iterrows(), start=1):
            obj_id = ctx.row_id(row)
            try:
                t_neo = neowise_by_row.get(idx)
                t_all = allwise_by_row.get(idx)
                if t_neo is None and t_all is None:
                    # Query succeeded but the source has no WISE detections.
                    results.append(ItemResult(obj_id=obj_id, success=True, data=None))
                    continue

                combined = combine_wisedata(t_neo, t_all)
                if len(combined) == 0:
                    results.append(ItemResult(obj_id=obj_id, success=True, data=None))
                    continue

                out_path = ctx.store_dir / f"{obj_id}.fits"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                combined.meta = {}  # drop parser bookkeeping (e.g. `keywords`)
                combined.write(out_path, overwrite=True)
                results.append(
                    ItemResult(obj_id=obj_id, success=True, data=combined)
                )
            except Exception as exc:
                logger.warning("WISE %s: %s", obj_id, exc)
                results.append(
                    ItemResult(obj_id=obj_id, success=False, error=repr(exc))
                )
        return results

    def _query_catalog(
        self,
        ctx: FetchContext,
        catalog: str,
        selcols: str,
        table_text: str,
        *,
        required: bool,
    ) -> Optional[Table]:
        try:
            resp = ctx.client.post(
                GATOR_URL,
                files={"filename": ("upload.tbl", table_text, "text/plain")},
                data={
                    "spatial": "Upload",
                    "uradius": str(self.radius_arcsec),
                    "uradunits": "arcsec",
                    "catalog": catalog,
                    "outfmt": "1",
                    "selcols": selcols,
                },
            )
            resp.raise_for_status()
            text = resp.text

            if "QUERY_STATUS" in text and "ERROR" in text:
                raise RuntimeError(f"Gator error for {catalog}: {text[:300]}")

            n = _rows_retrieved(text)
            if n == 0:
                return None
            if n < 0:
                raise RuntimeError(
                    f"Cannot parse Gator response for {catalog} "
                    f"(status={resp.status_code}, len={len(text)})"
                )

            return ascii.read(text, format="ipac")
        except Exception as exc:
            if required:
                raise
            logger.warning("AllWISE query failed (continuing with NEOWISE only): %s", exc)
            return None

    @staticmethod
    def _slice_by_row(
        tbl: Optional[Table],
    ) -> Dict[int, Optional[Table]]:
        """Split a batch result table into per-input-row tables."""
        if tbl is None:
            return {}
        out: Dict[int, Optional[Table]] = {}
        bookkeeping = ("cntr_01", "dist_x", "pang_x", "ra_01", "dec_01")
        for cntr in np.unique(tbl["cntr_01"]):
            idx = int(cntr)
            sub = tbl[tbl["cntr_01"] == cntr]
            for col in bookkeeping:
                if col in sub.colnames:
                    sub.remove_column(col)
            out[idx] = sub
        return out
