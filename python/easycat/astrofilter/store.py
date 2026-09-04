"""Local persistence and in-memory cache for filters.

Filters are stored as **ECSV** files (astropy's human-readable table
format) -- one file per filter -- so the cache is transparent.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import astropy.units as u
from astropy.io import ascii
from astropy.table import Table

from .core import AstroFilter, DetectorType

logger = logging.getLogger("easycat.astrofilter")

# All SVO metadata keys are stored under this meta key of the ECSV table.
META_KEY = "svo_metadata"


def cache_filename(filter_id: str) -> str:
    """Deterministic, filesystem-safe file name for a filter ID."""
    return filter_id.replace("/", "__").replace(" ", "_") + ".ecsv"


def _to_table(filt: AstroFilter) -> Table:
    t = Table()
    t["wavelength"] = filt.wavelength  # Quantity -> unit preserved
    t["transmission"] = filt.transmission
    meta = {
        "filter_id": filt.filter_id,
        "facility": filt.facility,
        "instrument": filt.instrument,
        "band": filt.band,
        "detector_type": filt.detector_type.name,
        META_KEY: filt.metadata,
    }
    if filt._zp_vega_jy is not None:
        meta["zp_vega_jy"] = float(filt._zp_vega_jy.to_value(u.Jy))
    t.meta = meta
    return t


def _from_table(t: Table) -> AstroFilter:
    meta = dict(t.meta)
    svo = meta.pop(META_KEY, {}) or {}
    zp = meta.pop("zp_vega_jy", None)
    filt = AstroFilter(
        wavelength=t["wavelength"].quantity,
        transmission=np.asarray(t["transmission"], dtype=float),
        detector_type=DetectorType[meta.get("detector_type", "PHOTON")],
        filter_id=meta.get("filter_id", ""),
        facility=meta.get("facility", ""),
        instrument=meta.get("instrument", ""),
        band=meta.get("band", ""),
        metadata=dict(svo),
    )
    if zp is not None:
        filt.set_zp_vega(zp)
    return filt


class FilterStore:
    """ECSV-backed filter cache with an in-memory layer."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._memory: Dict[str, AstroFilter] = {}

    # ------------------------------------------------------------------ #
    def path_for(self, filter_id: str) -> Path:
        return self.root / cache_filename(filter_id)

    def contains(self, filter_id: str) -> bool:
        return (
            filter_id in self._memory
            or self.path_for(filter_id).exists()
        )

    def load(self, filter_id: str) -> Optional[AstroFilter]:
        if filter_id in self._memory:
            return self._memory[filter_id]
        path = self.path_for(filter_id)
        if not path.exists():
            return None
        try:
            t = ascii.read(path, format="ecsv")
            filt = _from_table(t)
        except Exception as exc:
            logger.warning("corrupted cache %s: %s", path, exc)
            return None
        self._memory[filter_id] = filt
        return filt

    def save(self, filt: AstroFilter) -> None:
        path = self.path_for(filt.filter_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        _to_table(filt).write(tmp, format="ascii.ecsv", overwrite=True)
        tmp.replace(path)
        self._memory[filt.filter_id] = filt

    def list_ids(self) -> List[str]:
        ids = list(self._memory)
        for path in sorted(self.root.glob("*.ecsv")):
            fid = path.stem.replace("__", "/")
            if fid not in ids:
                ids.append(fid)
        return ids

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search cached filters by ID / facility / instrument / band."""
        q = query.lower()
        out = []
        for fid in self.list_ids():
            filt = self.load(fid)
            if filt is None:
                continue
            haystack = " ".join(
                [filt.filter_id, filt.facility, filt.instrument, filt.band,
                 str(filt.metadata.get("PhotSystem", "")),
                 str(filt.metadata.get("Description", ""))]
            ).lower()
            if q in haystack:
                out.append(
                    {
                        "filterID": filt.filter_id,
                        "Facility": filt.facility,
                        "Instrument": filt.instrument,
                        "Band": filt.band,
                        "DetectorType": str(filt.detector_type),
                        "WavelengthPivot": round(filt.wl_pivot.to_value(u.AA), 2),
                    }
                )
        return out
