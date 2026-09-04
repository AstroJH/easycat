"""SVO2 Filter Profile Service (FPS) data access.

Only *data acquisition* lives here: downloading filter profiles,
searching the FPS catalogue and fetching reference spectra (Vega / Sun).
"""
from __future__ import annotations

import os
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import astropy.units as u
import numpy as np
from astropy.io import ascii, votable
from astropy.table import Table
from astropy.units import Quantity

from ..download.client import HttpClient
from .core import AstroFilter, DetectorType
from .spectrum import Spectrum

logger = logging.getLogger("easycat.astrofilter")

URL_FPS = "https://svo2.cab.inta-csic.es/theory/fps"
URL_FILTER = f"{URL_FPS}/fps.php"
URL_VEGA = f"{URL_FPS}/morefiles/vega.dat"
URL_SUN = f"{URL_FPS}/morefiles/sun.dat"


def _client() -> HttpClient:
    return HttpClient(timeout=60, retries=3, backoff=1.0)


def normalize_filter_id(filter_id: str) -> str:
    """Normalise an easycat-style ID to the SVO ``facility/instrument.band``.

    Accepts either SVO style (e.g., ``HST/WFC3_IR.F160W``) or the dot
    style (e.g., ``HST.WFC3_IR.f160W``).
    """
    s = filter_id.strip()
    if "/" in s:
        return s
    
    # dot style: facility.instrument.band -> facility/instrument.band
    parts = s.split(".")
    if len(parts) == 3:
        return f"{parts[0]}/{parts[1]}.{parts[2]}"
    
    return s


def _parse_metadata(params) -> Dict[str, Any]:
    """Convert VOTable PARAMs to a plain dict (strings preserved)."""
    meta: Dict[str, Any] = {}
    for p in params:
        if p.ID is not None:
            meta[p.ID] = p.value
    return meta


def fetch_filter(filter_id: str, client: Optional[HttpClient] = None) -> AstroFilter:
    """Download a filter profile from SVO2 and build an :class:`AstroFilter`.

    Parameters
    ----------
    filter_id : str
        SVO ID, e.g. ``"HST/WFC3_IR.F160W"`` (or dot style).
    client : HttpClient or None

    Returns
    -------
    AstroFilter
        The filter with original transmission and full SVO metadata.

    Raises
    ------
    ValueError
        If the filter is not found or the response cannot be parsed.
    """
    client = client or _client()
    fid = normalize_filter_id(filter_id)
    resp = client.get(URL_FILTER, params={"ID": fid})
    resp.raise_for_status()

    try:
        v = votable.parse(io.BytesIO(resp.content))
    except Exception as exc:
        raise ValueError(f"cannot parse SVO2 response for {fid}") from exc

    for res in v.resources:
        if res.type == "results" and res.tables:
            tb = res.tables[0]
            meta = _parse_metadata(tb.params)
            arr = tb.array
            if len(arr) == 0:
                raise ValueError(f"empty transmission curve for {fid}")
            wl_unit = u.Unit(meta.get("WavelengthUnit", "Angstrom"))
            wl = Quantity(arr["Wavelength"].astype(float), wl_unit)
            tr = arr["Transmission"].astype(float)
            detector = DetectorType.from_svo(meta.get("DetectorType", "1"))

            slash = fid.find("/")
            dot = fid.find(".", slash + 1)
            facility = meta.get("Facility", fid[:slash] if slash > 0 else "")
            instrument = meta.get("Instrument", "")
            band = meta.get("Band", fid[dot + 1:] if dot > 0 else "")

            return AstroFilter(
                wavelength=wl,
                transmission=tr,
                detector_type=detector,
                filter_id=meta.get("filterID", fid),
                facility=facility,
                instrument=instrument,
                band=band,
                metadata=meta,
            )

    raise ValueError(f"filter not found: {fid}")


def search(query: str, client: Optional[HttpClient] = None) -> List[Dict[str, Any]]:
    """Search the SVO2 FPS catalogue.

    Notes
    -----
    SVO2's programmatic search endpoint frequently returns an empty table
    (the interactive form uses a different flow), so a miss here does not
    mean the filter does not exist.  Use :func:`fetch_filter` with the
    exact ID in that case.
    """
    client = client or _client()
    resp = client.get(
        URL_FILTER, params={"mode": "search", "search": query},
    )
    if resp.status_code != 200:
        return []
    try:
        v = votable.parse(io.BytesIO(resp.content))
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for res in v.resources:
        for tb in res.tables:
            names = [f.ID for f in tb.fields]
            if "filterID" not in names:
                continue
            for row in tb.array:
                out.append(
                    {n: row[n] for n in names if n in row.dtype.names}
                )
    return out


# --------------------------------------------------------------------------- #
# reference spectra
# --------------------------------------------------------------------------- #
def fetch_reference_spectrum(
    url: str, client: Optional[HttpClient] = None, name: str = ""
) -> Spectrum:
    """Fetch a two-column (wavelength-Angstrom, F_lambda) spectrum file.

    SVO reference files (``vega.dat`` / ``sun.dat``) contain
    ``lambda [AA]  F_lambda [erg s^-1 cm^-2 AA^-1]``.
    """
    client = client or _client()
    resp = client.get(url)
    resp.raise_for_status()
    lines = [ln for ln in resp.text.splitlines() if ln.strip() and not ln.lstrip().startswith(("#", "\\"))]
    wl = []
    fl = []
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 2:
            try:
                wl.append(float(parts[0]))
                fl.append(float(parts[1]))
            except ValueError:
                continue
    if len(wl) < 2:
        raise ValueError(f"could not parse reference spectrum: {url}")
    return Spectrum(
        np.asarray(wl) * u.AA,
        np.asarray(fl) * (u.erg / u.s / u.cm**2 / u.AA),
        name=name or url.rsplit("/", 1)[-1],
    )


_vega: Optional[Spectrum] = None
_sun: Optional[Spectrum] = None


def reference_cache_dir(cache_dir: Optional[Path] = None) -> Path:
    """Directory for cached reference spectra.

    Defaults to ``$EASYCAT_DATA/spectra`` or ``~/.easycat/spectra``.
    """
    if cache_dir is not None:
        return Path(cache_dir)
    base = os.environ.get("EASYCAT_DATA")
    root = Path(base) if base else Path.home() / ".easycat"
    return root / "spectra"


def _spectrum_cache_path(name: str, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{name}.ecsv"


def _save_spectrum_cache(spec: Spectrum, cache_dir: Optional[Path]) -> None:
    if cache_dir is None:
        return
    try:
        d = Path(cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        t = Table()
        t["wavelength"] = spec.wavelength
        t["flux"] = spec.flux
        t.meta = {"name": spec.name}
        t.write(_spectrum_cache_path(spec.name or "spec", d),
                format="ascii.ecsv", overwrite=True)
    except Exception as exc:
        logger.warning("could not cache spectrum %s: %s", spec.name, exc)


def _load_spectrum_cache(name: str, cache_dir: Optional[Path]) -> Optional[Spectrum]:
    if cache_dir is None:
        return None
    path = _spectrum_cache_path(name, cache_dir)
    if not path.exists():
        return None
    try:
        t = ascii.read(path, format="ascii.ecsv")
        return Spectrum(t["wavelength"].quantity, t["flux"].quantity, name=name)
    except Exception:
        return None


def get_vega_spectrum(client: Optional[HttpClient] = None,
                      cache_dir: Optional[Path] = None) -> Spectrum:
    """Vega reference spectrum (in-memory + on-disk cache).

    The first call downloads from SVO2; the result is cached to disk so
    later processes do not touch
    the network.
    """
    global _vega
    
    cache_dir = reference_cache_dir(cache_dir)
    if _vega is None:
        _vega = _load_spectrum_cache("vega", cache_dir)

    if _vega is None:
        _vega = fetch_reference_spectrum(URL_VEGA, client, name="vega")
        _save_spectrum_cache(_vega, cache_dir)

    return _vega


def get_sun_spectrum(client: Optional[HttpClient] = None,
                     cache_dir: Optional[Path] = None) -> Spectrum:
    """Solar reference spectrum (in-memory + on-disk cache)."""
    global _sun

    cache_dir = reference_cache_dir(cache_dir)   # persist by default
    if _sun is None:
        _sun = _load_spectrum_cache("sun", cache_dir)

    if _sun is None:
        _sun = fetch_reference_spectrum(URL_SUN, client, name="sun")
        _save_spectrum_cache(_sun, cache_dir)
        
    return _sun
