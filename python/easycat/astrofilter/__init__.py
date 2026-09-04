"""easycat filter subsystem.

Modules
-------
core      -- DetectorType + AstroFilter
spectrum  -- self-contained Spectrum / SED interface
svo2      -- SVO2 FPS data access (profiles + reference spectra)
store     -- ECSV persistence + in-memory cache
db        -- FilterDB facade (cache -> local -> SVO2)
cli       -- ``easycat filter`` command implementation

Quick start::

    from easycat.astrofilter import FilterDB, get_vega_spectrum

    db = FilterDB()
    f = db.get("SLOAN/SDSS.r")
    f.wl_pivot # detector-aware pivot wavelength
    f.synthetic_flux(get_vega_spectrum()) # in Jy
    f @ get_vega_spectrum() # same, via @ sugar
"""
from .core import AstroFilter, DetectorType
from .db import AstroFilterDB, FilterDB, default_cache_dir
from .spectrum import Spectrum
from .svo2 import (
    fetch_filter,
    get_sun_spectrum,
    get_vega_spectrum,
    search,
)

__all__ = [
    "AstroFilter",
    "DetectorType",
    "Spectrum",
    "FilterDB",
    "AstroFilterDB",
    "default_cache_dir",
    "fetch_filter",
    "get_vega_spectrum",
    "get_sun_spectrum",
    "search"
]
