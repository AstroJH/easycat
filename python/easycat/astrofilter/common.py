"""A curated list of commonly used SVO2 filter IDs.

Useful for offline ``easycat filter search`` when nothing has been cached
yet and the SVO2 online search is unavailable.  Entries are
``(filterID, description)``.
"""
from __future__ import annotations

COMMON_FILTERS = [
    # Broad-band optical
    ("SLOAN/SDSS.u", "SDSS u'"),
    ("SLOAN/SDSS.g", "SDSS g'"),
    ("SLOAN/SDSS.r", "SDSS r'"),
    ("SLOAN/SDSS.i", "SDSS i'"),
    ("SLOAN/SDSS.z", "SDSS z'"),
    ("Generic/Johnson.U", "Johnson U"),
    ("Generic/Johnson.B", "Johnson B"),
    ("Generic/Johnson.V", "Johnson V"),
    ("Generic/Cousins.R", "Cousins R"),
    ("Generic/Cousins.I", "Cousins I"),

    # Infrared surveys
    ("2MASS/2MASS.J", "2MASS J"),
    ("2MASS/2MASS.H", "2MASS H"),
    ("2MASS/2MASS.Ks", "2MASS Ks"),
    ("WISE/WISE.W1", "WISE W1 (3.4 um)"),
    ("WISE/WISE.W2", "WISE W2 (4.6 um)"),
    ("WISE/WISE.W3", "WISE W3 (12 um)"),
    ("WISE/WISE.W4", "WISE W4 (22 um)"),
    ("UKIRT/WFCAM.J", "UKIRT WFCAM J"),
    ("UKIRT/WFCAM.H", "UKIRT WFCAM H"),
    ("UKIRT/WFCAM.K", "UKIRT WFCAM K"),
    ("PANSTARRS/PS1.g", "Pan-STARRS1 g"),
    ("PANSTARRS/PS1.r", "Pan-STARRS1 r"),
    ("PANSTARRS/PS1.i", "Pan-STARRS1 i"),
    ("PANSTARRS/PS1.z", "Pan-STARRS1 z"),
    ("PANSTARRS/PS1.y", "Pan-STARRS1 y"),

    # Space missions
    ("GALEX/GALEX.FUV", "GALEX FUV"),
    ("GALEX/GALEX.NUV", "GALEX NUV"),
    ("SPITZER/IRAC.I1", "Spitzer IRAC ch1 (3.6 um)"),
    ("SPITZER/IRAC.I2", "Spitzer IRAC ch2 (4.5 um)"),
    ("SPITZER/IRAC.I3", "Spitzer IRAC ch3 (5.8 um)"),
    ("SPITZER/IRAC.I4", "Spitzer IRAC ch4 (8.0 um)"),
    ("HST/ACS_WFC.F435W", "HST ACS/WFC F435W"),
    ("HST/ACS_WFC.F606W", "HST ACS/WFC F606W"),
    ("HST/ACS_WFC.F814W", "HST ACS/WFC F814W"),
    ("HST/WFC3_UVIS.F275W", "HST WFC3/UVIS F275W"),
    ("HST/WFC3_UVIS.F475W", "HST WFC3/UVIS F475W"),
    ("HST/WFC3_UVIS.F625W", "HST WFC3/UVIS F625W"),
    ("HST/WFC3_UVIS.F850LP", "HST WFC3/UVIS F850LP"),
    ("HST/WFC3_IR.F105W", "HST WFC3/IR F105W"),
    ("HST/WFC3_IR.F125W", "HST WFC3/IR F125W"),
    ("HST/WFC3_IR.F160W", "HST WFC3/IR F160W"),
    ("JWST/NIRCam.F090W", "JWST NIRCam F090W"),
    ("JWST/NIRCam.F150W", "JWST NIRCam F150W"),
    ("JWST/NIRCam.F200W", "JWST NIRCam F200W"),
    ("JWST/NIRCam.F277W", "JWST NIRCam F277W"),
    ("JWST/NIRCam.F356W", "JWST NIRCam F356W"),
    ("JWST/NIRCam.F444W", "JWST NIRCam F444W"),
    
    # X-ray / UV (energy-integrating examples)
    ("XMM/OM.U", "XMM-OM UVW1"),
]

COMMON_FILTER_DICT = dict(COMMON_FILTERS)


def search_common(query: str) -> list:
    """Search the curated list (case-insensitive substring)."""
    q = query.lower()
    return [
        {"filterID": fid, "Description": desc, "source": "common"}
        for fid, desc in COMMON_FILTERS
        if q in (fid + " " + desc).lower()
    ]
