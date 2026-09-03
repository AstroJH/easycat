"""HEALPix helpers (nested scheme, as used by DESI).

easycat only needs to *convert* equatorial coordinates to the nested
HEALPix pixel number that DESI uses to organise its data files
(``nside = 64``).  The conversion is delegated to
`astropy-healpix <https://astropy-healpix.readthedocs.io>`_ (a small,
standard astropy-affiliated package); see
``docs/generated/healpix.md`` for the underlying concepts and formulas.

Reference: Górski et al. 2005, "HEALPix: A Framework for High-Resolution
Discretization and Fast Analysis of Data Distributed on the Sphere",
ApJ 622, 759.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
import astropy.units as u
from astropy_healpix import lonlat_to_healpix

__all__ = ["ra_dec_to_healpix"]


def ra_dec_to_healpix(
    ra: ArrayLike,
    dec: ArrayLike,
    nside: int = 64,
) -> NDArray[np.int64]:
    """Nested HEALPix pixel index from equatorial (RA, Dec) in degrees.

    Parameters
    ----------
    ra, dec : array_like
        ICRS coordinates in degrees.
    nside : int
        HEALPix resolution parameter (a power of two).  Default 64, the
        resolution used by DESI for its per-pixel data files.

    Returns
    -------
    ndarray of int64
        Nested pixel indices in ``[0, 12 * nside**2)``, same shape as input.
    """
    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    pix = lonlat_to_healpix(ra * u.deg, dec * u.deg, nside=nside, order="nested")
    return np.atleast_1d(np.asarray(pix, dtype=np.int64))
