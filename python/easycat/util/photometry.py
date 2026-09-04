"""Magnitude <-> flux density conversions.

Vega-based conversions use the filter's Vega zero-point flux density;
AB conversions use the constant 3631 Jy.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u
from astropy.units import Quantity

__all__ = ["mag2flux", "flux2mag", "ab_mag2flux", "ab_flux2mag"]

MAG2FLUX = 10.0 ** 0.4  # = 10^(0.4) used below


def mag2flux(mag: ArrayLike, zp: Quantity) -> Quantity:
    """Convert a magnitude to a flux density: ``flux = zp * 10^(-0.4 mag)``.

    Parameters
    ----------
    mag : array_like
        Magnitude(s).
    zp : Quantity
        Zero-point flux density (e.g. ``filter.zp_vega`` in Jy).
    """
    mag = np.asarray(mag, dtype=float)
    return (10.0 ** (-0.4 * mag)) * zp


def flux2mag(flux: ArrayLike, zp: Quantity) -> ArrayLike:
    """Convert a flux density to a magnitude: ``mag = -2.5 log10(flux/zp)``."""
    flux = np.asarray(flux, dtype=float)
    zp_val = float(zp.to_value(zp.unit))
    return -2.5 * np.log10(flux / zp_val)


def ab_mag2flux(mag: ArrayLike) -> Quantity:
    """AB magnitude -> flux density (Jy); AB zero point is 3631 Jy."""
    return mag2flux(mag, 3631.0 * u.Jy)


def ab_flux2mag(flux: ArrayLike) -> ArrayLike:
    """Flux density (Jy) -> AB magnitude."""
    return flux2mag(flux, 3631.0 * u.Jy)
