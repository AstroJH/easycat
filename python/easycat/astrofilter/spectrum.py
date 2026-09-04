"""A small, self-contained spectral flux-density ("SED") interface.

This replaces the previous dependency on ``raven.spectrum``: easycat
defines its own minimal, unit-correct :class:`Spectrum` that can hold a
flux density in either :math:`F_\\lambda` or :math:`F_\\nu` and evaluate
it on arbitrary wavelength grids (used by :class:`AstroFilter` for
synthetic photometry).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import astropy.units as u
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.units import Quantity
from scipy.interpolate import interp1d

FLAM = u.erg / u.s / u.cm**2 / u.AA # F_lambda: per Angstrom
FNU = u.Jy # F_nu

__all__ = ["Spectrum", "flam_to_fnu", "fnu_to_flam", "FLAM", "FNU"]


def flam_to_fnu(flam: Quantity, wavelength: Quantity) -> Quantity:
    """Convert a :math:`F_\\lambda` flux density to :math:`F_\\nu`.

    Uses :math:`F_\\nu = F_\\lambda \\lambda^2 / c`.
    """
    wl_cm = wavelength.to(u.cm)
    flam_cm = flam.to(u.erg / u.s / u.cm**2 / u.cm)
    return (flam_cm * wl_cm**2 / SPEED_OF_LIGHT.cgs).to(FNU)


def fnu_to_flam(fnu: Quantity, wavelength: Quantity) -> Quantity:
    """Convert a :math:`F_\\nu` flux density to :math:`F_\\lambda`."""
    wl_cm = wavelength.to(u.cm)
    return (fnu * SPEED_OF_LIGHT.cgs / wl_cm**2).to(
        u.erg / u.s / u.cm**2 / u.cm
    ).to(FLAM)


@dataclass
class Spectrum:
    """A 1-D spectral flux density.

    Parameters
    ----------
    wavelength : astropy Quantity
        1-D array of wavelengths (any length unit).
    flux : astropy Quantity
        1-D array of flux densities; the unit decides whether the values
        are :math:`F_\\lambda` (energy per area/time/wavelength) or
        :math:`F_\\nu` (e.g. Jy).
    name : str
        Optional label.
    kind : str
        Interpolation kind passed to ``scipy.interpolate.interp1d``
        (``"linear"`` default; ``"cubic"`` or ``"loglog"`` supported).

    Notes
    -----
    The spectrum is *immutable by convention*: conversions return new
    objects.  Values are kept in the native form they were provided in.
    """

    wavelength: Quantity
    flux: Quantity
    name: str = ""
    kind: str = "linear"

    def __post_init__(self) -> None:
        wl = np.asarray(self.wavelength.to_value(u.AA), dtype=float)
        fl = np.asarray(self.flux.value, dtype=float)
        if wl.ndim != 1 or fl.ndim != 1:
            raise ValueError("wavelength and flux must be 1-D")
        if len(wl) != len(fl):
            raise ValueError("wavelength and flux must have the same length")
        if len(wl) < 2:
            raise ValueError("a spectrum needs at least 2 points")
        if np.any(np.diff(wl) <= 0):
            order = np.argsort(wl)
            wl = wl[order]
            fl = fl[order]
        self._wl_aa = wl
        self._flux_value = fl
        self._interp = self._build_interp()

    # ------------------------------------------------------------------ #
    # identity / units
    # ------------------------------------------------------------------ #
    @property
    def is_flam(self) -> bool:
        """True if the stored flux is :math:`F_\\lambda`."""
        return self.flux.unit.is_equivalent(FLAM)

    @property
    def is_fnu(self) -> bool:
        """True if the stored flux is :math:`F_\\nu`."""
        return self.flux.unit.is_equivalent(FNU)

    def _build_interp(self):
        x = self._wl_aa
        y = self._flux_value
        if self.kind == "loglog":
            mask = (x > 0) & (y > 0)
            return interp1d(
                np.log10(x[mask]), np.log10(y[mask]),
                kind="linear", bounds_error=False, fill_value=np.nan,
            )
        return interp1d(x, y, kind=self.kind, bounds_error=False,
                        fill_value=np.nan)

    # ------------------------------------------------------------------ #
    # conversions
    # ------------------------------------------------------------------ #
    def to_flam(self) -> "Spectrum":
        """Return a copy with flux as :math:`F_\\lambda`."""
        if self.is_flam:
            return Spectrum(self.wavelength, self.flux, name=self.name, kind=self.kind)
        return Spectrum(
            self.wavelength,
            fnu_to_flam(self.flux, self.wavelength),
            name=self.name, kind=self.kind,
        )

    def to_fnu(self) -> "Spectrum":
        """Return a copy with flux as :math:`F_\\nu` (Jy)."""
        if self.is_fnu:
            return Spectrum(self.wavelength, self.flux, name=self.name, kind=self.kind)
        return Spectrum(
            self.wavelength,
            flam_to_fnu(self.flux, self.wavelength),
            name=self.name, kind=self.kind,
        )

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    def __call__(
        self,
        wavelength: Quantity,
        *,
        unit: u.Unit = FLAM,
    ) -> Quantity:
        """Evaluate the (interpolated) flux density at ``wavelength``.

        Parameters
        ----------
        wavelength : Quantity
            Target wavelength(s).
        unit : astropy Unit
            Output unit; ``FLAM`` (per Å) by default.  If the unit is a
            frequency flux density (e.g. Jy) the values are converted
            accordingly.

        Returns
        -------
        Quantity
            Flux density at ``wavelength`` in ``unit``.
        """
        wl_aa = np.asarray(wavelength.to_value(u.AA), dtype=float)
        scalar = wl_aa.ndim == 0
        wl_aa = np.atleast_1d(wl_aa)

        if self.kind == "loglog":
            logv = self._interp(np.log10(wl_aa))
            val = 10.0 ** logv
        else:
            val = self._interp(wl_aa)

        val = np.where(np.isfinite(val), val, np.nan)

        if self.is_flam:
            f = val * FLAM
        else:
            f = val * FNU

        # Normalise to F_lambda for the conversion step.
        if not f.unit.is_equivalent(FLAM):
            f = fnu_to_flam(f, wl_aa * u.AA)

        if unit.is_equivalent(FLAM):
            out = f.to(unit)
        elif unit.is_equivalent(FNU):
            out = flam_to_fnu(f, wl_aa * u.AA).to(unit)
        else:
            raise u.UnitConversionError(
                f"cannot convert {f.unit} to {unit}"
            )
        return out[0] if scalar else out

    # ------------------------------------------------------------------ #
    # constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(
        cls,
        wavelength: Quantity,
        flux: Quantity,
        *,
        name: str = "",
        kind: str = "linear",
    ) -> "Spectrum":
        return cls(wavelength, flux, name=name, kind=kind)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Spectrum(name={self.name!r}, n={len(self._wl_aa)}, "
            f"unit={self.flux.unit})"
        )
