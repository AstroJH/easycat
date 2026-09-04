"""Astronomical filter (passband) representation and science.

The physics here follows the SVO Filter Profile Service conventions and
the standard photometric-system literature (e.g. Tokunaga & Vacca 2005;
Bessell & Murphy 2012; Casagrande & VandenBerg 2014).

Detector types
--------------
* **Photon-counting** (``DetectorType = 1``): the tabulated response
  :math:`R(\\lambda)` is the *photon* response.  The band-averaged flux
  density is weighted by photon rate, i.e. by :math:`\\lambda R(\\lambda)`.
* **Energy-integrating** (``DetectorType = 0``): :math:`R(\\lambda)` is the
  *energy* response and the band average is weighted by :math:`R(\\lambda)`
  itself.

Both the **pivot wavelength** and the **synthetic photometry** depend on
the detector type (verified against SVO metadata):

* photon counting:
  :math:`\\lambda_\\mathrm{piv} = \\sqrt{\\int \\lambda R \\,d\\lambda /
  \\int R/\\lambda \\,d\\lambda}`
* energy integrating:
  :math:`\\lambda_\\mathrm{piv} = \\sqrt{\\int R \\,d\\lambda /
  \\int R/\\lambda^2 \\,d\\lambda}`

An :class:`AstroFilter` is effectively immutable: the original transmission
curve is preserved verbatim and invariant derived quantities are computed once.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import astropy.units as u
from astropy.constants import c as SPEED_OF_LIGHT
from astropy.units import Quantity
from scipy.interpolate import interp1d

from .spectrum import Spectrum, FLAM

__all__ = [
    "DetectorType",
    "AstroFilter",
    "synthetic_flux_lambda",
    "synthetic_flux_nu",
    "pivot_wavelength",
]

FLAM_CM = u.erg / u.s / u.cm**2 / u.cm


class DetectorType(Enum):
    """Detector response type of a passband."""

    ENERGY = 0
    PHOTON = 1

    @classmethod
    def from_svo(cls, value: Any) -> "DetectorType":
        if isinstance(value, DetectorType):
            return value
        s = str(value).strip().lower()
        if s in ("1", "photon", "photon_counter", "photon-counting"):
            return cls.PHOTON
        if s in ("0", "energy", "energy_counter", "energy-integrating"):
            return cls.ENERGY
        raise ValueError(f"cannot parse detector type: {value!r}")

    def __str__(self) -> str:
        return "photon" if self is DetectorType.PHOTON else "energy"

def pivot_wavelength(
    wavelength: Quantity,
    transmission: np.ndarray,
    detector_type: DetectorType,
) -> Quantity:
    """Detector-aware pivot wavelength (see module docstring)."""
    wl = np.asarray(wavelength.to_value(u.AA), dtype=float)
    tr = np.asarray(transmission, dtype=float)
    if detector_type is DetectorType.PHOTON:
        num = np.trapezoid(wl * tr, wl)
        den = np.trapezoid(tr / wl, wl)
    else:
        num = np.trapezoid(tr, wl)
        den = np.trapezoid(tr / wl**2, wl)
    if den <= 0 or num <= 0:
        raise ValueError("cannot compute pivot wavelength: non-positive integral")
    return np.sqrt(num / den) * u.AA


def synthetic_flux_lambda(
    spectrum: Spectrum,
    wavelength: Quantity,
    transmission: np.ndarray,
    detector_type: DetectorType,
    *,
    unit: u.Unit = FLAM,
) -> Quantity:
    """Band-averaged :math:`\\langle F_\\lambda\\rangle` through a passband.

    Photon counting:
        :math:`\\langle F_\\lambda\\rangle = \\int F_\\lambda \\lambda R
        \\,d\\lambda / \\int \\lambda R \\,d\\lambda`
    Energy integrating:
        :math:`\\langle F_\\lambda\\rangle = \\int F_\\lambda R \\,d\\lambda
        / \\int R \\,d\\lambda`
    """
    wl = np.asarray(wavelength.to_value(u.AA), dtype=float)
    tr = np.asarray(transmission, dtype=float)
    f_lam = np.asarray(
        spectrum(wl * u.AA, unit=FLAM).to_value(FLAM), dtype=float
    )
    if detector_type is DetectorType.PHOTON:
        num = np.trapezoid(f_lam * wl * tr, wl)
        den = np.trapezoid(wl * tr, wl)
    else:
        num = np.trapezoid(f_lam * tr, wl)
        den = np.trapezoid(tr, wl)
    if den <= 0:
        raise ValueError("cannot compute synthetic flux: zero denominator")
    return Quantity(num / den, FLAM).to(unit)


def synthetic_flux_nu(
    spectrum: Spectrum,
    wavelength: Quantity,
    transmission: np.ndarray,
    detector_type: DetectorType,
    *,
    unit: u.Unit = u.Jy,
) -> Quantity:
    """Band-averaged :math:`\\langle F_\\nu\\rangle` through a passband.

    Uses the detector-aware pivot wavelength:
    :math:`\\langle F_\\nu\\rangle = \\langle F_\\lambda\\rangle\\,
    \\lambda_\\mathrm{piv}^2 / c`.
    """
    f_lam = synthetic_flux_lambda(
        spectrum, wavelength, transmission, detector_type, unit=FLAM_CM
    )
    piv = pivot_wavelength(wavelength, transmission, detector_type).to(u.cm)
    f_nu = f_lam * piv**2 / SPEED_OF_LIGHT.cgs
    return f_nu.to(unit)


# --------------------------------------------------------------------------- #
# filter class
# --------------------------------------------------------------------------- #
class AstroFilter:
    """A single astronomical filter / passband.

    The object is effectively **immutable**: the original transmission curve
    and wavelength grid are preserved verbatim (``wavelength`` /
    ``transmission``), and geometry-derived quantities (``wl_pivot``,
    ``wl_mean``, ``fwhm``, ``wl_min``, ``wl_max``, ``transmission_normalized``)
    are invariant and therefore computed **once at construction**;
    ``zp_vega`` is computed once on first access and cached.

    Parameters
    ----------
    wavelength : Quantity
        1-D wavelength grid (any length unit).
    transmission : array_like
        Original response curve (photon or energy response, as given by
        the provider), same length as ``wavelength``.
    detector_type : DetectorType or str/int
        ``DetectorType.PHOTON`` (1) or ``DetectorType.ENERGY`` (0).
    filter_id : str
        Full identifier, e.g. ``"SLOAN/SDSS.r"``.  If omitted it is
        assembled from ``facility.instrument.band``.
    facility, instrument, band : str
        Identity components (also stored in ``metadata``).
    metadata : dict
        Provider metadata (SVO parameters, e.g. ``WavelengthPivot``,
        ``ZeroPoint``, ``MagSys``, ...) preserved for reference.
    """

    def __init__(
        self,
        wavelength: Quantity,
        transmission: np.ndarray,
        detector_type: DetectorType = DetectorType.PHOTON,
        filter_id: str = "",
        facility: str = "",
        instrument: str = "",
        band: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if isinstance(detector_type, (str, int)):
            detector_type = DetectorType.from_svo(detector_type)

        wl = np.asarray(wavelength.to_value(u.AA), dtype=float)
        tr = np.asarray(transmission, dtype=float)

        if wl.ndim != 1 or tr.ndim != 1:
            raise ValueError("wavelength and transmission must be 1-D")
        if len(wl) != len(tr):
            raise ValueError("wavelength and transmission must have equal length")
        if len(wl) < 2:
            raise ValueError("a filter needs at least 2 points")
        
        # preserve original data
        self._wl_aa = wl.copy()
        self._transmission = tr.copy()
        self.detector_type = detector_type
        self.facility = facility
        self.instrument = instrument
        self.band = band
        self.metadata = dict(metadata or {})
        if filter_id:
            self.filter_id = filter_id
        elif facility and instrument and band:
            self.filter_id = f"{facility}.{instrument}.{band}"
        else:
            self.filter_id = self.metadata.get("filterID", "")

        # --- invariant derived quantities: computed once at creation ---- #
        self._transmission_normalized = self._normalize(self._transmission)
        self.wl_pivot = pivot_wavelength(
            wl * u.AA, self._transmission, self.detector_type
        )
        self.wl_mean = (
            np.trapezoid(wl * tr, wl) / np.trapezoid(tr, wl) * u.AA
        )
        self.fwhm = self._calc_fwhm(wl, tr)
        self.wl_min = wl[0] * u.AA
        self.wl_max = wl[-1] * u.AA
        self.zp_ab = 3631.0 * u.Jy
        self._zp_vega_jy: Optional[Quantity] = None

    # ------------------------------------------------------------------ #
    # identity
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return self.filter_id

    def __repr__(self) -> str:
        return (
            f"AstroFilter({self.filter_id!r}, detector={self.detector_type}, "
            f"n={len(self._wl_aa)})"
        )

    # ------------------------------------------------------------------ #
    # original data
    # ------------------------------------------------------------------ #
    @property
    def wavelength(self) -> Quantity:
        return self._wl_aa * u.AA

    @property
    def transmission(self) -> np.ndarray:
        """Original (unnormalised) response curve."""
        return self._transmission.copy()

    # ------------------------------------------------------------------ #
    # derived quantities (computed once; see class docstring)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize(tr: np.ndarray) -> np.ndarray:
        peak = float(np.max(tr))
        return tr / peak if peak > 0 else tr.copy()

    @staticmethod
    def _calc_fwhm(wl: np.ndarray, tr: np.ndarray) -> Quantity:
        half = float(np.max(tr)) / 2
        inside = tr >= half
        if not np.any(inside):
            return 0.0 * u.AA
        idx = np.flatnonzero(inside)
        return (wl[idx[-1]] - wl[idx[0]]) * u.AA

    @property
    def transmission_normalized(self) -> np.ndarray:
        """Peak-normalised response (copy; cached value at construction)."""
        return self._transmission_normalized.copy()

    # ------------------------------------------------------------------ #
    # science
    # ------------------------------------------------------------------ #
    def synthetic_flux_lambda(
        self, spectrum: Spectrum, *, unit: u.Unit = FLAM
    ) -> Quantity:
        """Band-averaged :math:`\\langle F_\\lambda\\rangle`."""
        return synthetic_flux_lambda(
            spectrum, self.wavelength, self._transmission,
            self.detector_type, unit=unit,
        )

    def synthetic_flux(
        self, spectrum: Spectrum, *, unit: u.Unit = u.Jy
    ) -> Quantity:
        """Band-averaged :math:`\\langle F_\\nu\\rangle` (default: Jy)."""
        return synthetic_flux_nu(
            spectrum, self.wavelength, self._transmission,
            self.detector_type, unit=unit,
        )

    # -- zero points ---------------------------------------------------- #
    # ``zp_ab`` is set as a plain attribute at construction (3631 Jy).

    @property
    def zp_vega(self) -> Quantity:
        """Vega zero-point flux density in this band (Jy).

        Computed once as the synthetic flux of the SVO Vega reference
        spectrum and kept in ``_zp_vega_jy``; may be restored directly
        from the persisted cache (see :meth:`set_zp_vega`).  Compare with
        ``metadata["ZeroPoint"]``.
        """
        if self._zp_vega_jy is None:
            from .svo2 import get_vega_spectrum

            self._zp_vega_jy = self.synthetic_flux(
                get_vega_spectrum(), unit=u.Jy
            )
        return self._zp_vega_jy

    def set_zp_vega(self, value) -> None:
        """Store a precomputed Vega zero-point (bypasses computation)."""
        self._zp_vega_jy = Quantity(value, u.Jy)

    # -- magnitudes ----------------------------------------------------- #
    def ab_magnitude(self, spectrum: Spectrum) -> float:
        """AB magnitude of ``spectrum`` through this filter."""
        f_nu = self.synthetic_flux(spectrum, unit=u.Jy)
        return float(-2.5 * np.log10((f_nu / self.zp_ab).to_value("")))

    def vega_magnitude(self, spectrum: Spectrum) -> float:
        """Vega magnitude of ``spectrum`` through this filter."""
        f_nu = self.synthetic_flux(spectrum, unit=u.Jy)
        return float(-2.5 * np.log10((f_nu / self.zp_vega).to_value("")))

    # -- interpolation / plotting --------------------------------------- #
    def transmission_interp(self, kind: str = "linear") -> interp1d:
        """Interpolation of the *original* transmission over wavelength."""
        return interp1d(
            self._wl_aa, self._transmission, kind=kind,
            bounds_error=False, fill_value=0.0,
        )

    def plot(self, ax=None, *, normalized: bool = True, **kwargs):
        """Plot the transmission curve.

        Parameters
        ----------
        ax : matplotlib Axes or None
        normalized : bool
            Plot the peak-normalised response (default) or the original.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        y = self.transmission_normalized if normalized else self._transmission
        ax.plot(self._wl_aa, y, label=self.filter_id, **kwargs)
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Transmission" + (" (peak=1)" if normalized else ""))
        return ax

    # ------------------------------------------------------------------ #
    # syntactic sugar
    # ------------------------------------------------------------------ #
    def __matmul__(self, spectrum: Spectrum) -> Quantity:
        """``filter @ spectrum`` -> synthetic flux density in Jy."""
        return self.synthetic_flux(spectrum, unit=u.Jy)
