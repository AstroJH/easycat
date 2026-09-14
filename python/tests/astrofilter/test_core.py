import numpy as np
import astropy.units as u
import pytest

from easycat.astrofilter.core import (
    AstroFilter,
    DetectorType,
    pivot_wavelength,
    synthetic_flux_lambda,
    synthetic_flux_nu,
)
from easycat.astrofilter.spectrum import Spectrum


def top_hat_filter(lo=4000.0, hi=6000.0, n=4001, det=DetectorType.PHOTON):
    wl = np.linspace(lo, hi, n) * u.AA
    tr = np.ones(n)
    return AstroFilter(wavelength=wl, transmission=tr, detector_type=det,
                       filter_id="test/tophat", metadata={})


def const_fnu_spectrum(fnu=3631.0):
    wl = np.linspace(3000.0, 9000.0, 3001) * u.AA
    return Spectrum(wl, np.full(3001, fnu) * u.Jy)


def test_pivot_tophat_closed_form():
    lo, hi = 4000.0, 6000.0
    # photon: sqrt( (hi^2-lo^2)/2 / ln(hi/lo) )
    photon = np.sqrt(((hi**2 - lo**2) / 2) / np.log(hi / lo))
    f = top_hat_filter(det=DetectorType.PHOTON)
    assert np.isclose(f.wl_pivot.to_value(u.AA), photon, rtol=1e-4)
    # energy: sqrt( (hi-lo) / (1/lo - 1/hi) ) = sqrt(lo*hi)
    energy = np.sqrt(lo * hi)
    f2 = top_hat_filter(det=DetectorType.ENERGY)
    assert np.isclose(f2.wl_pivot.to_value(u.AA), energy, rtol=1e-4)


def test_pivot_photon_vs_energy_differ():
    f_p = top_hat_filter(det=DetectorType.PHOTON)
    f_e = top_hat_filter(det=DetectorType.ENERGY)
    assert not np.isclose(f_p.wl_pivot.value, f_e.wl_pivot.value)


def test_const_fnu_ab_mag_is_zero():
    # A constant F_nu = 3631 Jy source has AB magnitude 0 in any filter,
    # for BOTH detector types (the detector-aware weighting is the point).
    for det in (DetectorType.PHOTON, DetectorType.ENERGY):
        f = top_hat_filter(det=det)
        m = f.ab_magnitude(const_fnu_spectrum())
        assert abs(m) < 1e-9


def test_const_fnu_synthetic_nu_is_constant():
    for det in (DetectorType.PHOTON, DetectorType.ENERGY):
        for (lo, hi) in [(4000, 6000), (6000, 8000)]:
            f = top_hat_filter(lo=lo, hi=hi, det=det)
            val = f.synthetic_flux(const_fnu_spectrum(), unit=u.Jy)
            assert np.isclose(val.to_value(u.Jy), 3631.0, rtol=1e-6)


def test_const_flam_synthetic_lambda_is_constant():
    wl = np.linspace(3000.0, 9000.0, 3001) * u.AA
    flam = np.full(3001, 1e-14) * (u.erg / u.s / u.cm**2 / u.AA)
    spec = Spectrum(wl, flam)
    for det in (DetectorType.PHOTON, DetectorType.ENERGY):
        f = top_hat_filter(det=det)
        val = f.synthetic_flux_lambda(spec)
        assert np.isclose(val.value, 1e-14, rtol=1e-6)


def test_fwhm_and_mean():
    f = top_hat_filter(lo=4000, hi=6000)
    assert np.isclose(f.fwhm.to_value(u.AA), 2000.0, rtol=1e-3)
    assert np.isclose(f.wl_mean.to_value(u.AA), 5000.0, rtol=1e-3)


def test_original_transmission_preserved():
    wl = np.linspace(4000, 6000, 101) * u.AA
    tr = np.linspace(0.2, 0.9, 101)
    f = AstroFilter(wavelength=wl, transmission=tr, filter_id="x/y")
    # original data untouched
    assert np.array_equal(f.transmission, tr)
    assert np.array_equal(f.wavelength.value, wl.value)
    # normalised is a *derived* array (peak = 1)
    assert np.isclose(f.transmission_normalized.max(), 1.0)
    # original values are not peak-normalised
    assert not np.isclose(f.transmission.max(), 1.0)


def test_matmul_sugar():
    f = top_hat_filter()
    out = f @ const_fnu_spectrum()
    assert out.unit.is_equivalent(u.Jy)
    assert np.isclose(out.to_value(u.Jy), 3631.0, rtol=1e-6)


def test_bad_detector_raises():
    with pytest.raises(ValueError):
        DetectorType.from_svo("banana")


def test_derived_quantities_computed_once_and_stable():
    from easycat.astrofilter.core import pivot_wavelength

    wl = np.linspace(4000, 7000, 301) * u.AA
    tr = np.exp(-0.5 * ((wl.value - 5500) / 300) ** 2)
    f = AstroFilter(wavelength=wl, transmission=tr,
                    detector_type=DetectorType.PHOTON, filter_id="T/once")

    # precomputed at construction (plain attributes) and stable
    assert f.wl_pivot == f.wl_pivot           # same Quantity each access
    assert f.wl_pivot is f.wl_pivot           # no live recomputation
    assert f.wl_mean == f.wl_mean
    assert f.fwhm == f.fwhm
    # equals the (independent) pure-function value
    assert np.isclose(f.wl_pivot.to_value(u.AA),
                      pivot_wavelength(wl, tr, DetectorType.PHOTON).to_value(u.AA))
    # mutating a returned array does not corrupt internal/cached data
    f.transmission_normalized[:] = 99.0
    assert np.isclose(f.transmission_normalized.max(), 1.0)
    assert np.allclose(f.transmission, tr)
    # metadata is a copy
    f.metadata["x"] = 1
    assert f.metadata.get("x") == 1
