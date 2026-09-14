import numpy as np
import astropy.units as u
import pytest

from easycat.astrofilter.spectrum import Spectrum, flam_to_fnu, fnu_to_flam


def test_flam_fnu_roundtrip():
    wl = np.array([4000.0, 5000.0, 6000.0, 8000.0]) * u.AA
    flam = np.full(4, 1e-14) * (u.erg / u.s / u.cm**2 / u.AA)
    s = Spectrum(wl, flam, name="flat")
    snu = s.to_fnu()
    back = snu.to_flam()
    assert np.allclose(back.flux.value, flam.value, rtol=1e-10)


def test_fnu_conversion_value():
    # F_nu = F_lambda * lambda^2 / c ; check the Jy value at 5000 AA
    wl = 5000.0 * u.AA
    flam = 1e-14 * (u.erg / u.s / u.cm**2 / u.AA)
    fnu = flam_to_fnu(flam, wl)
    expected = flam.to(u.erg / u.s / u.cm**2 / u.cm) * wl.to(u.cm) ** 2 / (
        2.99792458e10 * u.cm / u.s
    )
    assert np.isclose(fnu.to_value(u.Jy), expected.to_value(u.Jy), rtol=1e-8)


def test_evaluate_scalar_and_array():
    wl = np.array([4000.0, 6000.0]) * u.AA
    flam = np.array([1e-14, 2e-14]) * (u.erg / u.s / u.cm**2 / u.AA)
    s = Spectrum(wl, flam)
    v = s(5000 * u.AA)
    assert v.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.AA)
    assert np.isclose(v.value, 1.5e-14)
    arr = s(np.array([4000.0, 6000.0]) * u.AA)
    assert arr.shape == (2,)


def test_evaluate_in_fnu():
    wl = np.array([4000.0, 6000.0]) * u.AA
    flam = np.full(2, 1e-14) * (u.erg / u.s / u.cm**2 / u.AA)
    s = Spectrum(wl, flam)
    fnu = s(5000 * u.AA, unit=u.Jy)
    assert fnu.unit.is_equivalent(u.Jy)
    # flat F_lambda => F_nu increases as lambda^2
    fnu6000 = s(6000 * u.AA, unit=u.Jy)
    assert fnu6000.value > fnu.value
