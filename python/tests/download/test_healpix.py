import numpy as np
import pytest

from easycat.healpix import ra_dec_to_healpix


def test_pixel_range():
    rng = np.random.default_rng(42)
    ra = rng.uniform(0, 360, 2000)
    dec = rng.uniform(-90, 90, 2000)
    pix = ra_dec_to_healpix(ra, dec, nside=64)
    assert pix.dtype == np.int64
    assert pix.min() >= 0
    assert pix.max() < 12 * 64 * 64


def test_scalar_and_array():
    p0 = ra_dec_to_healpix(7.047287, 18.8275, nside=64)
    assert p0.ndim == 1 and len(p0) == 1
    p1 = ra_dec_to_healpix([7.0, 180.0], [18.8, 62.7], nside=64)
    assert p1.shape == (2,)


def test_nside_scaling():
    # A pixel at nside=64 contains 16 nested pixels at nside=256.
    pix64 = ra_dec_to_healpix(30.0, 20.0, nside=64)[0]
    pix256 = ra_dec_to_healpix(30.0, 20.0, nside=256)[0]
    assert pix256 // 16 == pix64


def test_ra_wrap_consistency():
    # Same sky position expressed at ra=0 and ra=360 must agree.
    a = ra_dec_to_healpix(0.5, 10.0, nside=64)[0]
    b = ra_dec_to_healpix(360.5, 10.0, nside=64)[0]
    assert a == b
