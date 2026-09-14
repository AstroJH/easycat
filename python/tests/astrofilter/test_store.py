import numpy as np
import astropy.units as u

from easycat.astrofilter.core import AstroFilter, DetectorType
from easycat.astrofilter.store import FilterStore, cache_filename


def make_filter():
    wl = np.linspace(4000, 7000, 300) * u.AA
    tr = np.exp(-0.5 * ((wl.value - 5500) / 300) ** 2)
    return AstroFilter(
        wavelength=wl,
        transmission=tr,
        detector_type=DetectorType.ENERGY,
        filter_id="TEST/Inst.band",
        facility="TEST",
        instrument="Inst",
        band="band",
        metadata={
            "DetectorType": "0",
            "WavelengthPivot": "1234.5",
            "ZeroPoint": "9.99",
            "Description": "test filter",
        },
    )


def test_cache_filename():
    assert cache_filename("HST/WFC3_IR.F160W") == "HST__WFC3_IR.F160W.ecsv"


def test_roundtrip(tmp_path):
    store = FilterStore(tmp_path)
    f = make_filter()
    store.save(f)
    # a *fresh* store must read the file back from disk
    store2 = FilterStore(tmp_path)
    f2 = store2.load("TEST/Inst.band")
    assert f2 is not None
    assert f2.filter_id == f.filter_id
    assert f2.detector_type is DetectorType.ENERGY
    assert np.allclose(f2.transmission, f.transmission)
    assert np.allclose(f2.wavelength.value, f.wavelength.value)
    assert f2.metadata["ZeroPoint"] == "9.99"
    assert np.isclose(
        f2.wl_pivot.to_value(u.AA), f.wl_pivot.to_value(u.AA), rtol=1e-8
    )


def test_load_missing_returns_none(tmp_path):
    store = FilterStore(tmp_path)
    assert store.load("NOPE/Nope.x") is None


def test_zp_vega_persisted_loads_offline(tmp_path, monkeypatch):
    """zp_vega computed once is persisted; loading it must not hit network."""
    store = FilterStore(tmp_path)
    f = make_filter()
    f.set_zp_vega(1234.5)  # e.g. computed during an online fetch
    store.save(f)

    store2 = FilterStore(tmp_path)          # fresh store -> read from disk
    f2 = store2.load("TEST/Inst.band")
    assert f2._zp_vega_jy is not None

    def _no_network(*a, **k):
        raise AssertionError("network access during offline zp_vega load")

    monkeypatch.setattr("easycat.astrofilter.svo2.get_vega_spectrum", _no_network)
    assert np.isclose(f2.zp_vega.to_value(u.Jy), 1234.5)
    assert np.isclose(f2.zp_vega.to_value(u.Jy), 1234.5)  # stable
