"""Live tests against SVO2 (skipped when the service is unreachable)."""
import astropy.units as u
import pytest

from easycat.astrofilter import FilterDB, get_vega_spectrum


def _online():
    try:
        import requests

        return requests.get(
            "https://svo2.cab.inta-csic.es/theory/fps/fps.php",
            params={"ID": "SLOAN/SDSS.r"}, timeout=15,
        ).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _online(), reason="SVO2 unreachable")


@pytest.fixture()
def db(tmp_path, monkeypatch):
    # keep filter + reference-spectrum caches inside the tmp dir
    monkeypatch.setenv("EASYCAT_DATA", str(tmp_path))
    return FilterDB(cache_dir=tmp_path)


def test_pivot_matches_svo(db):
    for fid, det in [("SLOAN/SDSS.r", "photon"),
                     ("HST/WFC3_IR.F160W", "photon"),
                     ("WISE/WISE.W1", "energy"),
                     ("Generic/Johnson.U", "energy")]:
        f = db.fetch(fid)
        assert str(f.detector_type) == det
        svp = float(f.metadata["WavelengthPivot"])
        ours = float(f.wl_pivot.to_value(u.AA))
        assert abs(svp - ours) / svp < 1e-4


def test_vega_zeropoint_close_to_svo(db):
    vega = get_vega_spectrum()
    for fid in ["SLOAN/SDSS.r", "HST/WFC3_IR.F160W", "Generic/Johnson.U"]:
        f = db.fetch(fid)
        ours = f.synthetic_flux(vega, unit=u.Jy).to_value(u.Jy)
        svo = float(f.metadata["ZeroPoint"])
        assert abs(ours - svo) / svo < 1e-3


def test_vega_magnitude_is_zero(db):
    vega = get_vega_spectrum()
    f = db.fetch("SLOAN/SDSS.r")
    assert abs(f.vega_magnitude(vega)) < 1e-6
