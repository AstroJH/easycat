"""Offline tests for the new SDSS (MaNGA / image) and DESI (image) modes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easycat.download import (DESIArchive, PanSTARRSArchive, SDSSArchive,
                              WISEArchive, ZTFArchive)
from easycat.download.base import FetchContext, SurveyArchive


class _FakeResponse:
    def __init__(self, status_code=200):
        self.status_code = status_code


class _FakeHttpError(Exception):
    def __init__(self, status_code=404):
        super().__init__(f"HTTP {status_code}")
        self.response = _FakeResponse(status_code)


class _FakeClient:
    """Minimal HttpClient stand-in that records URLs / writes files."""

    def __init__(self, fail_status=None):
        self.urls = []
        self.fail_status = fail_status

    def download_file(self, url, dest, *, overwrite=False):
        self.urls.append(url)
        if self.fail_status is not None:
            raise _FakeHttpError(self.fail_status)
        Path(dest).write_bytes(b"data")
        return Path(dest)


def _rows(n=1, ra=185.0, dec=15.0):
    return pd.DataFrame({
        "obj_id": [f"s{i}" for i in range(n)],
        "raj2000": [ra] * n,
        "dej2000": [dec] * n,
    })


# --------------------------------------------------------------------------- #
# SDSS MaNGA
# --------------------------------------------------------------------------- #
def test_manga_drp_url():
    a = SDSSArchive(mode="manga", manga_product="LOGRSS")
    url = a.manga_url(8138, 12704)
    assert url == ("https://data.sdss.org/sas/dr17/manga/spectro/redux/v3_1_1/"
                   "8138/stack/manga-8138-12704-LOGRSS.fits.gz")


def test_manga_dap_url():
    a = SDSSArchive(mode="manga", manga_dap="SPX-MILESHC-MASTARSSP",
                    manga_product="MAPS")
    url = a.manga_url(8138, 12704)
    assert url == ("https://data.sdss.org/sas/dr17/manga/spectro/analysis/"
                   "v3_1_1/3.1.0/SPX-MILESHC-MASTARSSP/8138/12704/"
                   "manga-8138-12704-MAPS-SPX-MILESHC-MASTARSSP.fits.gz")


def test_manga_invalid_product_raises():
    with pytest.raises(ValueError):
        SDSSArchive(mode="manga", manga_product="NOPE")
    with pytest.raises(ValueError):
        SDSSArchive(mode="manga", manga_dap="NOPE-DAP", manga_product="MAPS")


def test_manga_batch_uses_index_and_handles_404(tmp_path, monkeypatch):
    a = SDSSArchive(mode="manga", manga_product="LOGRSS", radius_arcsec=3)
    monkeypatch.setattr(a, "manga_index", lambda ctx: {
        "plate": np.array([8138]),
        "ifudsgn": np.array([12704]),
        "plateifu": np.array(["8138-12704"]),
        "objra": np.array([185.0]),
        "objdec": np.array([15.0]),
    })

    # hit
    client = _FakeClient()
    res = a.fetch_batch(_rows(), FetchContext(store_dir=tmp_path, client=client))
    assert res[0].success and client.urls[0].endswith(
        "8138/stack/manga-8138-12704-LOGRSS.fits.gz")
    assert (tmp_path / "s0.fits.gz").exists()

    # no MaNGA target at this position -> success, no data
    res = a.fetch_batch(_rows(ra=10.0, dec=10.0),
                        FetchContext(store_dir=tmp_path, client=_FakeClient()))
    assert res[0].success and res[0].data is None

    # product missing -> success, no data (not an error)
    res = a.fetch_batch(_rows(), FetchContext(store_dir=tmp_path,
                                              client=_FakeClient(fail_status=404)))
    assert res[0].success and res[0].data is None


# --------------------------------------------------------------------------- #
# SDSS images
# --------------------------------------------------------------------------- #
def test_sdss_image_url():
    a = SDSSArchive(mode="image", image_width=256, image_height=256,
                    image_scale=0.5)
    url = a.image_url(185.0, 15.0)
    assert url.startswith("https://skyserver.sdss.org/dr18/SkyServerWS/ImgCutout/getjpeg?")
    assert "ra=185.0" in url and "dec=15.0" in url
    assert "width=256" in url and "height=256" in url and "scale=0.5" in url


def test_sdss_image_batch_writes_jpeg(tmp_path):
    a = SDSSArchive(mode="image")
    ctx = FetchContext(store_dir=tmp_path, client=_FakeClient())
    res = a.fetch_batch(_rows(2), ctx)
    assert all(r.success for r in res)
    assert (tmp_path / "s0.jpg").exists() and (tmp_path / "s1.jpg").exists()


# --------------------------------------------------------------------------- #
# DESI / Legacy Surveys images
# --------------------------------------------------------------------------- #
def test_desi_image_url():
    a = DESIArchive(mode="image", image_layer="ls-dr10", image_format="fits",
                    image_size=128, image_pixscale=0.5)
    url = a.image_url(185.0, 15.0)
    assert url.startswith("https://www.legacysurvey.org/viewer/cutout.fits?")
    assert "layer=ls-dr10" in url and "size=128" in url and "pixscale=0.5" in url

    b = DESIArchive(mode="image", image_format="jpg")
    assert "cutout.jpg?" in b.image_url(1.0, 2.0)


def test_desi_image_batch_writes_fits(tmp_path):
    a = DESIArchive(mode="image", image_format="fits")
    client = _FakeClient()
    res = a.fetch_batch(_rows(), FetchContext(store_dir=tmp_path, client=client))
    assert res[0].success
    assert (tmp_path / "s0.fits").exists()
    assert "legacysurvey.org/viewer/cutout.fits" in client.urls[0]


# --------------------------------------------------------------------------- #
# batch sizes (per-source vs batched modes)
# --------------------------------------------------------------------------- #
def test_default_batch_sizes():
    assert SDSSArchive(mode="manga").default_batch_size == 1
    assert SDSSArchive(mode="image").default_batch_size == 1
    assert SDSSArchive(mode="both").default_batch_size == 32
    assert DESIArchive(mode="image").default_batch_size == 1
    assert DESIArchive(mode="photometry").default_batch_size == 16


def test_manga_fast_path_skips_drpall(tmp_path):
    """Rows carrying plate/ifudesign (or plateifu) need no drpall download."""
    a = SDSSArchive(mode="manga", manga_product="LOGRSS")

    def _no_index(ctx):        # must never be called
        raise AssertionError("drpall index should not be needed")

    a.manga_index = _no_index  # type: ignore[assignment]

    rows = pd.DataFrame({
        "obj_id": ["x", "y"],
        "raj2000": [1.0, 2.0],
        "dej2000": [3.0, 4.0],
        "plate": [8138, 8138],
        "ifudsgn": [12704, 12705],
    })
    client = _FakeClient()
    res = a.fetch_batch(rows, FetchContext(store_dir=tmp_path, client=client))
    assert [r.success for r in res] == [True, True]
    assert client.urls[0].endswith("8138/stack/manga-8138-12704-LOGRSS.fits.gz")
    assert client.urls[1].endswith("8138/stack/manga-8138-12705-LOGRSS.fits.gz")

    # plateifu string form
    rows2 = pd.DataFrame({"obj_id": ["z"], "raj2000": [1.0], "dej2000": [2.0],
                          "plateifu": ["8138-12704"]})
    assert SDSSArchive._plate_ifu_from_row(rows2.iloc[0]) == (8138, 12704)


# --------------------------------------------------------------------------- #
# output_path() must match what each archive actually writes (safety net)
# --------------------------------------------------------------------------- #
def test_builtin_output_paths(tmp_path):
    ctx = FetchContext(store_dir=tmp_path, client=_FakeClient())

    assert WISEArchive().output_path(ctx, "a").name == "a.fits"

    assert ZTFArchive().output_path(ctx, "a").name == "a.csv"
    assert ZTFArchive(store_format="fits").output_path(ctx, "a").name == "a.fits"

    assert SDSSArchive(mode="both").output_path(ctx, "a").name == "a.fits"
    assert SDSSArchive(mode="photometry").output_path(ctx, "a").name == "a.fits"
    assert SDSSArchive(mode="manga").output_path(ctx, "a").name == "a.fits.gz"
    assert SDSSArchive(mode="image").output_path(ctx, "a").name == "a.jpg"

    assert DESIArchive(mode="photometry").output_path(ctx, "a").name == "a.fits"
    assert DESIArchive(mode="spectra").output_path(ctx, "a").name == "a.fits"
    assert DESIArchive(mode="image").output_path(ctx, "a").name == "a.fits"
    assert DESIArchive(mode="image", image_format="jpg").output_path(ctx, "a").name == "a.jpg"

    assert PanSTARRSArchive().output_path(ctx, "a").name == "a.csv"
    assert PanSTARRSArchive(store_format="fits").output_path(ctx, "a").name == "a.fits"

    # the base class must stay conservative: unknown -> None
    class _Bare(SurveyArchive):
        name = "bare"
        def fetch_batch(self, rows, ctx):
            return []

    assert _Bare().output_path(ctx, "a") is None
