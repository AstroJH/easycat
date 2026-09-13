"""Offline tests for the ESO MUSE Archive."""
from __future__ import annotations

from pathlib import Path

import pytest

from easycat.download import HttpError, MUSEArchive
from easycat.download.survey.muse import dataportal_url, valid_dp_id


DP_ID = "ADP.2024-04-30T18:20:44.624"


class _FakeClient:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def download_file(self, url, dest, **kwargs):
        self.calls.append({"url": url, "dest": Path(dest), "kwargs": dict(kwargs)})
        if self.error is not None:
            raise self.error
        path = Path(dest)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SIMPLE  =                    T")
        return path


def test_muse_dp_id_validation_and_url():
    assert valid_dp_id(DP_ID)
    assert not valid_dp_id("ADP../evil")
    assert not valid_dp_id("..")
    assert not valid_dp_id("")
    assert dataportal_url(DP_ID) == (
        f"https://dataportal.eso.org/dataPortal/file/{DP_ID}"
    )
    assert MUSEArchive().url_for(DP_ID).endswith(f"/{DP_ID}")
    with pytest.raises(ValueError):
        MUSEArchive().url_for("ADP../evil")


def test_muse_mode_validation_and_custom_base():
    archive = MUSEArchive(mode="whitelight", dataportal_base="https://mirror.test/files/")
    assert archive.url_for(DP_ID) == f"https://mirror.test/files/{DP_ID}"
    assert archive.item_metadata()["product"] == "muse-whitelight"
    with pytest.raises(ValueError):
        MUSEArchive(mode="mosaic")


def test_muse_fetch_one_defaults_and_override(tmp_path):
    callback = lambda done, total: None
    archive = MUSEArchive()
    client = _FakeClient()
    dest = tmp_path / "custom" / f"{DP_ID}.fits"

    result = archive.fetch_one(
        DP_ID, client=client, dest=dest, progress=callback,
        resume="auto",
    )

    assert result.success and result.data == dest
    assert dest.exists()
    assert result.meta["url"].endswith(f"/{DP_ID}")
    assert result.meta["dest"] == str(dest)
    assert result.meta["product"] == "muse-cube"
    assert result.meta["source"] == "ESO Phase 3"
    assert result.meta["size"] == dest.stat().st_size
    kwargs = client.calls[0]["kwargs"]
    assert kwargs["resume"] == "auto"          # per-call override wins
    assert kwargs["validate"] == "fits"
    assert kwargs["progress"] is callback
    assert "overwrite" not in kwargs


def test_muse_default_download_avoids_unreliable_range(tmp_path):
    archive = MUSEArchive()
    client = _FakeClient()

    result = archive.fetch_one(DP_ID, client=client, store_dir=tmp_path)

    assert result.success
    assert (tmp_path / f"{DP_ID}.fits").exists()
    kwargs = client.calls[0]["kwargs"]
    assert kwargs["resume"] == "never"
    assert kwargs["validate"] == "fits"


def test_muse_404_is_no_data(tmp_path):
    archive = MUSEArchive()
    client = _FakeClient(HttpError("missing", status=404))

    result = archive.fetch_one(DP_ID, client=client, store_dir=tmp_path)

    assert result.success
    assert result.data is None
    assert result.meta["url"].endswith(f"/{DP_ID}")


def test_muse_invalid_identifier_is_failure(tmp_path):
    archive = MUSEArchive()
    result = archive.fetch_one("ADP../evil", client=_FakeClient(), store_dir=tmp_path)

    assert not result.success
    assert "invalid ESO dp_id" in result.error


def test_muse_resolve_only_does_not_write(tmp_path):
    archive = MUSEArchive()
    client = _FakeClient()
    dest = tmp_path / f"{DP_ID}.fits"

    result = archive.fetch_one(DP_ID, client=client, dest=dest, download=False)

    assert result.success
    assert result.data is None
    assert not dest.exists()
    assert client.calls == []
    assert result.meta["url"].endswith(f"/{DP_ID}")
    assert result.meta["download"] is False
