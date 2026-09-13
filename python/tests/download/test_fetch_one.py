"""Tests for the single-item Archive API and provenance metadata."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from easycat.download import HttpError, SDSSArchive
from easycat.download.base import FetchContext, ItemResult, SurveyArchive


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


class _SingleArchive(SurveyArchive):
    name = "single"
    default_batch_size = 1

    def __init__(self, *, empty=False, **kwargs):
        super().__init__(**kwargs)
        self.empty = empty

    def item_metadata(self, row=None):
        meta = super().item_metadata(row)
        meta["version"] = "v1"
        return meta

    def fetch_batch(self, rows, ctx):
        results = []
        for _, row in rows.iterrows():
            obj_id = ctx.row_id(row)
            if self.empty:
                result = ItemResult(obj_id=obj_id, success=True, data=None)
                results.append(self.enrich_result(result, row=row))
                continue

            url = f"https://example.test/{obj_id}.fits"
            dest = ctx.dest(
                obj_id, ctx.store_dir / f"{obj_id}.fits", row=row,
            )
            try:
                self._download_file(ctx, url, dest)
                result = ItemResult(obj_id=obj_id, success=True, data=dest)
                results.append(
                    self.enrich_result(result, row=row, url=url, dest=dest)
                )
            except Exception as exc:
                result = ItemResult(obj_id=obj_id, success=False, error=repr(exc))
                results.append(self.enrich_result(result, row=row, url=url))
        return results

    def output_path(self, ctx, obj_id, *, row=None):
        group = str(row.get("group", "default")) if row is not None else "default"
        return ctx.store_dir / group / f"{obj_id}.fits"


def test_fetch_one_download_metadata_and_download_kwargs(tmp_path):
    callback = lambda done, total: None
    archive = _SingleArchive(download_kwargs={"trust_existing": True})
    client = _FakeClient()
    dest = tmp_path / "chosen" / "source.fits"

    result = archive.fetch_one(
        {"obj_id": "s1", "ra": 1.0, "dec": 2.0, "group": "g"},
        client=client,
        dest=dest,
        progress=callback,
        validate="fits",
        checksum=("sha256", "abc"),
        resume="never",
    )

    assert result.success
    assert result.data == dest
    assert dest.exists()
    assert result.meta["url"] == "https://example.test/s1.fits"
    assert result.meta["dest"] == str(dest)
    assert result.meta["size"] == dest.stat().st_size
    assert result.meta["product"] == "single"
    assert result.meta["version"] == "v1"
    assert result.meta["download"] is True
    assert result.meta["checksum"] == ("sha256", "abc")
    json.dumps(result.meta)  # provenance must be JSON/YAML-friendly

    kwargs = client.calls[0]["kwargs"]
    assert kwargs["trust_existing"] is True
    assert kwargs["validate"] == "fits"
    assert kwargs["checksum"] == ("sha256", "abc")
    assert kwargs["resume"] == "never"
    assert kwargs["progress"] is callback
    assert "overwrite" not in kwargs


def test_fetch_one_no_data_and_failure_are_distinct(tmp_path):
    empty = _SingleArchive(empty=True).fetch_one(
        "s1", client=_FakeClient(), store_dir=tmp_path,
    )
    assert empty.success and empty.data is None

    failed = _SingleArchive().fetch_one(
        "s1", client=_FakeClient(error=RuntimeError("boom")), store_dir=tmp_path,
    )
    assert not failed.success
    assert "boom" in failed.error


def test_fetch_one_requires_destination_for_real_download():
    archive = _SingleArchive()
    with pytest.raises(ValueError):
        archive.fetch_one("s1", client=_FakeClient())
    with pytest.raises(ValueError):
        archive.fetch_one(
            "s1", client=_FakeClient(), dest="a.fits", store_dir="out",
        )


def test_fetch_one_resolve_only_does_not_persist_payload(tmp_path):
    client = _FakeClient()
    dest = tmp_path / "not-written.fits"

    result = _SingleArchive().fetch_one(
        "s1", client=client, dest=dest, download=False,
    )

    assert result.success
    assert result.data is None
    assert not dest.exists()
    assert client.calls == []
    assert result.meta["url"] == "https://example.test/s1.fits"
    assert result.meta["dest"] == str(dest)
    assert result.meta["download"] is False


def test_dest_fn_old_and_new_signatures(tmp_path):
    old_hook = lambda obj_id, default: default.parent / "old" / default.name
    # Keep the historical positional order up to dest_fn.
    old = FetchContext(
        tmp_path, object(), "obj_id", "raj2000", "dej2000", 3.0, {}, old_hook,
    )
    assert old.dest("s1", tmp_path / "s1.fits") == tmp_path / "old" / "s1.fits"

    seen = []

    def with_row(obj_id, default, *, row=None):
        seen.append(row)
        return default.parent / str(row["group"]) / default.name

    new = FetchContext(store_dir=tmp_path, client=object(), dest_fn=with_row)
    row = pd.Series({"obj_id": "s1", "group": "g1"})
    assert new.dest("s1", tmp_path / "s1.fits", row=row) == tmp_path / "g1" / "s1.fits"
    assert seen[-1] is row

    archive = _SingleArchive()
    assert archive.resolve_output_path(new, "s1", row=row) == tmp_path / "g1" / "s1.fits"


def test_manga_fetch_one_uses_product_url_and_download_options(tmp_path):
    archive = SDSSArchive(
        mode="manga",
        manga_product="LOGCUBE",
        download_kwargs={"trust_existing": True},
    )
    client = _FakeClient()
    dest = tmp_path / "custom" / "manga-8138-12704-LOGCUBE.fits.gz"

    result = archive.fetch_one(
        "8138-12704", client=client, dest=dest,
        validate="gzip", resume="required",
    )

    assert result.success and result.data == dest
    assert dest.exists()
    assert "manga-8138-12704-LOGCUBE.fits.gz" in result.meta["url"]
    assert result.meta["dest"] == str(dest)
    assert result.meta["product"] == "LOGCUBE"
    assert result.meta["version"] == "v3_1_1"
    kwargs = client.calls[0]["kwargs"]
    assert kwargs["trust_existing"] is True
    assert kwargs["validate"] == "gzip"
    assert kwargs["resume"] == "required"
    assert "overwrite" not in kwargs


def test_manga_fetch_one_404_is_success_with_no_data(tmp_path):
    archive = SDSSArchive(mode="manga")
    client = _FakeClient(HttpError("missing", status=404))

    result = archive.fetch_one(
        "8138-12704", client=client, store_dir=tmp_path,
    )

    assert result.success
    assert result.data is None
    assert result.meta["url"].endswith("LOGRSS.fits.gz")


def test_item_result_meta_defaults_to_empty_dict():
    result = ItemResult(obj_id="s1", success=True)
    assert result.meta == {}
