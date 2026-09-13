"""Offline tests for WISE IPAC parsing / combination."""
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from astropy.io import ascii
from astropy.table import Table

from easycat.download.survey.wise import (
    WISEArchive,
    _build_ipac_table,
    combine_wisedata,
    _rows_retrieved,
)
from easycat.download.base import FetchContext

DATA = Path(__file__).parent.parent / "data" / "gator_neowise_upload.ipac"


def test_rows_retrieved():
    text = DATA.read_text()
    assert _rows_retrieved(text) == 3


def test_parse_gator_ipac():
    tbl = ascii.read(DATA.read_text(), format="ipac")
    assert len(tbl) == 3
    assert "cntr_01" in tbl.colnames
    assert set(tbl["cntr_01"].tolist()) == {1}


def test_slice_by_row_drops_bookkeeping():
    tbl = ascii.read(DATA.read_text(), format="ipac")
    slices = WISEArchive._slice_by_row(tbl)
    assert 1 in slices
    sub = slices[1]
    for col in ("cntr_01", "dist_x", "pang_x", "ra_01", "dec_01"):
        assert col not in sub.colnames


def test_build_ipac_table():
    import pandas as pd

    rows = pd.DataFrame({"raj2000": [1.0, 2.0], "dej2000": [3.0, 4.0]})
    text = _build_ipac_table(rows, "raj2000", "dej2000")
    assert "EQUINOX" in text
    assert "double" in text
    assert "1.0000000" in text


def test_build_ipac_table_aligns_wide_values():
    """Regression: values wider than a fixed 10-char column (e.g. a
    two-digit negative Dec or a three-digit RA) must widen the column
    instead of overflowing it -- Gator parses the upload as fixed width
    and otherwise rejects the whole table with a misleading
    'Table format is not right.' error.
    """
    import pandas as pd

    rows = pd.DataFrame({
        "raj2000": [0.0019783, 359.9999999, 100.5],
        "dej2000": [-0.4510883, -45.6789012, 19.4],
    })
    text = _build_ipac_table(rows, "raj2000", "dej2000")
    lines = text.splitlines()

    header, type_row = lines[1], lines[2]
    data = lines[3:]
    assert len(data) == 3

    # field boundaries (the '|' positions) define the declared widths
    bounds = [i for i, ch in enumerate(header) if ch == "|"]
    assert len(bounds) == 3
    ra_w = bounds[1] - bounds[0] - 1
    dec_w = bounds[2] - bounds[1] - 1

    # the widest values must fit in the declared columns
    assert max(len(s.split()[0]) for s in data) <= ra_w
    assert max(len(s.split()[1]) for s in data) <= dec_w
    assert ra_w >= len("359.9999999")      # three-digit RA
    assert dec_w >= len("-45.6789012")     # two-digit negative Dec

    # data rows are whitespace separated (no '|'), as Gator expects
    assert all("|" not in line for line in data)
    # type row uses the same column layout as the header
    assert [i for i, ch in enumerate(type_row) if ch == "|"] == bounds


def test_combine_wisedata_columns():
    # minimal NEOWISE table in the Gator schema
    neo = Table({
        "ra": [1.0, 1.0], "dec": [2.0, 2.0], "mjd": [58000.0, 58001.0],
        "w1mpro": [12.0, 12.1], "w1sigmpro": [0.1, 0.1], "w1rchi2": [1.0, 1.0],
        "w2mpro": [11.0, 11.1], "w2sigmpro": [0.1, 0.1], "w2rchi2": [1.0, 1.0],
        "na": [0, 0], "nb": [1, 1], "qi_fact": [1, 1],
        "cc_flags": ["0000", "0000"], "qual_frame": [5, 5],
        "saa_sep": [20.0, 20.0], "moon_masked": ["00", "00"],
    })
    allw = Table({
        "ra": [1.0], "dec": [2.0], "mjd": [55000.0],
        "w1mpro_ep": [12.5], "w1sigmpro_ep": [0.2], "w1rchi2_ep": [1.0],
        "w2mpro_ep": [11.5], "w2sigmpro_ep": [0.2], "w2rchi2_ep": [1.0],
        "na": [0], "nb": [0], "qi_fact": [1], "cc_flags": ["0000"],
        "saa_sep": [30.0], "moon_masked": ["00"],
    })
    combined = combine_wisedata(neo, allw)
    assert len(combined) == 3
    for col in ("raj2000", "dej2000", "mjd", "w1mag", "w1sigmag", "w1rchi2",
                "w2mag", "w2sigmag", "w2rchi2", "na", "nb", "qi_fact",
                "cc_flags", "qual_frame", "saa_sep", "moon_masked"):
        assert col in combined.colnames
    # AllWISE rows get qual_frame == -1
    assert set(np.asarray(combined["qual_frame"], int)) == {-1, 5}


def test_combine_wisedata_allwise_only():
    allw = Table({
        "ra": [1.0], "dec": [2.0], "mjd": [55000.0],
        "w1mpro_ep": [12.5], "w1sigmpro_ep": [0.2], "w1rchi2_ep": [1.0],
        "w2mpro_ep": [11.5], "w2sigmpro_ep": [0.2], "w2rchi2_ep": [1.0],
        "na": [0], "nb": [0], "qi_fact": [1], "cc_flags": ["0000"],
        "saa_sep": [30.0], "moon_masked": ["00"],
    })
    combined = combine_wisedata(None, allw)
    assert len(combined) == 1
    assert combined["qual_frame"][0] == -1


class _FakeResponse:
    status_code = 200

    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeClient:
    def __init__(self, text):
        self.text = text

    def post(self, *args, **kwargs):
        return _FakeResponse(self.text)


def test_allsky_search_keeps_quality_flags(tmp_path):
    response_table = Table({
        "cntr_01": [1],
        "dist_x": [0.12],
        "designation": ["J123456.78+123456.7"],
        "cc_flags": ["0000"],
        "ext_flg": [0],
        "var_flg": ["0000"],
    })
    buf = StringIO()
    ascii.write(response_table, buf, format="ipac")
    text = "\\ RowsRetrieved = 1\n" + buf.getvalue()

    rows = pd.DataFrame({
        "obj_id": ["s0"],
        "raj2000": [185.0],
        "dej2000": [15.0],
    })
    archive = WISEArchive(mode="allsky")
    results = archive.fetch_batch(
        rows,
        FetchContext(store_dir=tmp_path, client=_FakeClient(text)),
    )

    assert results[0].success
    assert results[0].data["cc_flags"][0] == "0000"
    assert results[0].data["ext_flg"][0] == 0
    assert results[0].data["var_flg"][0] == "0000"
    assert results[0].data["sep_arcsec"][0] == 0.12
    assert (tmp_path / "s0.fits").exists()
