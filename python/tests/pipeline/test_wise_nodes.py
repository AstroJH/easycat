"""Tests for WISE pipeline normalization and atomic storage."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from easycat.pipeline import DataPacket
from easycat.pipeline.survey.wise import (
    WiseBasicCriteriaNode,
    WiseLcStorage,
    WisePreprocessNode,
    WiseAggregator,
)


def _wise_frame():
    return pd.DataFrame({
        "mjd": [1.0, 2.0, 3.0],
        "w1mag": [12.0, 12.1, np.nan],
        "w2mag": [11.8, 11.9, 12.0],
        "w1sigmag": [0.1, 0.1, 0.1],
        "w2sigmag": [0.1, 0.1, 0.1],
        "na": [0, 0, 0],
        "nb": [0, 1, 0],
        "saa_sep": [10.0, 10.0, 10.0],
        "qi_fact": [1, 1, 1],
        "qual_frame": [1, 1, 1],
        "w1rchi2": [1.0, 1.0, 1.0],
        "w2rchi2": [1.0, 1.0, 1.0],
        "moon_masked": [b"00", b"00", b"00"],
        "cc_flags": [b"0000", b"0000", b"0000"],
    })


def test_wise_preprocess_handles_nan_and_bytes():
    packet = DataPacket(light_curve=_wise_frame())
    packet = WisePreprocessNode().execute(packet)
    assert len(packet.light_curve) == 2
    assert packet.light_curve["cc_flags"].iloc[0] == "0000"
    assert packet.get_result_value("removed_invalid", "preprocess@wise") == 1


def test_wise_basic_criteria_decodes_flags_and_records_cutflow():
    packet = DataPacket(light_curve=_wise_frame().iloc[:2].copy())
    packet = WiseBasicCriteriaNode().execute(packet)
    assert len(packet.light_curve) == 2
    assert packet.light_curve["cc_flags"].iloc[0] == "0000"
    assert packet.get_result_value("removed_cc", "basic_criteria@wise") == 0
    assert packet.get_result_value("output_rows", "basic_criteria@wise") == 2


def test_wise_storage_is_atomic_and_records_provenance(tmp_path):
    packet = DataPacket(
        obj_id="source-1",
        light_curve=_wise_frame().iloc[:2].copy(),
        metadata={"filepath": "input.fits"},
        provenance={"input_file": "/data/input.fits"},
    )
    output = tmp_path / "out" / "source-1.fits"
    packet.metadata["storage_path"] = output

    packet = WiseLcStorage().execute(packet)

    assert output.exists()
    with fits.open(output) as hdul:
        assert hdul[0].header["OBJID"] == "source-1"
        assert json.loads(hdul[0].header["PROV"])["input_file"].endswith("input.fits")
        assert hdul[1].name == "WISELC"


def test_wise_aggregator_uses_documented_error_formula():
    values = np.asarray([12.0, 12.2, 11.9, 12.1])
    errors = np.asarray([0.1, 0.15, 0.12, 0.11])
    mean_mag, uncertainty = WiseAggregator("W1").aggregate(values, errors)

    n = len(values)
    expected_variance = (
        np.sum((values - mean_mag) ** 2) / (n * (n - 1))
        + np.sum(errors**2) / n**2
        + 0.016**2 / n
    )
    assert uncertainty == pytest.approx(np.sqrt(expected_variance))


def test_wise_aggregator_single_measurement_uncertainty():
    _, sys_only = WiseAggregator("W2").aggregate([12.0])
    assert sys_only == pytest.approx(0.016)

    _, with_error = WiseAggregator("W2").aggregate([12.0], [0.1])
    assert with_error == pytest.approx(np.sqrt(0.1**2 + 0.016**2))
