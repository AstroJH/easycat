"""Tests for the cat2d selection, binning and statistics API."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import numpy as np
import pandas as pd
import pytest

from easycat.cat2d import (
    Cat2D,
    Cat2DCollection,
    adaptive_grid,
    polygon_area,
    polygon_mask,
    quantile_grid,
    regular_grid,
    summarize,
)


def make_data(n=200):
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "x": rng.uniform(0, 10, n),
        "y": rng.uniform(-2, 2, n),
        "value": rng.normal(5, 1, n),
        "error": rng.uniform(0.1, 0.4, n),
    })


def test_summarize_basic_and_weighted_statistics():
    stats = summarize([1.0, 2.0, 3.0], errors=[0.1, 0.2, 0.3])
    assert stats["count"] == 3
    assert stats["mean"] == 2.0
    assert stats["median"] == 2.0
    assert stats["std"] == pytest.approx(1.0)
    assert stats["error_weighted_mean"] == pytest.approx(
        np.average([1, 2, 3], weights=1 / np.asarray([0.1, 0.2, 0.3]) ** 2)
    )


def test_polygon_selection():
    x = np.asarray([0.0, 1.0, 2.0, 3.0])
    y = np.asarray([0.0, 1.0, 2.0, 3.0])
    vertices = [(0.5, -0.5), (1.5, -0.5), (1.5, 1.5), (0.5, 1.5)]
    mask = polygon_mask(x, y, vertices)
    assert mask.tolist() == [False, True, False, False]
    assert polygon_area(vertices) == pytest.approx(2.0)


def test_regular_grid_counts_and_stats():
    data = pd.DataFrame({
        "x": [0.1, 0.2, 0.8, 0.9],
        "y": [0.1, 0.2, 0.8, 0.9],
        "v": [1.0, 3.0, 5.0, 7.0],
    })
    result = regular_grid(
        data, x="x", y="y", x_edges=[0, 0.5, 1], y_edges=[0, 0.5, 1],
        value="v", min_count=2,
    )
    table = result.table.set_index(["x_lo", "y_lo"])
    assert table.loc[(0.0, 0.0), "count"] == 2
    assert table.loc[(0.0, 0.0), "mean"] == 2.0
    assert table.loc[(0.0, 0.0), "selected"]
    assert not table.loc[(0.0, 0.5), "selected"]


def test_quantile_and_adaptive_grid_cover_points():
    data = make_data(300)
    quantile = quantile_grid(data, x="x", y="y", xbins=5, ybins=4)
    assert quantile.table["count"].sum() == len(data)

    adaptive = adaptive_grid(
        data, x="x", y="y", max_count=30, min_count=1, max_depth=5,
    )
    assert adaptive.table["count"].sum() == len(data)
    assert len(adaptive.rectangles) > 1


def test_cat2d_plot_contour_and_marginals():
    cat = Cat2D(make_data(), "x", "y", "value")
    scatter = cat.scatter()
    assert scatter.artist is not None
    contour_ax, contour = cat.contour(bins=20, levels=5)
    assert contour_ax is not None and contour is not None
    marginals = cat.plot_with_marginals(bins=20)
    assert marginals.ax is not None and marginals.artist is not None
    selected = cat.subset(np.arange(10))
    assert len(selected.data) == 10


def test_hexbin_plot_and_value_reduction():
    cat = Cat2D(make_data(), "x", "y", "value")
    count_plot = cat.hexbin(gridsize=12)
    assert count_plot.artist is not None
    value_plot = cat.hexbin(
        gridsize=12, color_by="value", reduce_C_function=np.median,
    )
    assert value_plot.colorbar is not None


def test_collection_polygon_selection():
    first = make_data(80)
    second = make_data(120)
    collection = Cat2DCollection(
        {"S1": first, "S2": second}, x="x", y="y", value="value",
    )
    selector = collection.select_polygon()
    selector.vertices = [(0.0, -3.0), (10.0, -3.0), (10.0, 3.0), (0.0, 3.0)]
    result = selector.finish()

    assert set(result.to_dict()) == {"S1", "S2"}
    assert result["S1"].mask.sum() <= len(first)
    assert result["S2"].mask.sum() <= len(second)
    assert set(result.indices) == {"S1", "S2"}


def test_point_inspector_none_custom_and_last():
    data = make_data(20)
    cat = Cat2D(data, "x", "y", "value")
    plot = cat.scatter()
    inspector = cat.inspect_points(ax=plot.ax, default="none")

    info = inspector.inspect_xy(float(data.x.iloc[3]), float(data.y.iloc[3]))
    assert info is not None and info.index == 3
    assert inspector.last.index == 3
    inspector.disconnect()

    seen = []
    custom = cat.inspect_points(
        ax=cat.scatter().ax,
        on_click=lambda point: seen.append((point.sample, point.index)),
    )
    custom.inspect_xy(float(data.x.iloc[4]), float(data.y.iloc[4]))
    assert seen[-1][1] == 4
    custom.disconnect()
