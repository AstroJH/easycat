"""Regular, quantile and adaptive two-dimensional binning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from .statistics import DEFAULT_STATS, select_stats, summarize


@dataclass
class GridResult:
    """Statistics evaluated on rectangular two-dimensional cells."""

    table: pd.DataFrame
    rectangles: list[tuple[float, float, float, float]]

    @property
    def selected(self) -> pd.DataFrame:
        """Rows satisfying the configured minimum sample count."""
        return self.table[self.table["selected"]].reset_index(drop=True)

    def plot(
        self,
        *,
        color: str = "median",
        ax=None,
        cmap: str = "viridis",
        show_counts: bool = False,
        min_count: Optional[int] = None,
    ):
        """Draw selected cells colored by one statistic."""
        if ax is None:
            _, ax = plt.subplots()
        table = self.table
        if min_count is not None:
            table = table[table["count"] >= min_count]
        else:
            table = table[table["selected"]]
        if color not in table.columns:
            raise KeyError(f"unknown statistic column {color!r}")

        values = pd.to_numeric(table[color], errors="coerce").to_numpy(float)
        finite = np.isfinite(values)
        norm = mcolors.Normalize(
            vmin=np.nanmin(values) if finite.any() else 0.0,
            vmax=np.nanmax(values) if finite.any() else 1.0,
        )
        cmap_obj = plt.get_cmap(cmap)
        for row, value in zip(table.itertuples(index=False), values):
            face = "none" if not np.isfinite(value) else cmap_obj(norm(value))
            ax.add_patch(Rectangle(
                (row.x_lo, row.y_lo), row.x_hi - row.x_lo,
                row.y_hi - row.y_lo, facecolor=face,
                edgecolor="black", linewidth=0.5,
            ))
            if show_counts:
                ax.text(
                    row.x_center, row.y_center, str(int(row.count)),
                    ha="center", va="center", fontsize=7,
                )
        ax.set_xlim(table["x_lo"].min(), table["x_hi"].max())
        ax.set_ylim(table["y_lo"].min(), table["y_hi"].max())
        ax.figure.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap_obj),
            ax=ax, label=color,
        )
        return ax


def _finite_bounds(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot bin an empty or all-NaN dataset")
    return float(np.min(finite)), float(np.max(finite))


def _regular_edges(values: np.ndarray, bins: int) -> np.ndarray:
    lo, hi = _finite_bounds(values)
    if lo == hi:
        width = 0.5
        return np.asarray([lo - width, hi + width])
    return np.linspace(lo, hi, int(bins) + 1)


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot bin an empty or all-NaN dataset")
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, int(bins) + 1)))
    if edges.size < 2:
        return _regular_edges(finite, 1)
    return edges


def _rectangles_from_edges(
    x_edges: Sequence[float],
    y_edges: Sequence[float],
) -> list[tuple[float, float, float, float]]:
    return [
        (float(x0), float(y0), float(x1), float(y1))
        for x0, x1 in zip(x_edges[:-1], x_edges[1:])
        for y0, y1 in zip(y_edges[:-1], y_edges[1:])
    ]


def _rect_mask_parts(
    x: np.ndarray,
    y: np.ndarray,
    rectangle: tuple[float, float, float, float],
    *,
    x_max: float,
    y_max: float,
) -> np.ndarray:
    x0, y0, x1, y1 = rectangle
    x_right = x <= x1 if np.isclose(x1, x_max) else x < x1
    y_top = y <= y1 if np.isclose(y1, y_max) else y < y1
    return (x >= x0) & x_right & (y >= y0) & y_top


def summarize_rectangles(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    rectangles: Sequence[tuple[float, float, float, float]],
    value: Optional[str] = None,
    errors: Optional[str] = None,
    weights: Optional[str] = None,
    stats: Iterable[str] = DEFAULT_STATS,
    min_count: int = 1,
) -> GridResult:
    """Evaluate statistics on arbitrary non-overlapping rectangles."""
    if value is not None and value not in data.columns:
        raise KeyError(f"value column {value!r} not found")
    if errors is not None and errors not in data.columns:
        raise KeyError(f"error column {errors!r} not found")
    if weights is not None and weights not in data.columns:
        raise KeyError(f"weight column {weights!r} not found")

    x_arr = pd.to_numeric(data[x], errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(data[y], errors="coerce").to_numpy(float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not finite.any():
        raise ValueError("no finite x/y pairs available for binning")
    x_max = float(np.nanmax(x_arr[finite]))
    y_max = float(np.nanmax(y_arr[finite]))

    rows = []
    for x0, y0, x1, y1 in rectangles:
        mask = finite & _rect_mask_parts(
            x_arr, y_arr, (x0, y0, x1, y1),
            x_max=x_max, y_max=y_max,
        )
        count = int(mask.sum())
        row: dict[str, float | int | bool] = {
            "x_lo": x0, "x_hi": x1, "y_lo": y0, "y_hi": y1,
            "x_center": (x0 + x1) / 2.0, "y_center": (y0 + y1) / 2.0,
            "count": count, "selected": count >= min_count,
        }
        if value is not None and count:
            summary = summarize(
                data.loc[mask, value],
                errors=None if errors is None else data.loc[mask, errors],
                weights=None if weights is None else data.loc[mask, weights],
            )
            row.update(select_stats(summary, stats))
        else:
            for name in stats:
                row[name] = np.nan if name != "count" else count
        rows.append(row)
    return GridResult(pd.DataFrame(rows), list(rectangles))


def regular_grid(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    xbins: int = 20,
    ybins: int = 20,
    x_edges: Optional[Sequence[float]] = None,
    y_edges: Optional[Sequence[float]] = None,
    value: Optional[str] = None,
    errors: Optional[str] = None,
    weights: Optional[str] = None,
    stats: Iterable[str] = DEFAULT_STATS,
    min_count: int = 1,
) -> GridResult:
    """Bin on a uniform Cartesian grid."""
    x_arr = pd.to_numeric(data[x], errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(data[y], errors="coerce").to_numpy(float)
    x_edges = _regular_edges(x_arr, xbins) if x_edges is None else np.asarray(x_edges, float)
    y_edges = _regular_edges(y_arr, ybins) if y_edges is None else np.asarray(y_edges, float)
    rectangles = _rectangles_from_edges(x_edges, y_edges)
    return summarize_rectangles(
        data, x=x, y=y, rectangles=rectangles, value=value,
        errors=errors, weights=weights, stats=stats, min_count=min_count,
    )


def quantile_grid(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    xbins: int = 20,
    ybins: int = 20,
    value: Optional[str] = None,
    errors: Optional[str] = None,
    weights: Optional[str] = None,
    stats: Iterable[str] = DEFAULT_STATS,
    min_count: int = 1,
) -> GridResult:
    """Bin on data-driven quantile edges for approximately equal counts."""
    x_arr = pd.to_numeric(data[x], errors="coerce").to_numpy(float)
    y_arr = pd.to_numeric(data[y], errors="coerce").to_numpy(float)
    rectangles = _rectangles_from_edges(
        _quantile_edges(x_arr, xbins),
        _quantile_edges(y_arr, ybins),
    )
    return summarize_rectangles(
        data, x=x, y=y, rectangles=rectangles, value=value,
        errors=errors, weights=weights, stats=stats, min_count=min_count,
    )


def adaptive_grid(
    data: pd.DataFrame,
    *,
    x: str,
    y: str,
    max_count: int = 100,
    min_count: int = 1,
    max_depth: int = 8,
    min_span: float = 1e-12,
    value: Optional[str] = None,
    errors: Optional[str] = None,
    weights: Optional[str] = None,
    stats: Iterable[str] = DEFAULT_STATS,
) -> GridResult:
    """Recursively split data-rich cells along their longest axis.

    Splitting uses the median coordinate so each child retains roughly half
    of the points.  Recursion stops at ``max_depth``, when a cell has at most
    ``max_count`` points, or when it cannot be split further.
    """
    x_all = pd.to_numeric(data[x], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(data[y], errors="coerce").to_numpy(float)
    finite = np.isfinite(x_all) & np.isfinite(y_all)
    if not finite.any():
        raise ValueError("no finite x/y pairs available for binning")
    x_lo, x_hi = _finite_bounds(x_all[finite])
    y_lo, y_hi = _finite_bounds(y_all[finite])
    rectangles: list[tuple[float, float, float, float]] = []

    def split(rect, indices, depth):
        x0, y0, x1, y1 = rect
        if len(indices) <= max_count or depth >= max_depth:
            rectangles.append(rect)
            return
        xspan = x1 - x0
        yspan = y1 - y0
        if xspan >= yspan and xspan > min_span:
            split_value = float(np.median(x_all[indices]))
            left = indices[x_all[indices] < split_value]
            right = indices[x_all[indices] >= split_value]
            if left.size and right.size:
                split((x0, y0, split_value, y1), left, depth + 1)
                split((split_value, y0, x1, y1), right, depth + 1)
                return
        if yspan > min_span:
            split_value = float(np.median(y_all[indices]))
            bottom = indices[y_all[indices] < split_value]
            top = indices[y_all[indices] >= split_value]
            if bottom.size and top.size:
                split((x0, y0, x1, split_value), bottom, depth + 1)
                split((x0, split_value, x1, y1), top, depth + 1)
                return
        rectangles.append(rect)

    split((x_lo, y_lo, x_hi, y_hi), np.flatnonzero(finite), 0)
    return summarize_rectangles(
        data, x=x, y=y, rectangles=rectangles, value=value,
        errors=errors, weights=weights, stats=stats, min_count=min_count,
    )


__all__ = [
    "GridResult",
    "adaptive_grid",
    "quantile_grid",
    "regular_grid",
    "summarize_rectangles",
]
