"""High-level two-dimensional catalogue analysis API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from .binning import (
    GridResult,
    adaptive_grid,
    quantile_grid,
    regular_grid,
)
from .selection import PolygonSelector, SelectionResult
from .statistics import DEFAULT_STATS


_UNSET = object()


@dataclass
class ScatterPlot:
    """Return value for :meth:`Cat2D.scatter`."""

    fig: Any
    ax: Any
    artist: Any
    colorbar: Any = None


@dataclass
class MarginalPlot:
    """Return value for :meth:`Cat2D.plot_with_marginals`."""

    fig: Any
    ax: Any
    ax_top: Any
    ax_right: Any
    artist: Any


@dataclass
class HexBinPlot:
    """Return value for :meth:`Cat2D.hexbin`."""

    fig: Any
    ax: Any
    artist: Any
    colorbar: Any = None


class Cat2D:
    """Select and analyse two columns of a catalogue.

    Parameters
    ----------
    data : pandas.DataFrame
        Source catalogue.
    x, y : str
        Column names used for the horizontal and vertical coordinates.
    value : str or None
        Optional third column used for colour encoding and binned statistics.
    errors, weights : str or None
        Optional columns used by error-weighted statistics.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        x: str,
        y: str,
        value: Optional[str] = None,
        *,
        errors: Optional[str] = None,
        weights: Optional[str] = None,
    ):
        self.data = data.reset_index(drop=True).copy()
        self.x = str(x)
        self.y = str(y)
        self.value = None if value is None else str(value)
        self.errors = None if errors is None else str(errors)
        self.weights = None if weights is None else str(weights)
        self._validate_columns()

    def _validate_columns(self) -> None:
        required = [self.x, self.y]
        optional = [self.value, self.errors, self.weights]
        missing = [name for name in required + optional if name and name not in self.data.columns]
        if missing:
            raise KeyError(f"columns not found: {missing}")

    @property
    def x_values(self) -> np.ndarray:
        return pd.to_numeric(self.data[self.x], errors="coerce").to_numpy(float)

    @property
    def y_values(self) -> np.ndarray:
        return pd.to_numeric(self.data[self.y], errors="coerce").to_numpy(float)

    @property
    def value_values(self) -> Optional[np.ndarray]:
        if self.value is None:
            return None
        return pd.to_numeric(self.data[self.value], errors="coerce").to_numpy(float)

    def subset(self, mask: Iterable[bool] | Iterable[int] | SelectionResult) -> "Cat2D":
        """Return a new Cat2D restricted by a boolean mask, indices or selection."""
        if isinstance(mask, SelectionResult):
            bool_mask = mask.mask
        else:
            values = np.asarray(mask)
            if values.dtype == bool:
                if values.size != len(self.data):
                    raise ValueError("boolean mask length does not match data")
                bool_mask = values
            else:
                bool_mask = np.zeros(len(self.data), dtype=bool)
                bool_mask[values.astype(int)] = True
        return Cat2D(
            self.data.loc[bool_mask].copy(),
            self.x,
            self.y,
            self.value,
            errors=self.errors,
            weights=self.weights,
        )

    def scatter(
        self,
        *,
        ax=None,
        color_by: object = _UNSET,
        cmap: str = "viridis",
        colorbar: bool = True,
        **kwargs,
    ) -> ScatterPlot | Any:
        """Plot a two-dimensional scatter and optionally encode a value column."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        if color_by is _UNSET:
            color_column = self.value
        elif color_by is None or color_by is False:
            color_column = None
        else:
            color_column = str(color_by)
        if color_column is not None and color_column not in self.data.columns:
            raise KeyError(f"colour column {color_column!r} not found")
        color_values = (
            None if color_column is None
            else pd.to_numeric(self.data[color_column], errors="coerce").to_numpy(float)
        )
        kwargs.setdefault("s", 12)
        kwargs.setdefault("alpha", 0.7)
        if color_values is not None:
            kwargs["c"] = color_values
            kwargs["cmap"] = cmap
        artist = ax.scatter(
            self.x_values,
            self.y_values,
            **kwargs,
        )
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        bar = None
        if color_values is not None and colorbar:
            bar = fig.colorbar(artist, ax=ax, label=color_column)
        return ScatterPlot(fig=fig, ax=ax, artist=artist, colorbar=bar)

    def plot(self, *, ax=None, **kwargs):
        """Alias for :meth:`scatter`, returning the matplotlib axes."""
        result = self.scatter(ax=ax, **kwargs)
        return result.ax if isinstance(result, ScatterPlot) else result

    def hexbin(
        self,
        *,
        ax=None,
        gridsize: int = 50,
        color_by: object = _UNSET,
        reduce_C_function=None,
        mincnt: Optional[int] = None,
        cmap: str = "viridis",
        colorbar: bool = True,
        **kwargs,
    ) -> HexBinPlot:
        """Plot hexagonally binned counts or a reduced value column."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        if color_by is _UNSET:
            color_column = self.value
        elif color_by is None or color_by is False:
            color_column = None
        else:
            color_column = str(color_by)
        color_values = None
        if color_column is not None:
            if color_column not in self.data.columns:
                raise KeyError(f"colour column {color_column!r} not found")
            color_values = pd.to_numeric(
                self.data[color_column], errors="coerce",
            ).to_numpy(float)

        valid = np.isfinite(self.x_values) & np.isfinite(self.y_values)
        if color_values is not None:
            valid &= np.isfinite(color_values)
        x = self.x_values[valid]
        y = self.y_values[valid]
        c = None if color_values is None else color_values[valid]

        hex_kwargs = dict(kwargs)
        hex_kwargs.update({"gridsize": gridsize, "mincnt": mincnt, "cmap": cmap})
        if c is not None and reduce_C_function is not None:
            hex_kwargs["reduce_C_function"] = reduce_C_function
        artist = ax.hexbin(x, y, C=c, **hex_kwargs)
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        bar = None
        if colorbar:
            label = "count" if c is None else color_column
            bar = fig.colorbar(artist, ax=ax, label=label)
        return HexBinPlot(fig=fig, ax=ax, artist=artist, colorbar=bar)

    def inspect_points(
        self,
        *,
        ax=None,
        on_click=None,
        default: str = "print",
        columns=None,
        pick_radius: float = 8.0,
        highlight: bool = True,
    ):
        """Create a click-to-inspect selector for points on this dataset."""
        from .inspection import PointInspector

        if ax is None:
            ax = self.plot()
        return PointInspector(
            self, ax=ax, on_click=on_click, default=default,
            columns=columns, pick_radius=pick_radius, highlight=highlight,
        )

    def select_polygon(self, *, ax=None, marker_size: float = 4.0) -> PolygonSelector:
        """Attach an interactive polygon selector to a scatter plot."""
        selector = PolygonSelector(self, ax=ax, marker_size=marker_size)
        selector.ax.set_title("Left click: add | right/backspace: undo | Enter: finish")
        return selector

    def contour(
        self,
        *,
        ax=None,
        bins: int = 50,
        levels: int | Sequence[float] = 10,
        weighted: bool = False,
        filled: bool = False,
        cmap: str = "viridis",
        **kwargs,
    ):
        """Draw a density or value-weighted contour map."""
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        weights = self.value_values if weighted and self.value is not None else None
        hist, x_edges, y_edges = np.histogram2d(
            self.x_values[np.isfinite(self.x_values) & np.isfinite(self.y_values)],
            self.y_values[np.isfinite(self.x_values) & np.isfinite(self.y_values)],
            bins=bins,
            weights=None if weights is None else weights[np.isfinite(self.x_values) & np.isfinite(self.y_values)],
        )
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
        contour = (
            ax.contourf(x_centers, y_centers, hist.T, levels=levels, cmap=cmap, **kwargs)
            if filled else
            ax.contour(x_centers, y_centers, hist.T, levels=levels, cmap=cmap, **kwargs)
        )
        ax.set_xlabel(self.x)
        ax.set_ylabel(self.y)
        fig.colorbar(contour, ax=ax, label="weighted density" if weighted else "density")
        return ax, contour

    def plot_with_marginals(
        self,
        *,
        bins: int = 50,
        figsize: tuple[float, float] = (7, 7),
        scatter_kwargs: Optional[dict] = None,
    ) -> MarginalPlot:
        """Plot the central scatter with marginal histograms."""
        fig = plt.figure(figsize=figsize)
        grid = GridSpec(4, 4, figure=fig, hspace=0.05, wspace=0.05)
        ax = fig.add_subplot(grid[1:, :-1])
        ax_top = fig.add_subplot(grid[0, :-1], sharex=ax)
        ax_right = fig.add_subplot(grid[1:, -1], sharey=ax)

        artist = self.scatter(ax=ax, colorbar=True, **(scatter_kwargs or {}))
        valid = np.isfinite(self.x_values) & np.isfinite(self.y_values)
        ax_top.hist(self.x_values[valid], bins=bins, color="0.35")
        ax_right.hist(self.y_values[valid], bins=bins, orientation="horizontal", color="0.35")
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)
        return MarginalPlot(
            fig=fig,
            ax=ax,
            ax_top=ax_top,
            ax_right=ax_right,
            artist=artist.artist if isinstance(artist, ScatterPlot) else artist,
        )

    def regular_grid(self, **kwargs) -> GridResult:
        """Bin using uniform Cartesian edges."""
        return regular_grid(
            self.data,
            x=self.x,
            y=self.y,
            value=self.value,
            errors=self.errors,
            weights=self.weights,
            **kwargs,
        )

    def quantile_grid(self, **kwargs) -> GridResult:
        """Bin using quantile-derived edges."""
        return quantile_grid(
            self.data,
            x=self.x,
            y=self.y,
            value=self.value,
            errors=self.errors,
            weights=self.weights,
            **kwargs,
        )

    def adaptive_grid(self, **kwargs) -> GridResult:
        """Bin using recursive median splitting."""
        return adaptive_grid(
            self.data,
            x=self.x,
            y=self.y,
            value=self.value,
            errors=self.errors,
            weights=self.weights,
            **kwargs,
        )


__all__ = ["Cat2D", "HexBinPlot", "MarginalPlot", "ScatterPlot"]
