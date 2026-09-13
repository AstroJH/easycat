"""Click-to-inspect behavior for scatter plots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class PointInfo:
    """Information for the catalogue point nearest to a click."""

    sample: str
    index: int
    row: pd.Series
    source: object


class PointInspector:
    """Inspect the nearest plotted point on a left click.

    Parameters
    ----------
    source : Cat2D or Cat2DCollection
        Plotted catalogue object.
    ax : matplotlib.axes.Axes
        Axes containing the points.
    on_click : callable or None
        Optional callback receiving :class:`PointInfo`.  When supplied it
        replaces the default action.
    default : {"print", "none"}
        Default behavior when no callback is supplied.
    columns : sequence or None
        Restrict the printed row to these columns.
    pick_radius : float
        Maximum click distance in display pixels.
    highlight : bool
        Mark the selected point with a red ring.
    """

    def __init__(
        self,
        source,
        *,
        ax,
        on_click: Optional[Callable[[PointInfo], None]] = None,
        default: str = "print",
        columns: Optional[Sequence[str]] = None,
        pick_radius: float = 8.0,
        highlight: bool = True,
    ):
        if default not in ("print", "none"):
            raise ValueError("default must be 'print' or 'none'")
        self.source = source
        self.ax = ax
        self.on_click = on_click
        self.default = default
        self.columns = None if columns is None else list(columns)
        self.pick_radius = float(pick_radius)
        self.highlight = bool(highlight)
        self.last: Optional[PointInfo] = None
        self._highlight_artist = None
        self._canvas = ax.figure.canvas
        self._cid = self._canvas.mpl_connect(
            "button_press_event", self._on_click,
        )
        self._samples = self._normalise_sources(source)

    @staticmethod
    def _normalise_sources(source) -> Mapping[str, object]:
        if hasattr(source, "samples"):
            return source.samples
        return {"": source}

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button != 1:
            return

        self.inspect_xy(float(event.xdata), float(event.ydata))

    def _nearest(self, x: float, y: float) -> tuple[Optional[PointInfo], float]:
        """Return the nearest catalogue row and its display-pixel distance."""
        click = self.ax.transData.transform((x, y))
        best = None
        best_distance = np.inf
        for name, cat in self._samples.items():
            finite = np.isfinite(cat.x_values) & np.isfinite(cat.y_values)
            indices = np.flatnonzero(finite)
            if indices.size == 0:
                continue
            points = self.ax.transData.transform(np.column_stack([
                cat.x_values[indices], cat.y_values[indices],
            ]))
            distances = np.hypot(points[:, 0] - click[0], points[:, 1] - click[1])
            local = int(np.argmin(distances))
            if distances[local] < best_distance:
                best_distance = float(distances[local])
                best = (name, int(indices[local]), cat)

        if best is None or best_distance > self.pick_radius:
            return None, best_distance
        name, index, cat = best
        return PointInfo(
            sample=name,
            index=index,
            row=cat.data.iloc[index].copy(),
            source=cat,
        ), best_distance

    def inspect_xy(self, x: float, y: float, *, dispatch: bool = True):
        """Inspect the nearest point to data coordinates ``(x, y)``.

        This is public so callers and tests can trigger the same behavior as
        a mouse click without synthesizing a Matplotlib event.
        """
        info, distance = self._nearest(float(x), float(y))
        if info is None:
            return None
        self.last = info
        if self.highlight:
            self._draw_highlight(info.source, info.index)
        if dispatch:
            if self.on_click is not None:
                self.on_click(info)
            elif self.default == "print":
                self._print_info(info)
        return info

    def _draw_highlight(self, cat, index: int) -> None:
        if self._highlight_artist is not None:
            self._highlight_artist.remove()
        self._highlight_artist = self.ax.scatter(
            [cat.x_values[index]], [cat.y_values[index]],
            s=130, facecolors="none", edgecolors="red", linewidths=1.8,
            zorder=10,
        )
        self._canvas.draw_idle()

    def _print_info(self, info: PointInfo) -> None:
        label = f"[{info.sample}] " if info.sample else ""
        print(f"{label}row={info.index}")
        row = info.row
        if self.columns is not None:
            row = row.reindex(self.columns)
        print(row.to_string())

    def disconnect(self) -> None:
        """Disconnect the click callback."""
        self._canvas.mpl_disconnect(self._cid)


__all__ = ["PointInfo", "PointInspector"]
