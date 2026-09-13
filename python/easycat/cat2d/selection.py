"""Polygon selection and interactive vertex editing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, TYPE_CHECKING

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.path import Path

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from .dataset import Cat2D


def polygon_mask(
    x: Iterable[float],
    y: Iterable[float],
    vertices: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Return a boolean point-in-polygon mask using a closed matplotlib path."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size != y_arr.size:
        raise ValueError("x and y must have the same length")
    if len(vertices) < 3:
        return np.zeros(x_arr.size, dtype=bool)

    polygon = np.asarray(vertices, dtype=float)
    valid = np.isfinite(x_arr) & np.isfinite(y_arr)
    mask = np.zeros(x_arr.size, dtype=bool)
    if np.any(valid):
        points = np.column_stack([x_arr[valid], y_arr[valid]])
        # ``closed=True`` adds a CLOSEPOLY code that can make the point-in-
        # polygon test depend on the created path's sentinel vertex.  A plain
        # Path is already treated as closed when the first/last vertices are
        # connected by matplotlib for containment tests.
        mask[valid] = Path(polygon).contains_points(points)
    return mask


def polygon_area(vertices: Sequence[tuple[float, float]]) -> float:
    """Return the absolute polygon area using the shoelace formula."""
    polygon = np.asarray(vertices, dtype=float)
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2.0)


@dataclass
class SelectionResult:
    """A geometric selection and the corresponding source rows."""

    source: "Cat2D"
    mask: np.ndarray
    vertices: np.ndarray
    area: float

    @property
    def indices(self) -> np.ndarray:
        return np.flatnonzero(self.mask)

    @property
    def data(self):
        return self.source.data.loc[self.mask].copy()

    def to_cat2d(self) -> "Cat2D":
        return self.source.subset(self.mask)


@dataclass
class MultiSelectionResult:
    """One polygon applied independently to multiple named samples."""

    source: object
    masks: dict[str, np.ndarray]
    vertices: np.ndarray
    area: float

    def __getitem__(self, name: str) -> "SelectionResult":
        return SelectionResult(
            source=self.source[name],
            mask=self.masks[name],
            vertices=self.vertices,
            area=self.area,
        )

    @property
    def indices(self) -> dict[str, np.ndarray]:
        return {name: np.flatnonzero(mask) for name, mask in self.masks.items()}

    @property
    def data(self) -> dict[str, object]:
        return {name: self[name].data for name in self.masks}

    def to_dict(self) -> dict[str, "Cat2D"]:
        return {name: self[name].to_cat2d() for name in self.masks}


class PolygonSelector:
    """Matplotlib polygon editor.

    Controls
    --------
    left click
        Add a vertex.
    right click / Backspace
        Remove the latest vertex.
    Enter / 'f'
        Close and apply the polygon.
    Escape
        Reset the current polygon.

    The constructor is non-blocking, making it usable in Jupyter.  Call
    :meth:`finish` programmatically when the polygon is complete.
    """

    def __init__(
        self,
        source: "Cat2D",
        *,
        ax: Optional["Axes"] = None,
        marker_size: float = 4.0,
        line_width: float = 1.2,
    ):
        self.source = source
        self.ax = source.plot(ax=ax)
        self.marker_size = marker_size
        self.line_width = line_width
        self.vertices: list[tuple[float, float]] = []
        self.result: Optional[SelectionResult] = None
        self.finished = False
        self._preview: Optional[Line2D] = None
        self._points = None
        self._canvas = self.ax.figure.canvas
        self._connections = [
            self._canvas.mpl_connect("button_press_event", self._on_click),
            self._canvas.mpl_connect("motion_notify_event", self._on_move),
            self._canvas.mpl_connect("key_press_event", self._on_key),
        ]

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:
            self.vertices.append((float(event.xdata), float(event.ydata)))
            self._redraw_preview(event.xdata, event.ydata)
        elif event.button == 3:
            self.undo()

    def _on_move(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        if self.vertices:
            self._redraw_preview(event.xdata, event.ydata)

    def _on_key(self, event) -> None:
        if event.key in ("enter", "f"):
            self.finish()
        elif event.key in ("backspace", "delete"):
            self.undo()
        elif event.key == "escape":
            self.reset()

    def _clear_artists(self) -> None:
        for artist in (self._preview, self._points):
            if artist is not None:
                artist.remove()
        self._preview = None
        self._points = None

    def _redraw_preview(self, cursor_x: float, cursor_y: float) -> None:
        self._clear_artists()
        vertices = self.vertices
        if not vertices:
            self._canvas.draw_idle()
            return

        closed = vertices + [(cursor_x, cursor_y)]
        if len(closed) >= 3:
            closed = closed + [vertices[0]]
        x, y = zip(*closed)
        self._preview, = self.ax.plot(
            x, y, color="white", lw=self.line_width, linestyle="--", zorder=5,
        )
        vx, vy = zip(*vertices)
        self._points = self.ax.scatter(
            vx, vy, s=self.marker_size**2, color="white", edgecolor="black",
            zorder=6,
        )
        self._canvas.draw_idle()

    def undo(self) -> None:
        """Remove the most recently added vertex."""
        if self.vertices:
            self.vertices.pop()
        self._clear_artists()
        if self.vertices:
            x, y = self.vertices[-1]
            self._redraw_preview(x, y)
        else:
            self._canvas.draw_idle()

    def reset(self) -> None:
        """Clear the polygon and selection result."""
        self.vertices.clear()
        self.result = None
        self.finished = False
        self._clear_artists()
        self._canvas.draw_idle()

    def finish(self) -> SelectionResult:
        """Close the polygon, compute the mask and return the result."""
        if len(self.vertices) < 3:
            raise ValueError("at least three vertices are required")
        if hasattr(self.source, "samples"):
            masks = {
                name: polygon_mask(cat.x_values, cat.y_values, self.vertices)
                for name, cat in self.source.samples.items()
            }
            self.result = MultiSelectionResult(
                source=self.source,
                masks=masks,
                vertices=np.asarray(self.vertices, dtype=float),
                area=polygon_area(self.vertices),
            )
            self.finished = True
            self.close()
            return self.result
        mask = polygon_mask(
            self.source.x_values,
            self.source.y_values,
            self.vertices,
        )
        self.result = SelectionResult(
            source=self.source,
            mask=mask,
            vertices=np.asarray(self.vertices, dtype=float),
            area=polygon_area(self.vertices),
        )
        self.finished = True
        self.close()
        return self.result

    def close(self) -> None:
        """Disconnect GUI callbacks."""
        for cid in self._connections:
            self._canvas.mpl_disconnect(cid)
        self._connections.clear()


__all__ = [
    "MultiSelectionResult",
    "PolygonSelector",
    "SelectionResult",
    "polygon_area",
    "polygon_mask",
]
